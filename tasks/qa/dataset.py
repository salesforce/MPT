import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_metric, load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, BertConfig
from transformers import default_data_collator, EvalPrediction
import numpy as np
import logging

from tasks.qa.utils_qa import postprocess_qa_predictions


def Convert_tydia_to_squad(example):
    new_example = {}
    new_example['id'] = example['paragraphs'][0]['qas'][0]['id']
    new_example['title'] = example['paragraphs'][0]['qas'][0]['id']   # example['paragraphs'][0]['title']
    new_example['context'] = example['paragraphs'][0]['context']
    new_example['question'] = example['paragraphs'][0]['qas'][0]['question']
    new_example['answers'] = {}
    new_example['answers']['text'] = [example['paragraphs'][0]['qas'][0]['answers'][0]['text']]
    new_example['answers']['answer_start'] = [example['paragraphs'][0]['qas'][0]['answers'][0]['answer_start']]
    return new_example



def preprocess_squad_batch(examples, question_column, context_column, answer_column):
        questions = examples[question_column]
        contexts = examples[context_column]
        answers = examples[answer_column]

        def generate_input(_question, _context):
            return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

        inputs = [generate_input(question, context) for question, context in zip(questions, contexts)]
        targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
        return inputs, targets



class SQuAD:

    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args, qa_args) -> None:
        self.data_args = data_args
        self.training_args = training_args
        self.qa_args = qa_args
        self.version_2 = data_args.dataset_name == "squad_v2"

        ## Lifu : add for evalating xquad and mlqa
        if data_args.dataset_name=='xquad':
            raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_name+ '.'+ data_args.lang)
            column_names = raw_datasets['validation'].column_names

        elif data_args.dataset_name=='mlqa':
            ## it is not done yet   :Lifu
            raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_name+ '.'+ data_args.lang + '.'+ data_args.lang)
            column_names = raw_datasets['validation'].column_names
        elif data_args.dataset_name=='tydia':
            #raw_datasets = load_dataset('json', data_files='data/tydiqa/tydiqa-goldp-v1.1-train/tydiqa.'+ data_args.lang + '.train.json', field='data')
            raw_datasets = load_dataset('json', data_files={'train': 'data/tydiqa/tydiqa-goldp-v1.1-train/tydiqa.'+ data_args.lang + '.train.json', 'validation': 'data/tydiqa/tydiqa-goldp-v1.1-dev/tydiqa.'+ data_args.lang + '.dev.json'}, field='data')
            # {'paragraphs': [{'context': 'Quantum field theory naturally began with the study of electromagnetic interactions, as the electromagnetic field was the only known classical field as of the 1920s.[8]:1', 'qas': [{'answers': [{'answer_start': 159, 'text': '1920s'}], 'id': '12', 'question': 'When was quantum field theory developed?'}]}]}
            raw_datasets['train'] = raw_datasets['train'].map(Convert_tydia_to_squad)
            raw_datasets['validation'] = raw_datasets['validation'].map(Convert_tydia_to_squad)
            column_names = raw_datasets['train'].column_names

        else:
            raw_datasets = load_dataset(data_args.dataset_name)
            column_names = raw_datasets['train'].column_names

        self.question_column_name = "question"
        self.context_column_name = "context"
        self.answer_column_name = "answers"

        self.tokenizer = tokenizer

        self.pad_on_right = tokenizer.padding_side == "right" # True
        self.max_seq_len = 384 #data_args.max_seq_length

        if training_args.do_train:
            self.train_dataset = raw_datasets['train']
            self.train_dataset = self.train_dataset.map(
                self.prepare_train_dataset,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on train dataset",
            )
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            if data_args.dataset_name=='mlqa':
                ### test mlqa directly on test for mlqa
                self.eval_examples = raw_datasets['test']
            elif data_args.dataset_name=='tydia':
                self.eval_examples = raw_datasets['validation']
            else:
                self.eval_examples = raw_datasets['validation']

            if data_args.max_eval_samples is not None:
                self.eval_examples = self.eval_examples.select(range(data_args.max_eval_samples))
            self.eval_dataset = self.eval_examples.map(
                self.prepare_eval_dataset,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on validation dataset",
            )
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        self.predict_dataset = None

        self.data_collator = default_data_collator

        ## Lifu: set metric. Always use squad metric
        #if data_args.dataset_name!='xquad':
        #    self.metric = load_metric(data_args.dataset_name)
        #else:
        self.metric = load_metric('squad')

    def prepare_train_dataset(self, examples):
        examples['question'] = [q.lstrip() for q in examples['question']]

        tokenized = self.tokenizer(
            examples['question' if self.pad_on_right else 'context'],
            examples['context' if self.pad_on_right else 'question'],
            truncation='only_second' if self.pad_on_right else 'only_first',
            max_length=self.max_seq_len,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_maping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        tokenized["start_positions"] = []
        tokenized["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized['input_ids'][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            
            sequence_ids = tokenized.sequence_ids(i)
            sample_index = sample_maping[i]
            answers = examples['answers'][sample_index]

            if len(answers['answer_start']) == 0:
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span 
                # (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized["start_positions"].append(cls_index)
                    tokenized["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized["end_positions"].append(token_end_index + 1)
            
        return tokenized

    def prepare_eval_dataset(self, examples):
        # if self.version_2:
        examples['question'] = [q.lstrip() for q in examples['question']]
        
        tokenized = self.tokenizer(
            examples['question' if self.pad_on_right else 'context'],
            examples['context' if self.pad_on_right else 'question'],
            truncation='only_second' if self.pad_on_right else 'only_first',
            max_length=self.max_seq_len,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []

        for i in range(len(tokenized["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized["offset_mapping"][i])
            ]
        return tokenized

    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)

    def post_processing_function(self, examples, features, predictions, stage='eval'):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=self.version_2,
            n_best_size=self.qa_args.n_best_size,
            max_answer_length=self.qa_args.max_answer_length,
            null_score_diff_threshold=self.qa_args.null_score_diff_threshold,
            output_dir=self.training_args.output_dir,
            prefix=stage,
            log_level=logging.INFO
        )
        if self.version_2: # squad_v2
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex['answers']} for ex in examples]
        #print(formatted_predictions[:5], references[:5])
        #from transformers.data.metrics.squad_metrics import squad_evaluate
        #print(squad_evaluate(formatted_predictions, references))
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)






class SQuAD_seq2seq:

    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args, qa_args) -> None:
        self.data_args = data_args
        self.training_args = training_args
        self.qa_args = qa_args
        self.version_2 = data_args.dataset_name == "squad_v2"

        ## Lifu : add for evalating xquad and mlqa
        if data_args.dataset_name=='xquad':
            raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_name+ '.'+ data_args.lang)
            column_names = raw_datasets['validation'].column_names

        elif data_args.dataset_name=='mlqa':
            ## it is not done yet   :Lifu
            raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_name+ '.'+ data_args.lang + '.'+ data_args.lang)
            column_names = raw_datasets['validation'].column_names
        elif data_args.dataset_name=='tydia':
            #raw_datasets = load_dataset('json', data_files='data/tydiqa/tydiqa-goldp-v1.1-train/tydiqa.'+ data_args.lang + '.train.json', field='data')
            raw_datasets = load_dataset('json', data_files={'train': 'data/tydiqa/tydiqa-goldp-v1.1-train/tydiqa.'+ data_args.lang + '.train.json', 'validation': 'data/tydiqa/tydiqa-goldp-v1.1-dev/tydiqa.'+ data_args.lang + '.dev.json'}, field='data')
            # {'paragraphs': [{'context': 'Quantum field theory naturally began with the study of electromagnetic interactions, as the electromagnetic field was the only known classical field as of the 1920s.[8]:1', 'qas': [{'answers': [{'answer_start': 159, 'text': '1920s'}], 'id': '12', 'question': 'When was quantum field theory developed?'}]}]}
            raw_datasets['train'] = raw_datasets['train'].map(Convert_tydia_to_squad)
            raw_datasets['validation'] = raw_datasets['validation'].map(Convert_tydia_to_squad)
            column_names = raw_datasets['train'].column_names

        else:
            raw_datasets = load_dataset(data_args.dataset_name)
            column_names = raw_datasets['train'].column_names

        self.question_column_name = "question"
        self.context_column_name = "context"
        self.answer_column_name = "answers"

        self.tokenizer = tokenizer

        self.pad_on_right = tokenizer.padding_side == "right" # True
        self.max_seq_length = 384 #data_args.max_seq_length

        self.max_answer_length = 30
        self.padding = "max_length"


        if training_args.do_train:
            self.train_dataset = raw_datasets['train']
            self.train_dataset = self.train_dataset.map(
                self.prepare_train_dataset,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on train dataset",
            )
    
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            if data_args.dataset_name=='mlqa':
                ### test mlqa directly on test for mlqa
                self.eval_examples = raw_datasets['test']
            elif data_args.dataset_name=='tydia':
                self.eval_examples = raw_datasets['validation']
            else:
                self.eval_examples = raw_datasets['validation']

            if data_args.max_eval_samples is not None:
                self.eval_examples = self.eval_examples.select(range(data_args.max_eval_samples))
            self.eval_dataset = self.eval_examples.map(
                self.prepare_eval_dataset,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=True,
                desc="Running tokenizer on validation dataset",
            )
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        self.predict_dataset = None

        self.data_collator = default_data_collator

        ## Lifu: set metric. Always use squad metric
        #if data_args.dataset_name!='xquad':
        #    self.metric = load_metric(data_args.dataset_name)
        #else:
        self.metric = load_metric('squad')

    def prepare_train_dataset(self, examples):
        inputs, targets = preprocess_squad_batch(examples, 'question', 'context', 'answers')

        model_inputs = self.tokenizer(inputs, max_length=self.max_seq_length, padding=self.padding, truncation=True)
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_answer_length, padding=self.padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length": # and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def prepare_eval_dataset(self, examples):
        inputs, targets = preprocess_squad_batch(examples, 'question', 'context', 'answers')
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=True,
            #return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_answer_length, padding=self.padding, truncation=True)

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
       
        ## Lifu: for too long sequence; to be considered in the future  
        # sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

        sample_mapping = list(range(len(model_inputs["input_ids"])))
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        model_inputs["example_id"] = []

        for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            model_inputs["example_id"].append(examples["id"][sample_index])

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length": # and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)

    def post_processing_function(self, examples, features, outputs, stage='eval'):

        # Decode the predicted tokens.
        #print(len(outputs))
        #print(outputs[0])
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}
        # Let's loop over all the examples!
        for example_index, example in enumerate(examples):
            # This is the index of the feature associated to the current example.
            feature_index = feature_per_example[example_index]
            predictions[example["id"]] = decoded_preds[feature_index]

        # Format the result to the format the metric expects.
        if self.version_2:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex['answers']} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references) 
