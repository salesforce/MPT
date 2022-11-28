import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)


from transformers import MT5ForConditionalGeneration, T5Tokenizer, PreTrainedTokenizerFast, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

from tasks.qa.dataset import SQuAD, SQuAD_seq2seq

from training.trainer_qa import QuestionAnsweringTrainer
from training.trainer_seq2seq_qa import QuestionAnsweringSeq2seqTrainer

from model.utils import get_model, TaskType

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, qa_args = args

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2,
        revision=model_args.model_revision,
    )

    # corrected offset mappings
    from transformers import XLMRobertaTokenizerFast
    #tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base", from_slow=True)

    if 'xl' in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True,
        )
    else:
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base", from_slow=True)
    # 

    """
    ## this could give you wrong offset mapping ( exclude space for a word with space) -Lifu
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True,
    )
    """
    if model_args.prefix or model_args.prompt:
        model = get_model(model_args, TaskType.QUESTION_ANSWERING, config, fix_bert=True)

        dataset = SQuAD(tokenizer, data_args, training_args, qa_args)

    else:
        if 'mt5' in model_args.model_name_or_path:
            # add later -Lifu
            print('mt5')
            training_args.generation_max_length = 30
            training_args.predict_with_generate = True
            #training_args.generation_num_beams = 5
            #model = MT5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
            #tokenizer = PreTrainedTokenizerFast.from_pretrained(model_args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
            dataset = SQuAD_seq2seq(tokenizer, data_args, training_args, qa_args)
           
        else:
            model = get_model(model_args, TaskType.QUESTION_ANSWERING, config, fix_bert=False)

            dataset = SQuAD(tokenizer, data_args, training_args, qa_args)

    if 'mt5' in model_args.model_name_or_path:

        data_collator = DataCollatorForSeq2Seq(
          tokenizer,
          model=model,
          label_pad_token_id=-100,
          pad_to_multiple_of=8 if training_args.fp16 else None,
        )
        
        trainer = QuestionAnsweringSeq2seqTrainer(
          model=model,
          args=training_args,
          train_dataset=dataset.train_dataset if training_args.do_train else None,
          eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
          eval_examples=dataset.eval_examples if training_args.do_eval else None,
          tokenizer=tokenizer,
          data_collator=data_collator,
          post_process_function=dataset.post_processing_function,
          compute_metrics=dataset.compute_metrics,
        ) 

    else:

        trainer = QuestionAnsweringTrainer(
          model=model,
          args=training_args,
          train_dataset=dataset.train_dataset if training_args.do_train else None,
          eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
          eval_examples=dataset.eval_examples if training_args.do_eval else None,
          tokenizer=tokenizer,
          data_collator=dataset.data_collator,
          post_process_function=dataset.post_processing_function,
          compute_metrics=dataset.compute_metrics,
        )

    return trainer, dataset.predict_dataset


