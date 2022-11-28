import logging
import os
from typing import Dict, OrderedDict

from transformers import Trainer

logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)

class BaseTrainer(Trainer):
    def __init__(self, *args, predict_dataset = None, test_key = "accuracy", **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)


    def compute_hiddens(self, model_args, data_args):
        eval_dataloader = self.get_eval_dataloader()

        #self.callback_handler.eval_dataloader = eval_dataloader
        # Do this before wrapping.
        #eval_dataset = getattr(eval_dataloader, "dataset", None)
        import numpy as np
        model = self._wrap_model(self.model, training=False)
         
        model.eval()
        R = {}
        r = []
        l= []
        for step, inputs in enumerate(eval_dataloader):
            #model.config.use_return_dict=False
            #print(inputs.keys())
            #inputs['return_dict']=True
            #print(inputs['labels'])
            l.append(inputs['labels'])
            # move into cuda            
            inputs = self._prepare_inputs(inputs)

            if model_args.prefix:
                #print('prefix')
                outputs = model.hiddens(**inputs)
            else:
                inputs.pop("labels")
                outputs = model.roberta(**inputs)
                #print(outputs[0][:,0,:].size())
                outputs = outputs[0][:,0,:]
            ## for fine-tuning model
            #print(outputs.size())
            r.append(outputs.cpu().detach().numpy())

        r = np.concatenate(r, axis=0)
        l = np.concatenate(l, axis=0)

        R['input'] = r
        R['label'] = l

        import pickle
        checkpointname = data_args.dataset_name
        #checkpointname = 'mnli'
        if model_args.prefix:
             f = open( 'pt50'+ checkpointname + data_args.lang+ ".pkl","wb")
        else:
             f = open( 'ft1'+checkpointname + data_args.lang+ ".pkl","wb")

        # write the python object (dict) to pickle file
        pickle.dump(R,f)
        print(r.shape)
        f.close()
        return outputs 

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}


            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if eval_metrics["eval_"+self.test_key] > self.best_metrics["best_eval_"+self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_"+self.test_key] = eval_metrics["eval_"+self.test_key]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics["test_"+self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_"+self.test_key] = test_metrics["test_"+self.test_key]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
