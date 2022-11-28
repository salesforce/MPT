from tasks.glue.dataset import task_to_keys as glue_tasks
from tasks.superglue.dataset import task_to_keys as superglue_tasks

GLUE_DATASETS = list(glue_tasks.keys())
SUPERGLUE_DATASETS = list(superglue_tasks.keys())
NER_DATASETS = ["conll2003", "conll2004", "ontonotes", "wikiann"]
SRL_DATASETS = ["conll2005", "conll2012"]
QA_DATASETS = ["squad", "squad_v2", 'xquad', "mlqa", "tydia"]
POS_DATASETS = ["udpos"]

TASKS = ["glue", "superglue", "ner", "srl", "qa", "pos"]

DATASETS = GLUE_DATASETS + SUPERGLUE_DATASETS + NER_DATASETS + SRL_DATASETS + QA_DATASETS + POS_DATASETS

ADD_PREFIX_SPACE = {
    'bert': False,
    'roberta': True,
    'xlm-roberta':True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': True,
}

USE_FAST = {
    'bert': True,
    'roberta': True,
    'xlm-roberta':True,
    'deberta': True,
    'gpt2': True,
    'deberta-v2': False,
}
