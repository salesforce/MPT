# MPT
Code to train model from "[Prompt-Tuning Can Be Much Better Than Fine-Tuning on Cross-lingual Understanding With Multilingual Language Models](https://arxiv.org/pdf/2210.12360.pdf)", accept by EMNLP 2022.

## install

After installing pytorch, the following command is used to install other dependencies:
```
pip install -r requirements.txt
```

## Script Examples for Prompt-Tuning
Following are some scripts that can be used to replicate experiments. It is better to have a look these scripts.

### XNLI
* **prompt tuning Train**
```
sh run_script/run_mnli_xlm_robert.sh xlm-roberta-large 0 0.001 32 1
```
* **Corss-Lingual Evluation, prefix length  32**
```
sh run_script/run_xnli_evalonly.sh checkpoints/mnli-xlm-roberta-large-32-0.001/best_checkpoint
```


### Xquad
* **prompt tuning Train**
```
sh run_script/run_squad_xlm_roberta.sh xlm-roberta-large 0.005 0  16 1
```

* **Corss-Lingual Evluation**
```
sh run_script/run_xquad_evalonly.sh checkpoints/squad/xlm-checkpoint-112000
```



### pawsx
* **prompt tuning Train**
```
sh run_script/run_pawsx_xlm_robert.sh xlm-roberta-large 0 0.005 1  16
```

* **Corss-Lingual Evluation**
```
sh run_script/run_pawsx_evalonly.sh checkpoints/paws-x-xlm-roberta-large-16-0.005/best_checkpoint/
```

### Udpos
* **data**
The processed Universal Dependencies 2.2 data is also provided in the folder `data/udpos`. Please cite the original dataset paper if you use it.
```
@article{nivre2018universal,
  title={Universal Dependencies 2.2},
  author={Nivre, Joakim and Abrams, Mitchell and Agi{\'c}, {\v{Z}}eljko and Ahrenberg, Lars and Antonsen, Lene and Aranzabe, Maria Jesus and Arutie, Gashaw and Asahara, Masayuki and Ateyah, Luma and Attia, Mohammed and others},
  year={2018}
}
```

* **prompt tuning train**
```
sh run_script/run_pos_xlm.sh xlm-roberta-large 0 0.01 16 1
```

* **Corss-Lingual Evluation**
```
bash run_script/run_pos_evalonly.sh checkpoints/udpos-xlm-roberta-large-16-0.01/best_checkpoint/
```

## Script Examples for Prompt-Tuning

* **Fine-Tuning**

`run_mnli_xlm_ft.sh`, `run_pawsx_xlm_robert_ft.sh`, `run_pos_xlm_ft.sh`, `run_squad_xlm_roberta_ft.sh`

* **Evaluatioin**

`run_xnli_ft_evalonly.sh`, `run_pawsx_ft_evalonly.sh`, `run_pos_ft_evalonly.sh`, `run_xquad_ft_evalonly.sh` , `run_tydia_ft_evalonly.sh`


## License
Our code is BSD-3 licensed. See LICENSE.txt for details.

