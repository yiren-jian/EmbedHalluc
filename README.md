# EmbedHalluc for Few-shot Language Learners

This repo covers the implementation of the following paper:  **"Embedding Hallucination for Few-shot Language Learning"** by [Yiren Jian](https://cs.dartmouth.edu/~yirenjian/), [Chongyang Gao](https://gcyzsl.github.io/) and [Soroush Vosoughi](https://www.cs.dartmouth.edu/~soroush/), accepted to NAACL 2022.

<img src="figures/overview.png" width="400">

If you find this repo useful for your research, please consider citing the paper.
```bibtex
@article{Jian2022EmbedHalluc,
    author = {Jian, Yiren and Gao, Chongyang and Vosoughi, Soroush},
    title = {Embedding Hallucination for Few-shot Language Learning},
    journal = {Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
    year = {2022}
}
```


Our code is  heavily borrowed from [LM-BFF](https://github.com/princeton-nlp/LM-BFF).

## Requirements

We tested our code on Nvidia RTX-A6000 and RTX-8000 and Nvidia A100. The requirements of packages are provided in `requirements.txt`. Install packages by
```shell
pip install -r requirements.txt
```

## Download data
We use pre-processed datasets (SST-2, SST-5, MR, CR, MPQA, Subj, TREC, CoLA, MNLI, SNLI, QNLI, RTE, MRPC, QQP) from  [LM-BFF](https://github.com/princeton-nlp/LM-BFF). LM-BFF offers helpful scripts for downloading and preparing the dataset. Simply run the commands below.
```shell
cd data
bash download_dataset.sh
```
Then use the following command to generate 16-shot datasets we used in the study.
```shell
python tools/generate_k_shot_data.py
```

## Collecting Real Embeddings
You will need to first collect real embeddings from the few-shot dataset (for further training of Hallucinator).
```shell
for task in sst-5 #CoLA SST-2 MRPC QQP MNLI QNLI SNLI RTE mr subj trec cr mpqa
do
  for seed in 13 #21 42 87 100
  do
      for bs in 2
      do
          for lr in 1e-5
          do
              TAG=LM-BFF \
              TYPE=finetune \
              TASK=$task \
              BS=$bs \
              LR=$lr \
              SEED=$seed \
              MODEL=roberta-large \
              bash run_collect_embeddings.sh
          done
      done
  done
done
```

## Training of Hallucinator
```shell
for TASK in sst-5 #cola sst-2 mrpc qqp mnli qnli snli rte mr subj trec cr mpqa
do
  for SEED in 13 #21 42 87 100
  do
    python cwgan_main.py \
      --task $TASK \
      --seed $SEED \
      --few_shot_type 'finetune' \
      --model_name_or_path 'roberta-large'
  done
done
```

## Running our fine-tuning
Assuming you have one GPU in you system, we show an example of running our fine-tuning on SST-5.

```shell
for task in sst-5 # CoLA SST-2 MRPC QQP MNLI QNLI SNLI RTE mr subj trec cr mpqa
do
  for seed in 13 #21 42 87 100
  do
      for emb_bs in 4 #6 8
      do
          for emb_lr in 1e-6 #1e-5 5e-6 1e-6
          do
              for kd_temperature in 0.4    #### is only used for LabelCalib
              do
                  TAG=exp \
                  TYPE=finetune \
                  TASK=$task \
                  BS=2 \
                  LR=1e-5 \
                  EMB_BS=$emb_bs \
                  EMB_LR=$emb_lr \
                  KD_TEMPERATURE=$kd_temperature \
                  SEED=$seed \
                  MODEL=roberta-large \
                  bash run_experiment.sh
              done
          done
      done
  done
done
```
Our framework also applies to prompt-based method, i.e., ```TYPE=prompt```.


## Collecting results
```
python tools/gather_result.py --condition "{'tag': 'exp', 'task_name': 'sst-5', 'few_shot_type': 'finetune', 'model_name_or_path': 'roberta-large'}"
```
It will collect the results from ```log``` and compute the mean and standard deviation over those 5 train-test splits.

## Contacts
For any questions, please contact authors.


## Acknowlegements
Thanks to [LM-BFF](https://github.com/princeton-nlp/LM-BFF), for the preliminary implementations.
