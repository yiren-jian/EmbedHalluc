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
