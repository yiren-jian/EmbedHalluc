for task in sst-5 CoLA SST-2 MRPC QQP MNLI QNLI SNLI RTE mr subj trec cr mpqa
do
  for seed in 13 21 42 87 100
  do
      for emb_bs in 4 6 8
      do
          for emb_lr in 1e-6 1e-5 5e-6 1e-6
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
