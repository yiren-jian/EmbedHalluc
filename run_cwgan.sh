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
