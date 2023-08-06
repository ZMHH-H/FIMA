python3 eval.py \
  --log_dir $PATH_TO_LOG \
  --resume $PATH_TO_FINETUNED_MODEL \
  -a I3D \
  --seed 42 \
  --num_class 101 \
  --lr 0.01 \
  --weight_decay 0.0001 \
  --lr_decay 0.1 \
  -fpc 16 \
  -b 16 \
  -e \
  -j 16 \
  -cs 224 \
  --finetune \
  --epochs 150 \
  --schedule 60 120 \
  --dist_url 'tcp://localhost:10001' --multiprocessing_distributed --world_size 1 --rank 0 \
  $PATH_TO_UCF101