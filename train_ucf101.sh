python3 train.py \
  --log_dir $PATH_TO_LOG_DIR \
  --ckp_dir $PATH_TO_CKP_DIR \
  -a I3D \
  --dataset ucf101 \
  --lr 0.00375 \
  -fpc 16 \
  -cs 224 \
  -b 24 \
  -j 16 \
  --cos \
  --epochs 200 \
  --pos_ratio 0.7 \
  --dist_url 'tcp://localhost:10001' --multiprocessing_distributed --world_size 1 --rank 0 \
  $PATH_TO_UCF101