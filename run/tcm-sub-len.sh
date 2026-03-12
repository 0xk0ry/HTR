# Ablation: effect of --tcm-sub-len on IAM
# Sweep over context sub-string lengths: 3, 5, 7, 9, 11
# All other hyper-parameters are kept identical to the IAM baseline.

BASE_ARGS="--dataset iam --tcm-enable --use-masking \
  --nb-cls 80 \
  --data-path /kaggle/input/iam-vt-lines/lines/ \
  --train-data-list /kaggle/input/iam-vt-lines/train.ln \
  --val-data-list   /kaggle/input/iam-vt-lines/val.ln \
  --test-data-list  /kaggle/input/iam-vt-lines/test.ln \
  --max-lr 1e-3 --warm-up-iter 1000 --weight-decay 0.05 \
  --train-bs 32 --val-bs 8 --total-iter 100001 \
  --mask-ratio 0.4 --max-span-length 8 --attn-mask-ratio 0.1 \
  --img-size 512 64 --proj 8 --alpha 1 --proba 0.5"

# tcm-sub-len = 3
python train.py $BASE_ARGS \
  --tcm-sub-len 3 \
  --exp-name "tcm_sub_len_3"

# tcm-sub-len = 5  (default)
python train.py $BASE_ARGS \
  --tcm-sub-len 5 \
  --exp-name "tcm_sub_len_5"

# tcm-sub-len = 7
python train.py $BASE_ARGS \
  --tcm-sub-len 7 \
  --exp-name "tcm_sub_len_7"

# tcm-sub-len = 9
python train.py $BASE_ARGS \
  --tcm-sub-len 9 \
  --exp-name "tcm_sub_len_9"

# tcm-sub-len = 11
python train.py $BASE_ARGS \
  --tcm-sub-len 11 \
  --exp-name "tcm_sub_len_11"
