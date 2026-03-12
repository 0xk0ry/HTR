# Ablation: effect of loss weights (λ_CTC, λ_TCM) on IAM
# All other hyper-parameters are kept identical to the IAM baseline.

BASE_ARGS="--dataset iam --tcm-enable \
  --nb-cls 80 \
  --data-path ./data/iam/lines/ \
  --train-data-list ./data/iam/train.ln \
  --val-data-list   ./data/iam/val.ln \
  --test-data-list  ./data/iam/test.ln \
  --max-lr 1e-3 --warm-up-iter 1000 --weight-decay 0.05 \
  --train-bs 32 --val-bs 8 --total-iter 100001 \
  --mask-ratio 0.4 --max-span-length 8 --attn-mask-ratio 0.1 \
  --img-size 512 64 --proj 8 --alpha 1 --proba 0.5"

# λ_CTC=0.1, λ_TCM=0.5
python train.py $BASE_ARGS \
  --ctc-lambda 0.1 --tcm-lambda 0.5 \
  --exp-name "loss_lctc0.1_ltcm0.5"

# λ_CTC=0.1, λ_TCM=1.0  (default / paper setting)
python train.py $BASE_ARGS \
  --ctc-lambda 0.1 --tcm-lambda 1.0 \
  --exp-name "loss_lctc0.1_ltcm1.0"

# λ_CTC=0.1, λ_TCM=2.0
python train.py $BASE_ARGS \
  --ctc-lambda 0.1 --tcm-lambda 2.0 \
  --exp-name "loss_lctc0.1_ltcm2.0"

# λ_CTC=0.5, λ_TCM=1.0
python train.py $BASE_ARGS \
  --ctc-lambda 0.5 --tcm-lambda 1.0 \
  --exp-name "loss_lctc0.5_ltcm1.0"

# λ_CTC=1.0, λ_TCM=1.0  (equal weight)
python train.py $BASE_ARGS \
  --ctc-lambda 1.0 --tcm-lambda 1.0 \
  --exp-name "loss_lctc1.0_ltcm1.0"