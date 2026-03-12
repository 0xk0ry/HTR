# (A) Full model — baseline (TCM + masking + mixed-mask schedule)
python train.py --dataset iam --tcm-enable --use-masking \
  --exp-name "ablation_full" --nb-cls 80 \
  --data-path ./data/iam/lines/ \
  --train-data-list ./data/iam/train.ln --val-data-list ./data/iam/val.ln \
  --test-data-list ./data/iam/test.ln \
  --max-lr 1e-3 --warm-up-iter 1000 --weight-decay 0.05 \
  --train-bs 32 --val-bs 8 --total-iter 100001 \
  --mask-ratio 0.4 --max-span-length 8 --attn-mask-ratio 0.1 \
  --img-size 512 64 --proj 8 --alpha 1 --proba 0.5

# (B) w/o TCM  — drop --tcm-enable
python train.py --dataset iam --use-masking \
  --exp-name "ablation_no_tcm" --nb-cls 80 \
  --data-path ./data/iam/lines/ \
  --train-data-list ./data/iam/train.ln --val-data-list ./data/iam/val.ln \
  --test-data-list ./data/iam/test.ln \
  --max-lr 1e-3 --warm-up-iter 1000 --weight-decay 0.05 \
  --train-bs 32 --val-bs 8 --total-iter 100001 \
  --mask-ratio 0.4 --max-span-length 8 --attn-mask-ratio 0.1 \
  --img-size 512 64 --proj 8 --alpha 1 --proba 0.5

# (C) w/o masking — drop --use-masking
python train.py --dataset iam --tcm-enable \
  --exp-name "ablation_no_masking" --nb-cls 80 \
  --data-path ./data/iam/lines/ \
  --train-data-list ./data/iam/train.ln --val-data-list ./data/iam/val.ln \
  --test-data-list ./data/iam/test.ln \
  --max-lr 1e-3 --warm-up-iter 1000 --weight-decay 0.05 \
  --train-bs 32 --val-bs 8 --total-iter 100001 \
  --attn-mask-ratio 0.1 \
  --img-size 512 64 --proj 8 --alpha 1 --proba 0.5

# (D) w/o TCM and w/o masking — CTC-only backbone
python train.py --dataset iam \
  --exp-name "ablation_ctc_only" --nb-cls 80 \
  --data-path ./data/iam/lines/ \
  --train-data-list ./data/iam/train.ln --val-data-list ./data/iam/val.ln \
  --test-data-list ./data/iam/test.ln \
  --max-lr 1e-3 --warm-up-iter 1000 --weight-decay 0.05 \
  --train-bs 32 --val-bs 8 --total-iter 100001 \
  --img-size 512 64 --proj 8 --alpha 1 --proba 0.5

# (E) w/o mixed masking — span-only masking (disable random+block branches)
python train.py --dataset iam --tcm-enable --use-masking \
  --r-rand 0.0 --r-block 0.0 --r-span 1.0 \
  --exp-name "ablation_span_mask_only" --nb-cls 80 \
  --data-path ./data/iam/lines/ \
  --train-data-list ./data/iam/train.ln --val-data-list ./data/iam/val.ln \
  --test-data-list ./data/iam/test.ln \
  --max-lr 1e-3 --warm-up-iter 1000 --weight-decay 0.05 \
  --train-bs 32 --val-bs 8 --total-iter 100001 \
  --mask-ratio 0.4 --max-span-length 8 --attn-mask-ratio 0.1 \
  --img-size 512 64 --proj 8 --alpha 1 --proba 0.5

# (F) w/o attention masking — drop --attn-mask-ratio
python train.py --dataset iam --tcm-enable --use-masking \
  --attn-mask-ratio 0.0 \
  --exp-name "ablation_no_attn_mask" --nb-cls 80 \
  --data-path ./data/iam/lines/ \
  --train-data-list ./data/iam/train.ln --val-data-list ./data/iam/val.ln \
  --test-data-list ./data/iam/test.ln \
  --max-lr 1e-3 --warm-up-iter 1000 --weight-decay 0.05 \
  --train-bs 32 --val-bs 8 --total-iter 100001 \
  --mask-ratio 0.4 --max-span-length 8 \
  --img-size 512 64 --proj 8 --alpha 1 --proba 0.5