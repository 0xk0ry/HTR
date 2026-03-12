# Subtractive ablation on IAM
# Start from the full model and remove one component at a time.
# Backbone choices:
#   htr_convtext = MVP + ConvTextBlock  (full model)
#   htr_vt       = MVP + plain ViT Block (removes ConvText block)
#   resnet18     = ResNet18 + plain ViT Block (removes MVP block)

BASE_ARGS="--dataset iam --use-masking \
  --nb-cls 80 \
  --data-path ./data/iam/lines/ \
  --train-data-list ./data/iam/train.ln \
  --val-data-list   ./data/iam/val.ln \
  --test-data-list  ./data/iam/test.ln \
  --max-lr 1e-3 --warm-up-iter 1000 --weight-decay 0.05 \
  --train-bs 32 --val-bs 8 --total-iter 100001 \
  --mask-ratio 0.4 --max-span-length 8 --attn-mask-ratio 0.1 \
  --img-size 512 64 --proj 8 --alpha 1 --proba 0.5"

# (A) Full model — baseline (MVP + ConvText + TCM)
python train.py $BASE_ARGS \
  --backbone htr_convtext --tcm-enable \
  --exp-name "ablation_full"

# (B) w/o TCM — MVP + ConvText, no textual context module
python train.py $BASE_ARGS \
  --backbone htr_convtext \
  --exp-name "ablation_no_tcm"

# (C) w/o ConvText Block — MVP patch embed + plain ViT encoder (htr_vt)
python train.py $BASE_ARGS \
  --backbone htr_vt \
  --exp-name "ablation_no_convtext"

# (D) w/o MVP Block — ResNet18 patch embed + plain ViT encoder
python train.py $BASE_ARGS \
  --backbone resnet18 \
  --exp-name "ablation_no_mvp"
