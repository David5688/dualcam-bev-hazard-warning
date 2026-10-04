#!/bin/bash

python infer_dualcam_bev_fpn_gt.py \
  --data carla_dualcam_gt_out \
  --split test.txt \
  --model dualcam_bev_fpn_model.pt \
  --num_classes 5 \
  --out pred_test_fpn.mp4