import os
import cv2
import csv
import argparse
import numpy as np
import torch

from dualcam_bev_fpn_model import DualCamBEVFPN
from dualcam_bev_fpn_resnet import DualCamBEVFPNResNet
from dualcam_bev_lss import DualCamBEVLSS

IMG_H, IMG_W = 224, 400
CLASS_NAMES = ["background", "vehicle", "pedestrian", "road", "sidewalk", "lane_marking"]


def read_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_W, IMG_H))  # (IMG_W, IMG_H) set from CLI in main()
    img = img[:, :, ::-1].copy()
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="carla_dualcam_gt_out")
    ap.add_argument("--split", default="test.txt")
    ap.add_argument("--model", default="dualcam_bev_fpn_model.pt")
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--out_csv", default="metrics_fpn.csv")
    ap.add_argument("--arch", default="fpn", choices=["fpn", "resnet", "lss"])
    ap.add_argument("--img_h", type=int, default=224)
    ap.add_argument("--img_w", type=int, default=400)
    args = ap.parse_args()

    global IMG_H, IMG_W
    IMG_H, IMG_W = args.img_h, args.img_w

    device = "cuda" if torch.cuda.is_available() else "cpu"

    front_dir = os.path.join(args.data, "front_rgb")
    rear_dir = os.path.join(args.data, "rear_rgb")
    gt_dir = os.path.join(args.data, "bev_gt")
    with open(os.path.join(args.data, args.split)) as f:
        files = [l.strip() for l in f if l.strip()]

    sample = cv2.imread(os.path.join(gt_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    bev_h, bev_w = sample.shape

    if args.arch == "lss":
        meta = torch.load(args.model, map_location="cpu", weights_only=False)
        ds = meta.get("lss_downsample", 8) if isinstance(meta, dict) else 8
        pool = meta.get("lss_pool", 0.5) if isinstance(meta, dict) else 0.5
        del meta
        model = DualCamBEVLSS.from_calib_file(
            os.path.join(args.data, "calibration.json"),
            num_classes=args.num_classes, bev_h=bev_h, bev_w=bev_w,
            img_h=IMG_H, img_w=IMG_W, downsample=ds, pool_res=pool, pretrained=False,
        ).to(device)
    elif args.arch == "resnet":
        model = DualCamBEVFPNResNet(num_classes=args.num_classes, bev_h=bev_h, bev_w=bev_w, pretrained=False).to(device)
    else:
        model = DualCamBEVFPN(num_classes=args.num_classes, bev_h=bev_h, bev_w=bev_w).to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    C = args.num_classes
    inter = np.zeros(C, dtype=np.int64)
    union = np.zeros(C, dtype=np.int64)
    correct = 0
    total = 0

    with torch.no_grad():
        for name in files:
            front = read_img(os.path.join(front_dir, name))
            rear = read_img(os.path.join(rear_dir, name))
            imgs = torch.stack([front, rear], dim=0).unsqueeze(0).to(device)
            pred = model(imgs)[0]
            pred = torch.argmax(pred, dim=0).cpu().numpy().astype(np.int64)
            gt = cv2.imread(os.path.join(gt_dir, name), cv2.IMREAD_GRAYSCALE).astype(np.int64)

            correct += int((pred == gt).sum())
            total += gt.size
            for c in range(C):
                p = pred == c
                g = gt == c
                inter[c] += int((p & g).sum())
                union[c] += int((p | g).sum())

    iou = inter / np.maximum(union, 1)
    present = union > 0
    miou = float(iou[present].mean())
    pa = correct / total

    print("=" * 52)
    print(f"{'class':12s} {'IoU':>8s}")
    for c in range(C):
        tag = "" if present[c] else "  (absent)"
        print(f"{CLASS_NAMES[c]:12s} {iou[c]:8.4f}{tag}")
    print("-" * 52)
    print(f"{'mIoU':12s} {miou:8.4f}   (over present classes)")
    print(f"{'PixelAcc':12s} {pa:8.4f}")
    print(f"samples: {len(files)}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "IoU"])
        for c in range(C):
            w.writerow([CLASS_NAMES[c], f"{iou[c]:.6f}"])
        w.writerow(["mIoU", f"{miou:.6f}"])
        w.writerow(["PixelAcc", f"{pa:.6f}"])
        w.writerow(["num_samples", len(files)])
    print("saved:", args.out_csv)


if __name__ == "__main__":
    main()
