import os
import cv2
import argparse
import numpy as np
import torch


from dualcam_bev_fpn_model import DualCamBEVFPN

IMG_H, IMG_W = 224, 400

COLOR = {
    0: (0, 0, 0),       # background
    1: (0, 0, 255),     # vehicle
    2: (255, 0, 255),   # pedestrian
    3: (80, 80, 80),    # road
    4: (0, 255, 0),     # sidewalk
    5: (0, 255, 255),   # lane marking (yellow)
}


def read_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img[:, :, ::-1].copy()
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


def colorize(label):
    out = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    for k, c in COLOR.items():
        out[label == k] = c

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="carla_dualcam_gt_out")
    parser.add_argument("--model", default="dualcam_bev_fpn_model.pt")
    parser.add_argument("--out", default="pred_vs_gt.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--split", default="test.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    front_dir = os.path.join(args.data, "front_rgb")
    rear_dir = os.path.join(args.data, "rear_rgb")
    gt_dir = os.path.join(args.data, "bev_gt")
    
    split_path = os.path.join(args.data, args.split)

    if not os.path.exists(split_path):
        raise FileNotFoundError(split_path)

    with open(split_path, "r") as f:
        files = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if len(files) == 0:
        raise RuntimeError("No images found")

    sample_gt = cv2.imread(
        os.path.join(gt_dir, files[0]),
        cv2.IMREAD_GRAYSCALE
    )

    bev_h, bev_w = sample_gt.shape

    model = DualCamBEVFPN(
        num_classes=args.num_classes,
        bev_h=bev_h,
        bev_w=bev_w
    ).to(device)

    ckpt = torch.load(args.model, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    
    
    model.eval()

    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (1280, 720)
    )

    with torch.no_grad():
        for i, name in enumerate(files):
            front_path = os.path.join(front_dir, name)
            rear_path = os.path.join(rear_dir, name)
            gt_path = os.path.join(gt_dir, name)

            front_raw = cv2.imread(front_path)
            rear_raw = cv2.imread(rear_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            front = read_img(front_path)
            rear = read_img(rear_path)

            imgs = torch.stack([front, rear], dim=0)
            imgs = imgs.unsqueeze(0).to(device)

            pred = model(imgs)

            prob = torch.softmax(pred[0], dim=0)
            pred_label = torch.argmax(prob, dim=0).cpu().numpy().astype(np.uint8)

            pred_color = colorize(pred_label)
            gt_color = colorize(gt)

            front_show = cv2.resize(front_raw, (640, 360))
            rear_show = cv2.resize(rear_raw, (640, 360))

            pred_show = cv2.resize(
                pred_color,
                (640, 360),
                interpolation=cv2.INTER_NEAREST
            )

            gt_show = cv2.resize(
                gt_color,
                (640, 360),
                interpolation=cv2.INTER_NEAREST
            )

            cv2.putText(front_show, "Front RGB", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(rear_show, "Rear RGB", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(pred_show, "Prediction", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(gt_show, "Ground Truth", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            top = np.hstack([front_show, rear_show])
            bottom = np.hstack([pred_show, gt_show])
            frame = np.vstack([top, bottom])

            writer.write(frame)

            if i % 50 == 0:
                print(
                    "frame", i,
                    "pred_unique:", np.unique(pred_label),
                    "gt_unique:", np.unique(gt)
                )

    writer.release()
    print("saved:", args.out)


if __name__ == "__main__":
    main()
