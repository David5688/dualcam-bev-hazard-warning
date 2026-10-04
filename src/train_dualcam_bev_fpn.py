import os
import csv
import json
import hashlib
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from dualcam_bev_dataset import DualCamBEVDataset
from dualcam_bev_fpn_model import DualCamBEVFPN
from dualcam_bev_fpn_resnet import DualCamBEVFPNResNet
from dualcam_bev_lss import DualCamBEVLSS


def build_model(arch, num_classes, bev_h, bev_w, data_dir=None, img_h=224, img_w=400,
                lss_downsample=8, lss_pool=0.5):
    if arch == "lss":
        return DualCamBEVLSS.from_calib_file(
            os.path.join(data_dir, "calibration.json"),
            num_classes=num_classes, bev_h=bev_h, bev_w=bev_w,
            img_h=img_h, img_w=img_w, downsample=lss_downsample, pool_res=lss_pool,
            pretrained=True,
        )
    if arch == "resnet":
        return DualCamBEVFPNResNet(num_classes=num_classes, bev_h=bev_h, bev_w=bev_w, pretrained=True)
    return DualCamBEVFPN(num_classes=num_classes, bev_h=bev_h, bev_w=bev_w)


def dice_loss(logits, target, num_classes, eps=1.0):
    """Soft multi-class Dice loss (directly optimizes IoU-like overlap)."""
    probs = F.softmax(logits, dim=1)
    target_1hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * target_1hot).sum(dims)
    card = probs.sum(dims) + target_1hot.sum(dims)
    dice = (2.0 * inter + eps) / (card + eps)
    return 1.0 - dice.mean()


class ComboLoss(nn.Module):
    """weighted CE + optional Dice term."""
    def __init__(self, ce_weight, dice_w, num_classes):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_w = dice_w
        self.num_classes = num_classes

    def forward(self, logits, target):
        loss = self.ce(logits, target)
        if self.dice_w > 0:
            loss = loss + self.dice_w * dice_loss(logits, target, self.num_classes)
        return loss


CLASS_NAMES = [
    "Background",
    "Vehicle",
    "Pedestrian",
    "Road",
    "Sidewalk",
    "LaneMarking",
]


def compute_class_weights(dataset, num_classes, device, cache_path=None):
    # The histogram is a property of the label files alone -- it does not depend
    # on the model, the schedule, or anything else that varies between runs -- yet
    # it is recomputed from scratch on every launch, single-threaded.  On the v4
    # training set (11,429 frames) that measured 35 minutes before the first epoch
    # even started.  Cache it, keyed on the exact frame list so a changed split
    # can never silently reuse the wrong weights.
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            if cached.get("key") == _weights_cache_key(dataset, num_classes):
                hist = torch.tensor(cached["hist"], dtype=torch.float32)
                print("class weights: reusing cached histogram (%s)" % cache_path)
                return _weights_from_hist(hist, num_classes, device)
            print("class-weight cache is for a different frame set, recomputing")
        except Exception as exc:
            print("class-weight cache unreadable (%s), recomputing" % exc)

    hist = torch.zeros(num_classes)

    print("Calculating class weights from train set...")

    for batch in dataset:
        label = batch[1]                    # (imgs, label) or (imgs, label, depth)
        hist += torch.bincount(
            label.view(-1),
            minlength=num_classes
        ).float()

    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump({"key": _weights_cache_key(dataset, num_classes),
                           "hist": hist.tolist()}, fh)
            print("class weights: cached to %s" % cache_path)
        except Exception as exc:
            print("could not write class-weight cache (%s)" % exc)

    return _weights_from_hist(hist, num_classes, device)


def _weights_cache_key(dataset, num_classes):
    """Identify the exact set of label files behind a (possibly Concat) dataset."""
    subsets = getattr(dataset, "datasets", [dataset])
    parts = []
    for s in subsets:
        names = getattr(s, "names", None) or getattr(s, "files", None) or []
        parts.append("%s:%d:%s" % (getattr(s, "root", "?"), len(names),
                                   hashlib.sha1(
                                       "|".join(map(str, names)).encode("utf-8")
                                   ).hexdigest()[:16]))
    return "n%d|%s" % (num_classes, ";".join(parts))


def _weights_from_hist(hist, num_classes, device):
    freq = hist / hist.sum()

    weights = 1.0 / torch.sqrt(freq + 1e-6)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, 0.2, 5.0)

    print("pixel histogram:", hist.tolist())
    print("pixel ratio:", freq.tolist())
    print("class weights:", weights.tolist())

    return weights.to(device)


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.matrix[:] = 0

    def update(self, pred, target):
        """
        pred:   [B,H,W]
        target: [B,H,W]
        """

        pred = pred.detach().cpu().numpy().astype(np.int64)
        target = target.detach().cpu().numpy().astype(np.int64)

        mask = (target >= 0) & (target < self.num_classes)

        hist = np.bincount(
            self.num_classes * target[mask] + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.matrix += hist

    def compute(self):
        cm = self.matrix.astype(np.float64)

        tp = np.diag(cm)
        gt = cm.sum(axis=1)
        pred = cm.sum(axis=0)
        total = cm.sum()

        union = gt + pred - tp

        iou = tp / np.maximum(union, 1.0)
        acc = tp / np.maximum(gt, 1.0)

        mean_iou = np.mean(iou)
        pixel_acc = tp.sum() / np.maximum(total, 1.0)
        mean_acc = np.mean(acc)

        freq = gt / np.maximum(total, 1.0)
        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()

        return {
            "iou": iou,
            "acc": acc,
            "mIoU": mean_iou,
            "PA": pixel_acc,
            "mAcc": mean_acc,
            "FWIoU": fwiou,
            "confusion_matrix": cm,
        }


def depth_loss_fn(model, depth_logits, depth_m):
    """BEVDepth supervision: cross-entropy of predicted depth distribution vs the
    true depth bin, averaged over feature cells with in-range depth."""
    bins, valid = model.depth_to_bins(depth_m)              # (B*N, fH, fW)
    ce = F.cross_entropy(depth_logits, bins, reduction="none")
    denom = valid.sum().clamp(min=1)
    return (ce * valid).sum() / denom


def train_one_epoch(model, loader, criterion, optimizer, device, depth_weight=0.0):
    model.train()
    total_loss = 0.0

    for batch in loader:
        if depth_weight > 0:
            imgs, label, depth = batch
            depth = depth.to(device, non_blocking=True)
        else:
            imgs, label = batch
        imgs = imgs.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        if depth_weight > 0:
            pred, depth_logits = model(imgs, return_depth=True)
            loss = criterion(pred, label) + depth_weight * depth_loss_fn(model, depth_logits, depth)
        else:
            pred = model(imgs)
            loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()

    total_loss = 0.0
    cm = ConfusionMatrix(num_classes)

    for imgs, label in loader:
        imgs = imgs.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        pred = model(imgs)
        loss = criterion(pred, label)

        total_loss += loss.item()

        pred_label = pred.argmax(dim=1)
        cm.update(pred_label, label)

    metrics = cm.compute()
    val_loss = total_loss / max(len(loader), 1)

    return val_loss, metrics


def write_csv_header(path, num_classes):
    header = [
        "epoch",
        "train_loss",
        "val_loss",
    ]

    for name in CLASS_NAMES[:num_classes]:
        header.append(f"{name}_IoU")

    header += [
        "Mean_IoU",
        "Pixel_Accuracy",
        "Mean_Accuracy",
        "FWIoU",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv(path, epoch, train_loss, val_loss, metrics, num_classes):
    row = [
        epoch,
        train_loss,
        val_loss,
    ]

    for i in range(num_classes):
        row.append(metrics["iou"][i])

    row += [
        metrics["mIoU"],
        metrics["PA"],
        metrics["mAcc"],
        metrics["FWIoU"],
    ]

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def print_metrics(epoch, train_loss, val_loss, metrics, best_miou):
    iou = metrics["iou"]

    print("=" * 70)
    print(f"Epoch {epoch:03d}")
    print()
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Val Loss   : {val_loss:.4f}")
    print()

    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:14s} IoU : {iou[i]:.4f}")

    print()
    print(f"Mean IoU       : {metrics['mIoU']:.4f}")
    print(f"Pixel Accuracy : {metrics['PA']:.4f}")
    print(f"Mean Accuracy  : {metrics['mAcc']:.4f}")
    print(f"FWIoU          : {metrics['FWIoU']:.4f}")
    print()
    print(f"Best mIoU      : {best_miou:.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="carla_dualcam_gt_out")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save", default="dualcam_bev_fpn_best.pt")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weights_cache", default="../metrics/class_weights_cache.json",
                        help="where to cache the label histogram the class weights are "
                             "derived from. It depends only on the label files, but was "
                             "being recomputed single-threaded on every launch -- 35 min "
                             "before the first v4 epoch could start. Keyed on the exact "
                             "frame list, so changing the split invalidates it. Pass an "
                             "empty string to disable.")
    parser.add_argument("--metrics_csv", default="metrics.csv")
    parser.add_argument("--no_weight", action="store_true")
    parser.add_argument("--dice", type=float, default=0.0,
                        help="weight of Dice term added to CE (0 = off)")
    parser.add_argument("--cosine", action="store_true",
                        help="use CosineAnnealingLR schedule")
    parser.add_argument("--boost_vehicle", type=float, default=1.0,
                        help="multiply class-1 (vehicle) loss weight")
    parser.add_argument("--boost_lane", type=float, default=1.0,
                        help="multiply class-5 (lane marking) loss weight")
    parser.add_argument("--boost_road", type=float, default=1.0,
                        help="multiply class-3 (road) loss weight")
    parser.add_argument("--boost_sidewalk", type=float, default=1.0,
                        help="multiply class-4 (sidewalk) loss weight")
    parser.add_argument("--arch", default="fpn", choices=["fpn", "resnet", "lss"],
                        help="fpn = original from-scratch encoder; resnet = pretrained ResNet-18; "
                             "lss = ResNet-18 + lift-splat view transform")
    parser.add_argument("--img_h", type=int, default=224)
    parser.add_argument("--img_w", type=int, default=400)
    parser.add_argument("--augment", action="store_true",
                        help="train-set augmentation: rig-symmetric mirror/swap + colour jitter")
    parser.add_argument("--lss_downsample", type=int, default=8, choices=[4, 8, 16],
                        help="lss camera-encoder output stride (4 = densest frustum sampling)")
    parser.add_argument("--lss_pool", type=float, default=0.5,
                        help="lss BEV pooling cell size in metres")
    parser.add_argument("--towns", default=None,
                        help="comma-separated dataset dirs for multi-town training; "
                             "each must contain the --split_train / --split_val files. "
                             "calibration is taken from the first (all towns share the same rig)")
    parser.add_argument("--val_towns", default=None,
                        help="comma-separated dataset dirs to build the VALIDATION set from, "
                             "when validation lives in a different city than training. The v4 "
                             "collection is a city-level hold-out (train = 5 cities, val = "
                             "Town07, test = Town01), so its training cities contribute no val "
                             "frames at all and --towns alone cannot express the split. "
                             "Defaults to --towns, which is the older same-city behaviour.")
    parser.add_argument("--split_train", default="train.txt")
    parser.add_argument("--split_val", default="val.txt")
    parser.add_argument("--depth_weight", type=float, default=0.0,
                        help="BEVDepth supervision weight on the lss lift step (0 = off); "
                             "requires front_depth/rear_depth in the dataset (collect with --depth)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    want_depth = args.depth_weight > 0 and args.arch == "lss"
    if args.depth_weight > 0 and args.arch != "lss":
        raise SystemExit("--depth_weight only applies to --arch lss")

    if args.towns:
        town_dirs = [t.strip() for t in args.towns.split(",") if t.strip()]
        calib_dir = town_dirs[0]
        train_set = ConcatDataset([
            DualCamBEVDataset(d, img_size=(args.img_h, args.img_w),
                              split=args.split_train, augment=args.augment,
                              with_depth=want_depth)
            for d in town_dirs
        ])
        val_dirs = ([t.strip() for t in args.val_towns.split(",") if t.strip()]
                    if args.val_towns else town_dirs)
        val_set = ConcatDataset([
            DualCamBEVDataset(d, img_size=(args.img_h, args.img_w),
                              split=args.split_val)
            for d in val_dirs
        ])
        print("multi-town train:", ", ".join(os.path.basename(d) for d in town_dirs))
        print("multi-town val  :", ", ".join(os.path.basename(d) for d in val_dirs))
        overlap = set(town_dirs) & set(val_dirs)
        if overlap and args.val_towns:
            print("WARNING: %s appear in both --towns and --val_towns; with the v4 "
                  "city-level hold-out they must be disjoint"
                  % ", ".join(sorted(os.path.basename(d) for d in overlap)))
    else:
        calib_dir = args.data
        train_set = DualCamBEVDataset(
            args.data,
            img_size=(args.img_h, args.img_w),
            split=args.split_train,
            augment=args.augment,
            with_depth=want_depth
        )
        val_set = DualCamBEVDataset(
            args.data,
            img_size=(args.img_h, args.img_w),
            split=args.split_val
        )
    print("depth supervision:", want_depth, "weight" if want_depth else "", args.depth_weight if want_depth else "")

    print("train samples:", len(train_set))
    print("val samples:", len(val_set))

    if len(train_set) == 0:
        raise RuntimeError("train.txt is empty")

    if len(val_set) == 0:
        raise RuntimeError("val.txt is empty")

    sample = train_set[0]                   # (imgs, label) or (imgs, label, depth)
    bev_h, bev_w = sample[1].shape

    print("BEV size:", bev_h, bev_w)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = build_model(args.arch, args.num_classes, bev_h, bev_w,
                        data_dir=calib_dir, img_h=args.img_h, img_w=args.img_w,
                        lss_downsample=args.lss_downsample, lss_pool=args.lss_pool).to(device)
    print("arch:", args.arch)
    print("params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    print("augment:", args.augment)

    if args.no_weight:
        print("Using unweighted loss")
        class_weights = None
    else:
        # mirror/swap only permute pixels and colour jitter never touches labels,
        # so the histogram is augmentation-invariant -- but compute it with
        # augmentation off so the weights match earlier runs exactly.
        subsets = train_set.datasets if isinstance(train_set, ConcatDataset) else [train_set]
        saved_aug = [s.augment for s in subsets]
        saved_wd = [s.with_depth for s in subsets]
        for s in subsets:
            s.augment = False
            s.with_depth = False            # don't read depth maps just to histogram labels
        class_weights = compute_class_weights(
            train_set,
            args.num_classes,
            device,
            cache_path=args.weights_cache
        )
        for s, a, w in zip(subsets, saved_aug, saved_wd):
            s.augment = a
            s.with_depth = w

        if args.boost_vehicle != 1.0 and args.num_classes > 1:
            class_weights[1] *= args.boost_vehicle
        if args.boost_lane != 1.0 and args.num_classes > 5:
            class_weights[5] *= args.boost_lane
        if args.boost_road != 1.0 and args.num_classes > 3:
            class_weights[3] *= args.boost_road
        if args.boost_sidewalk != 1.0 and args.num_classes > 4:
            class_weights[4] *= args.boost_sidewalk
        print("boosted class weights:", class_weights.tolist())

    criterion = ComboLoss(class_weights, args.dice, args.num_classes)
    print(f"loss = CE + {args.dice} * Dice")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = None
    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        print("using CosineAnnealingLR")

    csv_path = os.path.join(
        os.path.dirname(args.save)
        if os.path.dirname(args.save) else ".",
        args.metrics_csv
    )

    write_csv_header(
        csv_path,
        args.num_classes
    )

    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            depth_weight=args.depth_weight if want_depth else 0.0
        )

        val_loss, metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            args.num_classes
        )

        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_miou": best_miou,
                    "num_classes": args.num_classes,
                    "bev_h": bev_h,
                    "bev_w": bev_w,
                    "arch": args.arch,
                    "img_h": args.img_h,
                    "img_w": args.img_w,
                    "lss_downsample": args.lss_downsample,
                    "lss_pool": args.lss_pool,
                    "depth_weight": args.depth_weight,
                },
                args.save
            )

            saved = True
        else:
            saved = False

        append_csv(
            csv_path,
            epoch,
            train_loss,
            val_loss,
            metrics,
            args.num_classes
        )

        print_metrics(
            epoch,
            train_loss,
            val_loss,
            metrics,
            best_miou
        )

        if saved:
            print("saved best model:", args.save)

        if scheduler is not None:
            scheduler.step()

    print("training finished")
    print("best mIoU:", best_miou)
    print("metrics csv:", csv_path)


if __name__ == "__main__":
    main()
