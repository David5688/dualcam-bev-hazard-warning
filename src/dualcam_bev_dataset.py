import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


# BEV layout (from carla_dualcam_collect_gt.draw_map_bev):
#     row -> ego x (forward), descending;  col -> ego y (right), ascending
#
# The rig admits exactly two geometric symmetries, both checked by the
# assertions in dualcam_bev_lss.py:
#
#   mirror   both cameras sit on the y=0 plane, so mirroring the world about it
#            mirrors each image horizontally and flips the BEV columns
#   swap     the front pose (x=+1.5, yaw 0) is the rear pose (x=-1.5, yaw 180)
#            rotated 180 deg about z, so swapping the two images rotates the
#            BEV by 180 deg (flip rows and columns)
#
# Nothing else is safe: crops, scales and rotations change the effective
# intrinsics, which would silently invalidate the frustum->BEV mapping that
# DualCamBEVLSS precomputes once at init.


class DualCamBEVDataset(Dataset):
    def __init__(self, root, img_size=(224, 400), split=None, augment=False, with_depth=False):
        self.root = root
        self.img_size = img_size
        self.augment = augment
        self.with_depth = with_depth

        self.front_dir = os.path.join(root, "front_rgb")
        self.rear_dir = os.path.join(root, "rear_rgb")
        self.label_dir = os.path.join(root, "bev_gt")
        self.front_depth_dir = os.path.join(root, "front_depth")
        self.rear_depth_dir = os.path.join(root, "rear_depth")

        if split is not None:
            split_path = os.path.join(root, split)
            with open(split_path, "r") as f:
                self.files = [line.strip() for line in f if line.strip()]
        else:
            self.files = sorted([
                f for f in os.listdir(self.front_dir)
                if f.endswith(".png")
            ])

    def __len__(self):
        return len(self.files)

    def read_img(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img = img[:, :, ::-1].copy()
        img = img.astype(np.float32) / 255.0
        return img

    def read_depth(self, path):
        """CARLA raw depth PNG (BGR) -> metres, nearest-resized to img_size."""
        bgr = cv2.imread(path)                       # BGR, uint8
        b = bgr[:, :, 0].astype(np.float32)
        g = bgr[:, :, 1].astype(np.float32)
        r = bgr[:, :, 2].astype(np.float32)
        m = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0) * 1000.0
        return cv2.resize(m, (self.img_size[1], self.img_size[0]),
                          interpolation=cv2.INTER_NEAREST)   # nearest: don't blend depths

    def photometric(self, img, rng):
        """Per-camera colour jitter. Leaves geometry untouched."""
        if rng.random() < 0.8:
            img = img * rng.uniform(0.7, 1.3)                    # brightness
        if rng.random() < 0.8:
            m = img.mean()
            img = (img - m) * rng.uniform(0.7, 1.3) + m          # contrast
        if rng.random() < 0.5:
            g = img.mean(axis=2, keepdims=True)
            img = g + (img - g) * rng.uniform(0.6, 1.4)          # saturation
        if rng.random() < 0.3:
            img = img + rng.normal(0.0, rng.uniform(0.005, 0.03), img.shape)
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx):
        name = self.files[idx]

        front = self.read_img(os.path.join(self.front_dir, name))
        rear = self.read_img(os.path.join(self.rear_dir, name))
        label = cv2.imread(
            os.path.join(self.label_dir, name),
            cv2.IMREAD_GRAYSCALE
        )

        fdepth = rdepth = None
        if self.with_depth:
            fdepth = self.read_depth(os.path.join(self.front_depth_dir, name))
            rdepth = self.read_depth(os.path.join(self.rear_depth_dir, name))

        if self.augment:
            rng = np.random.default_rng()

            if rng.random() < 0.5:                 # swap: rotate world 180 deg about z
                front, rear = rear, front
                label = label[::-1, ::-1]
                if self.with_depth:
                    fdepth, rdepth = rdepth, fdepth   # depth follows its camera

            if rng.random() < 0.5:                 # mirror: reflect world about y=0
                front = front[:, ::-1]
                rear = rear[:, ::-1]
                label = label[:, ::-1]
                if self.with_depth:                   # horizontal flip; depth values unchanged
                    fdepth = fdepth[:, ::-1]
                    rdepth = rdepth[:, ::-1]

            front = self.photometric(np.ascontiguousarray(front), rng)
            rear = self.photometric(np.ascontiguousarray(rear), rng)

        front = torch.from_numpy(np.ascontiguousarray(front)).permute(2, 0, 1)
        rear = torch.from_numpy(np.ascontiguousarray(rear)).permute(2, 0, 1)
        label = torch.from_numpy(np.ascontiguousarray(label).astype(np.int64))

        imgs = torch.stack([front, rear], dim=0)

        if self.with_depth:
            fd = torch.from_numpy(np.ascontiguousarray(fdepth).astype(np.float32))
            rd = torch.from_numpy(np.ascontiguousarray(rdepth).astype(np.float32))
            depth = torch.stack([fd, rd], dim=0)      # (2, Himg, Wimg)
            return imgs, label, depth

        return imgs, label
