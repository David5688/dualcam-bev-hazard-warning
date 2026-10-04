"""Dual-camera BEV segmentation with an LSS (Lift-Splat-Shoot) view transform.

Replaces the geometry-free `nn.Upsample` in dualcam_bev_fpn_resnet.py with a
real image->BEV projection:

  lift   each feature pixel into a frustum of D depth bins, predicting a depth
         distribution over those bins (unsupervised, as in LSS)
  splat  scatter the depth-weighted context features into ego-frame BEV cells
         using the true camera intrinsics K and extrinsics T_ego_cam
  shoot  a BEV encoder decodes the pooled grid into per-class logits

The camera rig is rigidly attached to the ego vehicle and the BEV grid is in
the ego frame, so frustum->BEV cell assignment is CONSTANT across frames.
We precompute it once at init and pooling becomes a fixed scatter_add.

Conventions (all taken from carla_dualcam_collect_gt.py, which produced the GT):
  ego frame     x forward, y right, z up   (CARLA)
  bev_gt[row, col]:  x = X_MIN + (BEV_H - 1 - row) * RES   -> front is row 0
                     y = Y_MIN + col * RES                 -> right is col max
  optical frame x right, y down, z forward
                P_opt = (y_cam, -z_cam, x_cam)

Run `python dualcam_bev_lss.py` to execute the geometry self-tests.
"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# optical -> CARLA camera frame:  x_c = z_o, y_c = x_o, z_c = -y_o
R_CAM_OPT = torch.tensor([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


def yaw_matrix(deg):
    """Rotation about +z by `deg`, matching carla_transform_to_matrix()."""
    t = torch.tensor(deg * torch.pi / 180.0)
    c, s = torch.cos(t), torch.sin(t)
    return torch.tensor([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


class Up(nn.Module):
    """LSS-style upsample-and-fuse block."""

    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))


class CamEncode(nn.Module):
    """ResNet-18 trunk -> (depth distribution over D bins, context of C channels).

    `downsample` selects the output stride: 16 uses the layer3 level, 8 adds a
    second top-down step onto layer2, 4 a third onto layer1. Each step down
    quadruples the number of frustum rays, which matters a lot for how much of
    the BEV grid receives any feature.
    """

    def __init__(self, D, C, downsample=8, pretrained=True):
        super().__init__()
        assert downsample in (4, 8, 16)
        self.D = D
        self.C = C
        self.downsample = downsample
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1   # 64,  stride 4
        self.layer2 = net.layer2   # 128, stride 8
        self.layer3 = net.layer3   # 256, stride 16
        self.layer4 = net.layer4   # 512, stride 32

        self.fuse16 = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        if downsample <= 8:
            self.fuse8 = nn.Sequential(
                nn.Conv2d(384, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        if downsample <= 4:
            self.fuse4 = nn.Sequential(
                nn.Conv2d(320, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        self.head = nn.Conv2d(256, D + C, 1)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        c4 = F.interpolate(c4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse16(torch.cat([c3, c4], dim=1))
        if self.downsample <= 8:
            y = F.interpolate(y, size=c2.shape[-2:], mode="bilinear", align_corners=False)
            y = self.fuse8(torch.cat([c2, y], dim=1))
        if self.downsample <= 4:
            y = F.interpolate(y, size=c1.shape[-2:], mode="bilinear", align_corners=False)
            y = self.fuse4(torch.cat([c1, y], dim=1))
        y = self.head(y)

        depth_logits = y[:, : self.D]                  # (B*N, D, fH, fW), pre-softmax
        depth = depth_logits.softmax(dim=1)
        ctx = y[:, self.D : self.D + self.C]           # (B*N, C, fH, fW)
        feat = depth.unsqueeze(1) * ctx.unsqueeze(2)   # (B*N, C, D, fH, fW)
        return feat, depth_logits


class BevEncode(nn.Module):
    """ResNet-18 style trunk over the pooled BEV grid, upsampled to the GT size."""

    def __init__(self, in_ch, num_classes, out_h, out_w):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        net = resnet18(weights=None)
        self.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = net.bn1
        self.relu = net.relu
        self.layer1 = net.layer1   # 64
        self.layer2 = net.layer2   # 128, /2
        self.layer3 = net.layer3   # 256, /2

        self.up1 = Up(256 + 64, 256, scale=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        x = self.up1(x, x1)
        x = self.up2(x)
        if x.shape[-2:] != (self.out_h, self.out_w):
            x = F.interpolate(x, size=(self.out_h, self.out_w),
                              mode="bilinear", align_corners=False)
        return x


class DualCamBEVLSS(nn.Module):
    def __init__(self, calib, num_classes=6, bev_h=240, bev_w=240,
                 img_h=224, img_w=400, camC=64, pool_res=0.5,
                 dbound=(2.0, 42.0, 0.5), downsample=8, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.img_h, self.img_w = img_h, img_w
        self.camC = camC
        self.downsample = downsample
        self.fH, self.fW = img_h // downsample, img_w // downsample

        bev = calib["bev"]
        self.x_min, self.x_max = float(bev["x_min"]), float(bev["x_max"])
        self.y_min, self.y_max = float(bev["y_min"]), float(bev["y_max"])
        self.pool_res = pool_res
        self.gx = int(round((self.x_max - self.x_min) / pool_res))
        self.gy = int(round((self.y_max - self.y_min) / pool_res))

        d0, d1, dd = dbound
        self.D = int(round((d1 - d0) / dd))
        self.d0, self.d1, self.dd = float(d0), float(d1), float(dd)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.camencode = CamEncode(self.D, camC, downsample=downsample, pretrained=pretrained)
        self.bevencode = BevEncode(camC, num_classes, bev_h, bev_w)

        geom = self._build_geometry(calib, dbound)          # (N, D, fH, fW, 3) ego xyz
        self._precompute_pooling(geom)

    # ---- geometry -------------------------------------------------------

    def _scaled_K(self, K, orig_w, orig_h):
        K = torch.tensor(K, dtype=torch.float32).clone()
        sx = self.img_w / float(orig_w)
        sy = self.img_h / float(orig_h)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return K

    def _frustum(self, dbound):
        d0, d1, dd = dbound
        ds = torch.arange(d0, d1, dd, dtype=torch.float32)
        D = ds.numel()
        ds = ds.view(-1, 1, 1).expand(D, self.fH, self.fW)
        xs = torch.linspace(0, self.img_w - 1, self.fW).view(1, 1, self.fW).expand(D, self.fH, self.fW)
        ys = torch.linspace(0, self.img_h - 1, self.fH).view(1, self.fH, 1).expand(D, self.fH, self.fW)
        return torch.stack((xs, ys, ds), -1)             # (D, fH, fW, 3) = (u, v, depth)

    def _build_geometry(self, calib, dbound):
        orig_w = calib["image"]["width"]
        orig_h = calib["image"]["height"]
        frustum = self._frustum(dbound)

        geoms = []
        for name in ("front", "rear"):
            cam = calib["cameras"][name]
            K = self._scaled_K(cam["K"], orig_w, orig_h)
            T = torch.tensor(cam["T_ego_cam"], dtype=torch.float32)
            R_ego_cam = T[:3, :3]
            t_ego_cam = T[:3, 3]

            # (u*d, v*d, d) -> optical-frame 3D point
            pts = torch.cat([frustum[..., :2] * frustum[..., 2:3], frustum[..., 2:3]], dim=-1)
            pts = pts @ torch.inverse(K).T                       # (D, fH, fW, 3) optical
            R_ego_opt = R_ego_cam @ R_CAM_OPT
            pts = pts @ R_ego_opt.T + t_ego_cam                  # -> ego
            geoms.append(pts)

        return torch.stack(geoms, 0)                             # (N, D, fH, fW, 3)

    def _precompute_pooling(self, geom):
        """Frustum points are fixed w.r.t. ego, so cache which BEV cell each lands in."""
        pts = geom.reshape(-1, 3)
        ix = torch.floor((pts[:, 0] - self.x_min) / self.pool_res).long()
        iy = torch.floor((pts[:, 1] - self.y_min) / self.pool_res).long()
        keep = (ix >= 0) & (ix < self.gx) & (iy >= 0) & (iy < self.gy)

        self.register_buffer("valid_idx", torch.nonzero(keep, as_tuple=True)[0])
        self.register_buffer("bev_idx", (ix[keep] * self.gy + iy[keep]))
        self.n_points = pts.shape[0]
        self.n_valid = int(keep.sum())

    # ---- forward --------------------------------------------------------

    def forward(self, imgs, return_depth=False):
        """imgs: [B, 2, 3, H, W] in [0,1] RGB, index 0 = front, 1 = rear."""
        B, N = imgs.shape[:2]
        x = imgs.flatten(0, 1)
        x = (x - self.mean) / self.std
        x, depth_logits = self.camencode(x)                      # feat (B*N,C,D,fH,fW)
        x = x.view(B, N, self.camC, self.D, self.fH, self.fW)

        # (B, C, N*D*fH*fW)
        x = x.permute(0, 2, 1, 3, 4, 5).reshape(B, self.camC, self.n_points)
        x = x[:, :, self.valid_idx]

        bev = x.new_zeros(B, self.camC, self.gx * self.gy)
        bev.scatter_add_(2, self.bev_idx.view(1, 1, -1).expand(B, self.camC, -1), x)
        bev = bev.view(B, self.camC, self.gx, self.gy)

        # ego x increases forward, but bev_gt row 0 is the front -> flip the x axis
        bev = torch.flip(bev, dims=[2])
        out = self.bevencode(bev)
        if return_depth:
            return out, depth_logits                             # (B*N, D, fH, fW)
        return out

    def depth_to_bins(self, depth_m):
        """Depth maps in metres -> (bin index, valid mask) at feature resolution.

        depth_m: [B, N, Himg, Wimg]. Min-pools each feature cell (foreground wins),
        maps to the D depth bins, masks out cells outside [d0, d1).
        """
        B, N = depth_m.shape[:2]
        d = depth_m.reshape(B * N, 1, depth_m.shape[-2], depth_m.shape[-1])
        # min-pool to feature grid: negate, max-pool, negate back
        d = -F.adaptive_max_pool2d(-d, (self.fH, self.fW)).squeeze(1)   # (B*N, fH, fW)
        bins = torch.floor((d - self.d0) / self.dd).long()
        valid = (d >= self.d0) & (d < self.d1)
        bins = bins.clamp(0, self.D - 1)
        return bins, valid

    @classmethod
    def from_calib_file(cls, path, **kw):
        with open(path) as f:
            calib = json.load(f)
        return cls(calib, **kw)


# ---- self-tests ---------------------------------------------------------

def _self_test():
    calib = {
        "image": {"width": 640, "height": 360, "fov": 90},
        "bev": {"x_min": -30.0, "x_max": 30.0, "y_min": -30.0, "y_max": 30.0,
                "resolution": 0.25, "height": 240, "width": 240},
        "cameras": {
            "front": {"K": [[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1]],
                      "T_ego_cam": torch.eye(4).tolist()},
            "rear": {"K": [[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1]],
                     "T_ego_cam": torch.eye(4).tolist()},
        },
    }
    calib["cameras"]["front"]["T_ego_cam"][0][3] = 1.5
    calib["cameras"]["front"]["T_ego_cam"][2][3] = 1.6
    Rz = yaw_matrix(180.0)
    for r in range(3):
        for c in range(3):
            calib["cameras"]["rear"]["T_ego_cam"][r][c] = float(Rz[r, c])
    calib["cameras"]["rear"]["T_ego_cam"][0][3] = -1.5
    calib["cameras"]["rear"]["T_ego_cam"][2][3] = 1.6

    DBOUND = (2.0, 42.0, 0.5)
    m = DualCamBEVLSS(calib, pretrained=False, dbound=DBOUND, downsample=8)
    assert (m.fH, m.fW) == (28, 50) and m.D == 80, (m.fH, m.fW, m.D)

    # 1. R_CAM_OPT maps optical basis to CARLA camera basis
    assert torch.allclose(R_CAM_OPT @ torch.tensor([0.0, 0.0, 1.0]),
                          torch.tensor([1.0, 0.0, 0.0])), "optical +z must be camera forward"
    assert torch.allclose(R_CAM_OPT @ torch.tensor([1.0, 0.0, 0.0]),
                          torch.tensor([0.0, 1.0, 0.0])), "optical +x must be camera right"
    assert torch.allclose(R_CAM_OPT @ torch.tensor([0.0, 1.0, 0.0]),
                          torch.tensor([0.0, 0.0, -1.0])), "optical +y must be camera down"
    print("ok  R_CAM_OPT basis mapping")

    # 2. K scaling: 640x360 -> 400x224 is anisotropic, so fx != fy
    K = m._scaled_K([[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1]], 640, 360)
    assert abs(K[0, 0] - 200.0) < 1e-4 and abs(K[0, 2] - 200.0) < 1e-4
    assert abs(K[1, 1] - 199.111) < 1e-2 and abs(K[1, 2] - 112.0) < 1e-2
    assert K[0, 0] != K[1, 1]
    print(f"ok  K rescaled  fx={K[0,0]:.3f} fy={K[1,1]:.3f} cx={K[0,2]:.3f} cy={K[1,2]:.3f}")

    # 3. principal ray of each camera lands where physics says it should
    geom = m._build_geometry(calib, DBOUND)               # (N, D, fH, fW, 3)
    # frustum pixel grid is linspace(0, W-1, fW); find the sample closest to the centre
    u = torch.linspace(0, m.img_w - 1, m.fW)
    v = torch.linspace(0, m.img_h - 1, m.fH)
    ui = int(torch.argmin((u - K[0, 2]).abs()))
    vi = int(torch.argmin((v - K[1, 2]).abs()))
    du = float(u[ui] - K[0, 2])
    dv = float(v[vi] - K[1, 2])

    d_idx = 16                                            # depth = 2.0 + 16 * 0.5 = 10.0 m
    depth = DBOUND[0] + d_idx * DBOUND[2]

    pf = geom[0, d_idx, vi, ui]
    exp_f = torch.tensor([1.5 + depth, du / float(K[0, 0]) * depth, 1.6 - dv / float(K[1, 1]) * depth])
    assert torch.allclose(pf, exp_f, atol=1e-3), f"front ray {pf} != {exp_f}"
    print(f"ok  front camera: near-centre ray at {depth:.0f} m -> ego {pf.tolist()}")

    pr = geom[1, d_idx, vi, ui]
    exp_r = torch.tensor([-1.5 - depth, -du / float(K[0, 0]) * depth, 1.6 - dv / float(K[1, 1]) * depth])
    assert torch.allclose(pr, exp_r, atol=1e-3), f"rear ray {pr} != {exp_r}"
    print(f"ok  rear camera:  near-centre ray at {depth:.0f} m -> ego {pr.tolist()}")

    # 4. front camera must live in +x, rear in -x
    assert geom[0, :, :, :, 0].min() > 0, "front frustum leaked behind the ego"
    assert geom[1, :, :, :, 0].max() < 0, "rear frustum leaked in front of the ego"
    print("ok  front frustum in +x, rear frustum in -x")

    # 4b. rig symmetries that license the geometric augmentations.
    #     yaw-180 about z maps the front camera pose exactly onto the rear one,
    #     so swapping the two images and rotating the BEV by 180 deg is exact.
    R180 = torch.tensor([-1.0, -1.0, 1.0])
    assert torch.allclose(geom[1], geom[0] * R180, atol=1e-4), "rot180 rig symmetry broken"
    print("ok  rig symmetry: geom_rear == R180 . geom_front  (front<->rear swap is exact)")

    #     Both cameras sit on the y=0 plane, so mirroring the world about it
    #     mirrors each image horizontally. CARLA sets cx = W/2 = 200 while the
    #     0-based pixel centres span 0..W-1 (midpoint 199.5), so mirroring the u
    #     axis is off by exactly one pixel. The residual is y-only and stays well
    #     inside one pooling cell, so the augmentation is safe.
    MIRROR = torch.tensor([1.0, -1.0, 1.0])
    one_px = 1.0 / float(K[0, 0]) * DBOUND[1]              # worst case, at max depth
    for cam, name in ((0, "front"), (1, "rear")):
        flipped = torch.flip(geom[cam], dims=[2])          # mirror the u axis
        err = (flipped - geom[cam] * MIRROR).abs()
        assert err.max() <= one_px, f"{name} mirror symmetry off by {err.max():.4f} m"
        assert err[..., 0].max() < 1e-5 and err[..., 2].max() < 1e-5, "mirror must only move y"
        resid = float(err.max())
    assert resid < m.pool_res, "mirror residual must stay inside one pooling cell"
    print(f"ok  rig symmetry: horizontal mirror <-> y-flip  (y-only residual {resid*100:.1f} cm "
          f"at {DBOUND[1]:.0f} m, from cx=W/2 vs pixel-centre convention; "
          f"pool cell is {m.pool_res*100:.0f} cm)")

    # 5. BEV row/col convention matches carla_dualcam_collect_gt.draw_map_bev
    #    a point straight ahead must land in a small row (top of the image)
    BEV_H, RES, X_MIN, Y_MIN = 240, 0.25, -30.0, -30.0
    x_ahead, y_ahead = 20.0, 0.0
    row = BEV_H - 1 - int((x_ahead - X_MIN) / RES)
    col = int((y_ahead - Y_MIN) / RES)
    assert row < BEV_H // 2, "ahead should be in the upper half"
    assert abs(col - BEV_H // 2) <= 1, "y=0 should be the horizontal centre"
    # our pooled grid, after flip(dim=x), must agree
    ix = int((x_ahead - X_MIN) / m.pool_res)
    row_coarse = m.gx - 1 - ix
    assert abs(row_coarse / m.gx - row / BEV_H) < 0.01, "coarse/fine row conventions disagree"
    print(f"ok  BEV convention: x={x_ahead} -> row {row}/{BEV_H} (coarse {row_coarse}/{m.gx})")

    # 6. pooling coverage, against the geometric ceiling of this 2-camera rig
    frac = m.n_valid / m.n_points
    print(f"ok  pooling: {m.n_valid}/{m.n_points} frustum points inside the grid ({frac:.1%})")
    cov = torch.zeros(m.gx * m.gy)
    cov.scatter_add_(0, m.bev_idx, torch.ones(m.n_valid))
    hit = (cov > 0).float().mean()

    xs = torch.arange(m.gx) * m.pool_res + m.x_min + m.pool_res / 2
    ys = torch.arange(m.gy) * m.pool_res + m.y_min + m.pool_res / 2
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    d0, d1 = DBOUND[0], DBOUND[1]
    fx_, fy_ = X - 1.5, Y                      # front camera, 90 deg hfov -> |x| >= |y|
    front = (fx_ > 0) & (fx_.abs() >= fy_.abs()) & (fx_.hypot(fy_) >= d0) & (fx_.hypot(fy_) <= d1)
    rx, ry = X + 1.5, Y
    rear = (rx < 0) & (rx.abs() >= ry.abs()) & (rx.hypot(ry) >= d0) & (rx.hypot(ry) <= d1)
    ceiling = (front | rear).float().mean()

    print(f"    {hit:.1%} of the {m.gx}x{m.gy} pooled cells receive >=1 point "
          f"(mean {cov[cov>0].mean():.1f} points per hit cell)")
    print(f"    geometric ceiling for this rig is {ceiling:.1%} "
          f"(2 x 90 deg fov, {d0:.0f}-{d1:.0f} m) -> reaching {hit/ceiling:.0%} of what is visible")
    assert hit <= ceiling + 1e-6, "coverage cannot exceed what the cameras can see"
    assert hit / ceiling > 0.6, f"sampling too sparse: {hit:.1%} of a {ceiling:.1%} ceiling"

    # 7. forward pass shape + gradient flow
    m.train()
    x = torch.rand(1, 2, 3, 224, 400)
    out = m(x)
    assert out.shape == (1, 6, 240, 240), out.shape
    out.sum().backward()
    g = m.camencode.head.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print(f"ok  forward {tuple(out.shape)}, gradient reaches camencode.head")

    # 8. depth head: return_depth shape + depth_to_bins correctness
    out2, depth_logits = m(x, return_depth=True)
    assert out2.shape == (1, 6, 240, 240)
    assert depth_logits.shape == (1 * 2, m.D, m.fH, m.fW), depth_logits.shape
    print(f"ok  return_depth: logits {tuple(depth_logits.shape)}")

    # constant 10 m depth -> bin (10-2)/0.5 = 16, all valid
    dm = torch.full((1, 2, 224, 400), 10.0)
    bins, valid = m.depth_to_bins(dm)
    assert bins.shape == (2, m.fH, m.fW) and valid.all()
    assert (bins == 16).all(), f"10 m should map to bin 16, got {bins.unique().tolist()}"
    # out-of-range depths must be masked out
    near = torch.full((1, 2, 224, 400), 1.0)      # < d0 = 2 m
    far = torch.full((1, 2, 224, 400), 50.0)      # > d1 = 42 m
    _, vn = m.depth_to_bins(near)
    _, vf = m.depth_to_bins(far)
    assert not vn.any() and not vf.any(), "out-of-range depth must be invalid"
    # min-pool picks the nearest object in each cell
    mixed = torch.full((1, 1, 224, 400), 40.0)
    mixed[..., :112, :200] = 5.0                  # a near patch
    bins_m, _ = m.depth_to_bins(mixed.expand(1, 2, 224, 400).contiguous())
    assert bins_m.min().item() == int((5.0 - m.d0) / m.dd), "min-pool must keep the near depth"
    print(f"ok  depth_to_bins: 10 m -> bin 16, out-of-range masked, min-pool keeps foreground")

    n = sum(p.numel() for p in m.parameters())
    print(f"ok  parameters: {n/1e6:.2f}M  (D={m.D}, C={m.camC}, "
          f"feat {m.fH}x{m.fW}, pooled {m.gx}x{m.gy} @ {m.pool_res} m)")
    print("\nall geometry self-tests passed")


if __name__ == "__main__":
    _self_test()
