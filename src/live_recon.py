# -*- coding: utf-8 -*-
"""Live 3D reconstruction panel: accumulate a point cloud while driving and render
it from a chase camera that follows the car.

Same idea as results/RECON_spin_*.mp4, but built frame by frame instead of offline,
and viewed from behind the car rather than orbited.

Depth can come from either the BEV model's own depth head (RGB alone) or a depth
camera; live_demo picks with --recon_source.  RGB-only works -- the model already
predicts a distribution over 80 depth bins per pixel, which is what places
features in the BEV, and live_demo's forward pass returns it for free -- but the
measured A/B (results/RECON_ABLATION_*) says a depth camera is clearly better
here: 86k points versus the full 260k cap on the same route.

The gap is COVERAGE, not accuracy.  Where the model is confident its depth is
already finer than this renderer's 0.35 m voxel (median 0.19 m at conf > 0.3 on
test_town01), but the confidence and sky filters throw away ~28% of pixels, and
they throw away exactly the complex structures -- foliage, thin poles -- that
make a reconstruction look like a reconstruction.  So: BEV and hazard warning
need no depth sensor, reconstruction visibly benefits from one.

The renderer is deliberately plain numpy + cv2: project the cloud into a virtual
pinhole, sort back-to-front, scatter into an image.  A real point-cloud engine
would look better but would not survive inside a 100 ms frame budget.
"""
import math

import cv2
import numpy as np

from depth_to_pointcloud import R_CAM_OPT


def backproject(depth_m, rgb, K, T_ego_cam, dmin, dmax, stride=1):
    """Depth image -> ego-frame xyz + colour.

    `stride` subsamples pixels: a full 800x448 frame is 358k points per camera and
    the cloud only needs enough to read as structure, not every pixel.
    """
    H, W = depth_m.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    us = us.ravel(); vs = vs.ravel()
    d = depth_m[vs, us]
    keep = (d >= dmin) & (d <= dmax)
    if not keep.any():
        return np.zeros((0, 3)), np.zeros((0, 3))
    us, vs, d = us[keep], vs[keep], d[keep]
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    optical = np.stack([us * d, vs * d, d], axis=1) @ Kinv.T
    R = np.asarray(T_ego_cam)[:3, :3] @ R_CAM_OPT
    t = np.asarray(T_ego_cam)[:3, 3]
    ego = optical @ R.T + t
    # Keep BGR.  accumulate_pointcloud.py flips to RGB because it writes PLY files;
    # this one paints straight into a cv2 image, so flipping here swaps red and blue
    # in the panel.
    col = rgb[vs, us].astype(np.float32)
    return ego, col


def world_from_ego(pts, ex, ey, yaw_deg):
    a = math.radians(yaw_deg); c, s = math.cos(a), math.sin(a)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.stack([ex + c * x - s * y, ey + s * x + c * y, z], axis=1)


class LiveRecon(object):
    """Growing world-frame point cloud with a chase-camera view."""

    def __init__(self, voxel=0.35, max_points=260_000, size=520,
                 back=16.0, up=10.0, pitch_deg=25.0, fov_deg=70.0,
                 near_m=18.0, mid_m=35.0, gain=1.25, max_radius=6):
        self.voxel = voxel
        self.max_points = max_points
        self.size = size
        self.back, self.up = back, up
        self.pitch = math.radians(pitch_deg)
        self.f = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        self.near_m, self.mid_m = near_m, mid_m     # kept for callers
        # Cap the splat, because the ideal size grows without bound as points
        # approach the virtual camera and blocks that big smear over the lane
        # markings they are supposed to show. 3 was enough for an RGB-only cloud
        # -- the confidence filter thins the near ground anyway -- but a depth
        # camera fills it, and then the 0.35 m voxel spacing projects to ~20 px
        # right under the car and a 3 px splat reads as a checkerboard.
        self.max_radius = max_radius
        self.gain = gain                            # CARLA street colour is dim
        self.xyz = np.zeros((0, 3), np.float32)
        self.rgb = np.zeros((0, 3), np.float32)
        self._keys = None

    def reset(self):
        """Drop the cloud.  Needed when the ego teleports: the accumulated points
        stay where they were mapped, so after an unstick jump the panel shows an
        empty patch next to a cloud hundreds of metres away."""
        self.xyz = np.zeros((0, 3), np.float32)
        self.rgb = np.zeros((0, 3), np.float32)

    def add(self, ego_pts, colours, ex, ey, yaw_deg):
        if len(ego_pts) == 0:
            return
        # Height sanity: the ego frame puts the road near z=0, so anything far below
        # it or absurdly high is a depth error, and those are the points that spray
        # into the view as noise.
        ok = (ego_pts[:, 2] > -2.5) & (ego_pts[:, 2] < 14.0)
        if not ok.any():
            return
        ego_pts, colours = ego_pts[ok], colours[ok]
        w = world_from_ego(ego_pts, ex, ey, yaw_deg).astype(np.float32)
        self.xyz = np.concatenate([self.xyz, w])
        self.rgb = np.concatenate([self.rgb, colours.astype(np.float32)])
        # Deduping is an np.unique over the whole cloud -- at 260k points that is the
        # single most expensive thing in the frame. Only do it once the surplus is
        # worth the cost, rather than on every add.
        if len(self.xyz) > self.max_points * 1.25:
            self._dedupe()

    def _voxel_keys(self):
        """One int64 per point identifying its voxel.

        Packing the three axes into a single integer rather than calling
        np.unique(..., axis=0) on an Nx3 array: axis=0 has to build a structured
        view and compare rows, which is roughly an order of magnitude slower, and
        this runs on a live frame.  Offsetting by the per-axis minimum keeps the
        cloud's own extent in the exponent instead of absolute world coordinates,
        so 21 bits per axis covers a ~700 km drive.
        """
        q = np.floor(self.xyz / self.voxel).astype(np.int64)
        q -= q.min(axis=0)
        if q.max() >= (1 << 21):        # never happens in practice; stay correct if it does
            return None
        return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]

    def _dedupe(self):
        """One point per voxel, newest wins, then cap the total at max_points.

        np.unique on the voxel keys is the whole trick -- it both removes the
        duplicates that pile up when the car re-sees a wall and caps the growth,
        so the cloud stays flat in memory instead of growing with drive time.
        """
        key = self._voxel_keys()
        if key is None:
            keys = np.floor(self.xyz / self.voxel).astype(np.int64)
            _, idx = np.unique(keys[::-1], axis=0, return_index=True)
        else:
            # last occurrence wins: reverse, unique keeps the first it meets, reverse back
            _, idx = np.unique(key[::-1], return_index=True)
        idx = len(self.xyz) - 1 - idx
        # np.unique hands back indices ordered by KEY, so idx as it stands is the
        # cloud sorted by world x-then-y-then-z.  Sorting it puts the points back in
        # the order they arrived, which is what the truncation below assumes.
        # Without this, "keep the last max_points" keeps the points with the largest
        # world x -- a slab of the map -- and the panel blanks out every time the cap
        # is hit.  That was the flicker: 316k pts -> 260k pts and an empty panel.
        idx.sort()
        self.xyz, self.rgb = self.xyz[idx], self.rgb[idx]
        if len(self.xyz) > self.max_points:
            self.xyz = self.xyz[-self.max_points:]
            self.rgb = self.rgb[-self.max_points:]

    def render(self, ex, ey, yaw_deg, ez=0.0):
        img = np.full((self.size, self.size, 3), 16, np.uint8)
        if len(self.xyz) < 10:
            return img

        a = math.radians(yaw_deg)
        c, s = math.cos(a), math.sin(a)
        # chase camera: behind the car along its heading, raised
        cx_w = ex - self.back * c
        cy_w = ey - self.back * s
        cz_w = ez + self.up

        d = self.xyz - np.array([cx_w, cy_w, cz_w], np.float32)
        # into the car's heading frame: x forward, y left, z up
        xf = d[:, 0] * c + d[:, 1] * s
        yf = -d[:, 0] * s + d[:, 1] * c
        zf = d[:, 2]
        # Tilt the camera down by self.pitch.  Optical axis becomes (cos, 0, -sin)
        # and camera-up becomes (sin, 0, cos), so the projection is a dot product
        # with each.  Getting these two signs the wrong way round puts the whole
        # scene below the frame -- which is exactly what the first version did.
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        xc = xf * cp - zf * sp
        zc = xf * sp + zf * cp

        front = xc > 0.5
        if not front.any():
            return img
        xc, yf, zc = xc[front], yf[front], zc[front]
        col = self.rgb[front]

        # CARLA is x-forward / y-RIGHT (真車實作指南 §5.1), so a point with yf > 0 is
        # to the right and belongs at u > centre.  Subtracting here mirrors the whole
        # scene left-to-right, which is subtle enough to survive a casual look.
        u = (self.size / 2.0 + self.f * yf / xc).astype(np.int32)
        v = (self.size / 2.0 - self.f * zc / xc).astype(np.int32)
        ok = (u >= 0) & (u < self.size) & (v >= 0) & (v < self.size)
        u, v, xc, col = u[ok], v[ok], xc[ok], col[ok]
        if len(u) == 0:
            return img

        # Painter's algorithm, and splat each point as a small square rather than a
        # single pixel.  One pixel per point reads as noise: the reference offline
        # render (results/RECON_spin_crisp.mp4) uses sized scatter markers, and that
        # is most of why it looks like structure rather than speckle.  Size falls off
        # with distance so near geometry stays solid while far geometry stays fine.
        # Sort on a uint16 depth key, not the float: numpy's stable sort on a
        # 16-bit integer is a radix sort, 4.3 ms for 250k points against 21 ms
        # for the float argsort (measured 2026-10-02).  2 mm steps reach 131 m,
        # beyond anything the cloud holds, and ties inside one step are between
        # points no painter's order could tell apart anyway.
        key = (65535 - np.minimum(xc * 500.0, 65535.0)).astype(np.uint16)
        order = np.argsort(key, kind="stable")
        u, v, xc = u[order], v[order], xc[order]
        cc = np.clip(col[order] * 255.0 * self.gain, 0, 255).astype(np.uint8)

        # Splat size has to track the projected voxel spacing, not sit at a constant.
        # The cloud carries one point per `voxel` metres, so at distance d neighbours
        # land f*voxel/d pixels apart: 3 px at 40 m but 13 px at 10 m.  A fixed 3x3
        # therefore looks continuous far away and disintegrates into separated dots
        # up close -- and a fixed 5x5 just turns those dots into a checkerboard,
        # because the gap is what needs closing, not the dot.  Sizing each splat as
        # half the local spacing makes neighbours meet and the ground read as a
        # surface at every distance.
        # Round UP.  Truncating leaves every splat a fraction smaller than the
        # spacing it has to cover, and the shortfall shows as horizontal seams where
        # the integer radius steps. Erring large just overlaps neighbours slightly.
        r = np.ceil(self.f * self.voxel / (2.0 * xc)).astype(np.int32)
        r = np.clip(r, 0, self.max_radius)
        # Splat by dilating an INDEX image rather than writing each square pixel by
        # pixel.  The obvious loop costs (2r+1)^2 scatter passes per radius, so raising
        # the cap from 3 to 6 nearly quadrupled the render and took the demo from 3.4
        # to 1.9 FPS.  Instead: scatter one pixel per point, carrying the point's
        # position in the depth-sorted array, then let cv2.dilate spread it -- dilate
        # is a separable max filter, so a 13x13 kernel costs about what 3x3 does, and
        # taking the MAXIMUM index means the nearest point wins every overlap for free.
        #
        # Radius rises as distance falls, so ascending radius is far-to-near, which is
        # the order the painter's algorithm needs; bands composite in that order and
        # within a band plain fancy-index assignment keeps the last (nearest) write.
        # float32 holds these indices exactly (well under 2^24 points).
        idx = np.full((self.size, self.size), -1.0, np.float32)
        ids = np.arange(len(u), dtype=np.float32)
        for radius in range(0, int(r.max()) + 1):
            sel = r == radius
            if not sel.any():
                continue
            band = np.full((self.size, self.size), -1.0, np.float32)
            band[v[sel], u[sel]] = ids[sel]
            if radius > 0:
                k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
                band = cv2.dilate(band, k)
            hit = band >= 0
            idx[hit] = band[hit]

        hit = idx >= 0
        if hit.any():
            img[hit] = cc[idx[hit].astype(np.int32)]

        # a dot where the car is, so the viewer can anchor the moving cloud
        eu = int(self.size / 2.0)
        e_xc = self.back * cp + self.up * sp
        e_zc = self.back * sp - self.up * cp
        if e_xc > 0.5:
            ev = int(self.size / 2.0 - self.f * e_zc / e_xc)
            if 0 <= ev < self.size:
                cv2.circle(img, (eu, ev), 5, (60, 200, 255), -1, cv2.LINE_AA)

        cv2.putText(img, f"{len(self.xyz)//1000}k pts", (10, self.size - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
        return img
