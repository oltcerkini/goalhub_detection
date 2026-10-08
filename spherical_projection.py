"""Equirectangular <-> perspective projection for the 180/360 spherical pipeline.

This module is intentionally standalone: it does not import or modify any
existing pipeline module. It is the only piece of the spherical phase.
"""

import numpy as np
import cv2


def equirect_to_perspective(equirect_img, yaw_deg, pitch_deg, fov_deg, out_w, out_h,
                            h_span_deg=360.0):
    """Reproject an equirectangular frame to a flat perspective (pinhole) view.

    CONVENTION
    ----------
    Equirect image centre (W/2, H/2) = yaw 0, pitch 0 (horizon, centre of sphere).

    Yaw increases to the right (positive x in equirect).

    Pitch increases upward (negative y in equirect).

    Horizontal span: the full image width covers ``h_span_deg`` of yaw.
        360.0 = standard full-sphere equirectangular (default).
        180.0 = VR180-style hemisphere equirect (the spec's "180 deg video").
    The vertical span is always 180 deg (equirectangular definition).

    Output image: principal point at centre, focal length
        fx = fy = (out_w / 2) / tan(fov_deg / 2)
    i.e. horizontal FOV = fov_deg with square pixels. Positive output y is down.

    Parameters
    ----------
    equirect_img : H x W x 3 BGR image, the full equirect frame
    yaw_deg      : 0 = looking at centre of frame, +right, -left
    pitch_deg    : 0 = horizon, +up, -down
    fov_deg      : horizontal field of view of the output
    out_w, out_h : output resolution
    h_span_deg   : yaw scope of the source frame (360.0 full sphere / 180.0 VR180)

    Returns
    -------
    out_img : out_h x out_w x 3 BGR perspective projection
    """
    H, W = equirect_img.shape[:2]

    # --- Output (pinhole) intrinsics: principal point at centre, square pixels ---
    fx = (out_w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    fy = fx
    cx = out_w / 2.0
    cy = out_h / 2.0

    # --- Camera basis for the requested view ---
    # World frame: +Y up, +Z forward at yaw=0/pitch=0, +X right.
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    fwd = np.array([np.sin(yaw) * np.cos(pitch),
                    np.sin(pitch),
                    np.cos(yaw) * np.cos(pitch)], dtype=np.float64)
    right = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)

    # --- Iterate over output pixels -> ray direction in world space ---
    us = np.arange(out_w, dtype=np.float64)
    vs = np.arange(out_h, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)                 # (out_h, out_w)
    x = (uu - cx) / fx                           # +x = right
    y = (vv - cy) / fy                           # +y = down in image space
    # ray = fwd + x*right + (-y)*up   (image y grows downward => -up)
    dx = fwd[0] + x * right[0] - y * up[0]
    dy = fwd[1] + x * right[1] - y * up[1]
    dz = fwd[2] + x * right[2] - y * up[2]

    # --- World ray -> spherical (yaw = azimuth, pitch = elevation) ---
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx /= norm
    dy /= norm
    dz /= norm
    ray_pitch = np.arcsin(np.clip(dy, -1.0, 1.0))
    ray_yaw = np.arctan2(dx, dz)

    # --- Spherical -> equirect pixel coordinates ---
    # yaw 0 -> W/2 ; +yaw -> +x.  pitch +up -> smaller y (toward top).
    map_x = ((np.degrees(ray_yaw) / h_span_deg) + 0.5) * W
    map_y = (0.5 - np.degrees(ray_pitch) / 180.0) * H
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Full-sphere frames wrap seamlessly at the +/-180 deg seam; a hemisphere
    # frame does not, so replicate at its edges instead.
    border = cv2.BORDER_WRAP if abs(h_span_deg - 360.0) < 1e-6 else cv2.BORDER_REPLICATE

    return cv2.remap(equirect_img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR, borderMode=border)
