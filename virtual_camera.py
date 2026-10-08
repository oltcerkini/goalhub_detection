"""Phase 2.7 — virtual camera controller (skeleton).

Maps a tracked ball position (equirect pixels) to a smooth virtual-camera
(yaw, pitch) that follows it without snapping. Standalone; no existing module
is touched. Output is a camera pose only — rendering is a later phase.
"""

import numpy as np

# Follow smoothing and limits.
ALPHA = 0.1                 # smoothing factor at the reference frame rate
REF_FPS = 30.0              # alpha is defined for this rate
PITCH_MIN, PITCH_MAX = -60.0, -10.0   # clamp the camera into the pitch band


def equirect_xy_to_spherical(x, y, eq_w, eq_h, span):
    """Equirect pixel -> (yaw_deg, pitch_deg). Inverse of the pixel mapping."""
    yaw = (x / eq_w - 0.5) * span
    pitch = (0.5 - y / eq_h) * 180.0
    return float(yaw), float(pitch)


def compute_camera_pose(ball_xy, prev_yaw, prev_pitch, dt, span,
                        eq_w, eq_h, alpha=ALPHA,
                        pitch_clamp=(PITCH_MIN, PITCH_MAX)):
    """Target the ball's direction and ease toward it.

    ball_xy      : (x, y) in equirect pixels (cropped content region)
    prev_yaw     : previous camera yaw (deg); pass 0.0 on the first call
    prev_pitch   : previous camera pitch (deg); pass a mid value (e.g. -30) first
    dt           : seconds since the previous call
    span         : 180 or 360 (yaw is clamped to -span/2 .. +span/2)
    eq_w, eq_h   : equirect content size in pixels

    Smoothing is frame-rate independent: `alpha` is the ease applied per 1/REF_FPS
    second, rescaled to the actual `dt`. Returns (yaw, pitch) in degrees.
    """
    tgt_yaw, tgt_pitch = equirect_xy_to_spherical(ball_xy[0], ball_xy[1],
                                                  eq_w, eq_h, span)
    a = 1.0 - (1.0 - alpha) ** (dt * REF_FPS)
    yaw = prev_yaw + a * (tgt_yaw - prev_yaw)
    pitch = prev_pitch + a * (tgt_pitch - prev_pitch)
    half = span / 2.0
    yaw = float(np.clip(yaw, -half, half))
    pitch = float(np.clip(pitch, pitch_clamp[0], pitch_clamp[1]))
    return yaw, pitch
