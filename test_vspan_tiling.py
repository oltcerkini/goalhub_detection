"""Check the vertical-span-aware projection and grid."""
import numpy as np

from spherical_projection import equirect_to_perspective
from spherical_tiling import (build_grid, hemisphere_mask, merge_to_equirect,
                              tile_pixel_to_equirect)


def _coord_image(w, h):
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (np.arange(w)[None, :] * 255 // (w - 1)).astype(np.uint8)  # B = x
    img[..., 1] = (np.arange(h)[:, None] * 255 // (h - 1)).astype(np.uint8)  # G = y
    return img


def main():
    W, H = 800, 400
    img = _coord_image(W, H)

    for vspan in (180.0, 40.0):
        tile = equirect_to_perspective(img, 0, 0, 60, 1920, 1080, 180.0, vspan)
        for (u, v) in [(960, 540), (960, 120), (960, 960), (300, 540), (1600, 540)]:
            X, Y = tile_pixel_to_equirect(u, v, 0, 0, 60, 180, W, H, vspan=vspan)
            eb = int(round(X / (W - 1) * 255))
            eg = int(round(Y / (H - 1) * 255))
            tb, tg = int(tile[v, u, 0]), int(tile[v, u, 1])
            assert abs(tb - eb) <= 6 and abs(tg - eg) <= 6, (vspan, u, v, tb, eb, tg, eg)

    assert len(build_grid(180, 60, vspan=180.0)) == 10
    assert len(build_grid(180, 60, vspan=40.0)) == 5

    # vspan 30: a 36-deg-tall tile runs past the top/bottom of the strip.
    m = hemisphere_mask(0, 0, 60, 180, vspan=30.0)
    assert not m[0, 960] and not m[-1, 960] and m[540, 960]
    assert hemisphere_mask(0, 0, 60, 180, vspan=180.0).all()

    det = {"tile_id": 0, "yaw": 0.0, "pitch": 0.0, "fov": 60.0,
           "bbox": (900.0, 500.0, 1020.0, 580.0), "conf": 0.9, "cls": 0}
    kept, _ = merge_to_equirect([det], 180.0, W, H, vspan=40.0)
    assert len(kept) == 1
    cx, cy = kept[0]["center"]
    assert abs(cx - W / 2) < 2 and abs(cy - H / 2) < 2, (cx, cy)

    print("test_vspan_tiling: ALL PASS")


if __name__ == "__main__":
    main()
