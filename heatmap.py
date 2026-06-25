"""Player position heatmap — density overlay on a pitch view.

Generates a 2D histogram of player positions and overlays it on
a pitch-shaped canvas using a warm colormap.
"""

import numpy as np
import cv2


class HeatmapGenerator:
    """Generates position density heatmaps from player detections."""

    def __init__(self, width=800, height=520, sigma=25, alpha=0.55):
        self.width = width          # output canvas width
        self.height = height        # output canvas height
        self.sigma = sigma          # Gaussian blur radius
        self.alpha = alpha          # overlay opacity

    def generate(self, player_detections, pitch_polygon=None):
        """Generate a heatmap image from player foot positions.

        Args:
            player_detections: list of dicts with 'bbox' [x1,y1,x2,y2]
            pitch_polygon: optional 4-point list for pitch outline overlay

        Returns:
            heatmap_img (H x W x 3 BGR numpy array)
        """
        # Extract foot positions (bottom-centre of bbox)
        feet = []
        for d in player_detections:
            bbox = d["bbox"]
            fx = (bbox[0] + bbox[2]) / 2.0
            fy = bbox[3]  # bottom = feet
            feet.append((fx, fy))

        if not feet:
            return self._blank_canvas(pitch_polygon)

        pts = np.array(feet)

        # Determine bounds from data (or pitch polygon)
        if pitch_polygon is not None:
            poly = np.array(pitch_polygon, dtype=np.float32)
            min_x, min_y = poly.min(axis=0)
            max_x, max_y = poly.max(axis=0)
        else:
            min_x, min_y = pts.min(axis=0)
            max_x, max_y = pts.max(axis=0)

        # Margin
        mx = max((max_x - min_x) * 0.05, 10)
        my = max((max_y - min_y) * 0.05, 10)
        min_x -= mx
        min_y -= my
        max_x += mx
        max_y += my

        # Scale to canvas
        scale_x = self.width / (max_x - min_x) if max_x > min_x else 1
        scale_y = self.height / (max_y - min_y) if max_y > min_y else 1

        canvas = np.zeros((self.height, self.width), dtype=np.float32)

        for fx, fy in feet:
            cx = int((fx - min_x) * scale_x)
            cy = int((fy - min_y) * scale_y)
            if 0 <= cx < self.width and 0 <= cy < self.height:
                canvas[cy, cx] += 1

        # Gaussian blur to create smooth density
        if self.sigma > 0:
            canvas = cv2.GaussianBlur(canvas, (0, 0), self.sigma)

        # Normalize to 0-1
        max_val = canvas.max()
        if max_val > 0:
            canvas /= max_val

        # Apply colormap (JET-like: blue → cyan → green → yellow → red)
        heatmap_coloured = cv2.applyColorMap(
            (canvas * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Create base pitch background
        background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        background[:] = (10, 40, 10)  # dark green

        # Blend
        overlay = cv2.addWeighted(background, 0.3, heatmap_coloured, 0.7, 0)

        # Draw pitch outline if provided
        if pitch_polygon is not None:
            poly_canvas = np.array([
                [(int((p[0] - min_x) * scale_x), int((p[1] - min_y) * scale_y))
                 for p in pitch_polygon]
            ], dtype=np.int32)
            cv2.polylines(overlay, [poly_canvas], True, (200, 200, 200), 1, cv2.LINE_AA)

        return overlay

    def _blank_canvas(self, pitch_polygon=None):
        """Return a blank pitch canvas when there's no data."""
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        bg[:] = (10, 40, 10)
        if pitch_polygon is not None:
            return bg
        return bg
