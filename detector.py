"""Player detection using rf-detr (COCO-pretrained, filters for 'person' class)."""

import sys
import numpy as np
import cv2


class PlayerDetector:
    """Wraps rf-detr for person detection with pitch-based filtering."""

    def __init__(self, model_size="medium"):
        self.model_size = model_size
        self._model = None
        self._person_class_id = 1  # rf-detr COCO dict: person = 1
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return
        try:
            if self.model_size == "nano":
                from rfdetr import RFDETRNano as Model
            elif self.model_size == "small":
                from rfdetr import RFDETRSmall as Model
            elif self.model_size == "large":
                from rfdetr import RFDETRLarge as Model
            elif self.model_size == "xlarge":
                from rfdetr import RFDETRXLarge as Model
            else:
                from rfdetr import RFDETRMedium as Model

            print(f"Loading RF-DETR-{self.model_size.capitalize()}...")
            self._model = Model()
            self._model.optimize_for_inference()
            self._model_loaded = True
            print("Model loaded.")
        except ImportError:
            print(
                "ERROR: rfdetr package not found. Install with: pip install rfdetr",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as e:
            print(f"ERROR loading rf-detr model: {e}", file=sys.stderr)
            sys.exit(1)

    def detect(self, image, threshold=0.5, resize_long_side=None):
        """Run detection and return sv.Detections filtered for person class.

        For high-resolution images (e.g. 4K), pass resize_long_side to
        downscale before detection — this drastically improves small-object
        recall since the model was trained at ~640px scale.

        Args:
            image: BGR numpy array
            threshold: confidence threshold
            resize_long_side: if set, downscale so longest edge = this (px)

        Returns:
            sv.Detections containing only 'person' class detections,
            or None on failure.
        """
        self._load_model()

        h, w = image.shape[:2]

        if resize_long_side is not None and max(h, w) > resize_long_side:
            scale = resize_long_side / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            detect_img = cv2.resize(image, (new_w, new_h))
            did_resize = True
        else:
            detect_img = image
            scale = 1.0
            did_resize = False

        # rf-detr expects RGB; convert from BGR
        rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)

        try:
            detections = self._model.predict(rgb, threshold=threshold)
        except Exception as e:
            print(f"Detection error: {e}", file=sys.stderr)
            return None

        if detections is None or len(detections) == 0:
            print("No detections found.")
            return None

        # Filter to person class only
        person_mask = detections.class_id == self._person_class_id
        people = detections[person_mask]

        if len(people) == 0:
            print("No people detected.")
            return None

        # Scale boxes back to original image coordinates
        if did_resize:
            inv_scale = 1.0 / scale
            people.xyxy[:, [0, 2]] *= inv_scale
            people.xyxy[:, [1, 3]] *= inv_scale

        print(f"Detected {len(people)} people.")

        if resize_long_side:
            # Show per-player pixel height for debugging
            heights = people.xyxy[:, 3] - people.xyxy[:, 1]
            print(f"  Player height range: {heights.min():.0f}-{heights.max():.0f}px "
                  f"in original ({w}x{h})")

        return people

    def detect_and_filter_by_polygon(self, image, polygon, threshold=0.5,
                                     resize_long_side=None):
        """Detect people and filter those inside a pitch polygon.

        Args:
            image: BGR numpy array
            polygon: (N, 2) array of polygon vertices (pitch boundary)
            threshold: confidence threshold
            resize_long_side: downscale to this size before detection

        Returns:
            tuple (all_people, inside_people) where both are sv.Detections
        """
        people = self.detect(image, threshold, resize_long_side=resize_long_side)
        if people is None or len(people) == 0:
            return None, None

        # Compute bottom-center of each detection (FEET position — feet are on the pitch)
        # Using feet position instead of bbox center eliminates flickering
        # when the player's torso center drifts near the polygon boundary.
        feet_positions = np.column_stack([
            (people.xyxy[:, 0] + people.xyxy[:, 2]) / 2,  # center x
            people.xyxy[:, 3],  # bottom y (feet)
        ])

        # Margin: allow players whose feet are up to 30px outside the pitch line
        # (sideline players, sliding tackles, etc.)
        margin_px = 30
        distances = np.array([
            cv2.pointPolygonTest(polygon, (float(c[0]), float(c[1])), True)
            for c in feet_positions
        ])
        inside = distances >= -margin_px

        inside_people = people[inside] if inside.any() else None
        outside_count = (~inside).sum()
        margin_count = ((distances < 0) & (distances >= -margin_px)).sum()
        print(f"  {inside.sum()} inside pitch (+ {margin_count} near line), "
              f"{outside_count} far outside")

        return people, inside_people
