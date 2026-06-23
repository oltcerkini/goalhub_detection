"""Unified player + ball + referee detection using YOLO26m.

One model, one inference call per frame. Supports both the COCO-pretrained
yolo26m.pt and the fine-tuned soccana variant (Player, Ball, Referee).

Configurable inference resolution: bump imgsz to 2560 for 4K footage
(default 1280 misses many small players and the ball at distance).
"""

import numpy as np
import cv2
import supervision as sv

# Our internal class IDs (aligns with soccana fine-tuned model)
PLAYER = 0
BALL = 1
REFEREE = 2

COCO_TO_OURS = {0: PLAYER, 32: BALL}  # person→player, sports_ball→ball


class YOLODetector:
    """Wraps YOLO for unified football detection.

    Higher imgsz = better small-object detection (players at distance, ball)
    but slower. For 4K football footage, 2560 is recommended. 1280 is the
    minimum that works for close-up shots.
    """

    # Default per-class confidence thresholds (used when conf=None)
    DEFAULT_CONF = {PLAYER: 0.20, BALL: 0.06, REFEREE: 0.20}

    def __init__(self, model_path=None, conf=0.25, iou=0.5, imgsz=2560,
                 per_class_conf=None):
        """
        Args:
            model_path: Path to .pt file. None = yolo26m.pt (auto-download).
            conf: Base confidence threshold. Per-class thresholds override this.
            iou: NMS IoU threshold.
            imgsz: Inference resolution (longest edge). 2560 recommended for 4K.
            per_class_conf: Dict {class_id: threshold} to override base conf.
                            Default uses PLAYER=0.25, BALL=0.08, REFEREE=0.25.
        """
        from ultralytics import YOLO
        if model_path is None:
            model_path = "yolo26m.pt"
        print(f"Loading YOLO model: {model_path} (imgsz={imgsz})")
        self.yolo = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self._model_path = model_path

        # Detect model type by number of classes
        self._names = self.yolo.names
        nc = len(self._names)
        if nc == 3:
            print(f"  Detected fine-tuned model ({nc} classes: {self._names})")
            self._class_map = {0: PLAYER, 1: BALL, 2: REFEREE}
        else:
            print(f"  Detected base COCO model ({nc} classes)")
            self._class_map = COCO_TO_OURS

        # Per-class confidence overrides
        self._per_class_conf = per_class_conf or dict(self.DEFAULT_CONF)

    # ── Main detection ───────────────────────────────────────────────────────

    def detect(self, image, conf=None):
        """Run detection and return sv.Detections with unified class_ids.

        Returns sv.Detections with our internal class mapping:
            0 = Player, 1 = Ball, 2 = Referee
        Returns None if nothing detected.
        """
        results = self.yolo(
            image, conf=conf or self.conf, iou=self.iou,
            imgsz=min(self.imgsz, max(image.shape[0], image.shape[1])), verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        # Remap to our internal classes
        our_cls = []
        for c in cls_ids:
            our_cls.append(self._class_map.get(c, -1))

        our_cls = np.array(our_cls, dtype=int)

        # Filter out unmapped classes
        valid = our_cls >= 0
        if not valid.any():
            return None

        detections = sv.Detections(
            xyxy=xyxy[valid],
            confidence=confs[valid],
            class_id=our_cls[valid],
        )
        return detections

    def detect_with_per_class_conf(self, image):
        """Run detection with per-class confidence thresholds.

        Runs YOLO at low base conf (0.05), then filters each class by its
        specific threshold. This catches low-confidence balls that would
        be missed with a single high threshold.

        Returns sv.Detections or None.
        """
        results = self.yolo(
            image, conf=0.05, iou=self.iou,
            imgsz=min(self.imgsz, max(image.shape[0], image.shape[1])), verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        our_cls = []
        for c in cls_ids:
            our_cls.append(self._class_map.get(c, -1))
        our_cls = np.array(our_cls, dtype=int)

        # Per-class threshold filter
        valid = np.ones(len(our_cls), dtype=bool)
        for i in range(len(our_cls)):
            if our_cls[i] < 0:
                valid[i] = False
            else:
                min_conf = self._per_class_conf.get(our_cls[i], self.conf)
                if confs[i] < min_conf:
                    valid[i] = False

        if not valid.any():
            return None

        return sv.Detections(
            xyxy=xyxy[valid],
            confidence=confs[valid],
            class_id=our_cls[valid],
        )

    # ── Convenience filters ──────────────────────────────────────────────────

    def get_players(self, detections):
        """Filter to Player class only."""
        if detections is None or len(detections) == 0:
            return None
        mask = detections.class_id == PLAYER
        return detections[mask] if mask.any() else None

    def get_ball(self, detections):
        """Filter to Ball class only."""
        if detections is None or len(detections) == 0:
            return None
        mask = detections.class_id == BALL
        return detections[mask] if mask.any() else None

    def get_referees(self, detections):
        """Filter to Referee class only."""
        if detections is None or len(detections) == 0:
            return None
        mask = detections.class_id == REFEREE
        return detections[mask] if mask.any() else None

    # ── Polygon filtering ────────────────────────────────────────────────────

    def filter_by_polygon(self, detections, polygon, margin_px=0):
        """Keep only detections whose feet are inside (or near) the pitch polygon.

        Uses feet position (bottom-center of bbox) to determine pitch membership.
        margin_px allows players slightly outside the line (sideline tackles, etc).

        Returns filtered sv.Detections.
        """
        if detections is None or len(detections) == 0:
            return None

        feet = np.column_stack([
            (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2,
            detections.xyxy[:, 3],
        ])
        distances = np.array([
            cv2.pointPolygonTest(polygon, (float(c[0]), float(c[1])), True)
            for c in feet
        ])
        inside = distances >= -margin_px
        return detections[inside] if inside.any() else None

    def detect_and_filter(self, image, polygon=None, conf=None):
        """Detect and filter to pitch polygon in one call.

        Uses per-class confidence thresholds by default (lower threshold for
        ball, higher for players/referees). When conf is explicitly provided,
        uses that single threshold for all classes.

        Returns:
            (full_detections, pitch_detections) — both sv.Detections or None.
        """
        dets = self.detect_with_per_class_conf(image) if conf is None else self.detect(image, conf=conf)
        if dets is None:
            return None, None
        if polygon is not None:
            pitch_dets = self.filter_by_polygon(dets, polygon)
        else:
            pitch_dets = dets
        return dets, pitch_dets

    # ── Bbox expansion ─────────────────────────────────────────────────────────

    @staticmethod
    def expand_bboxes(detections, expand_ratio=0.06, img_shape=None):
        """Expand all bboxes outward by a percentage of their size.

        YOLO boxes are often tight on the player, cutting off heads/feet.
        Expanding by 6% ensures the full player is inside the box.

        Args:
            detections: sv.Detections with xyxy.
            expand_ratio: Fraction of box width/height to add on each side.
            img_shape: (h, w) to clamp bboxes to image bounds. Can be None.

        Returns: sv.Detections with expanded xyxy (modified in-place).
        """
        if detections is None or len(detections) == 0:
            return detections
        ws = detections.xyxy[:, 2] - detections.xyxy[:, 0]
        hs = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        dw = ws * expand_ratio
        dh = hs * expand_ratio
        detections.xyxy[:, 0] -= dw  # x1
        detections.xyxy[:, 1] -= dh  # y1
        detections.xyxy[:, 2] += dw  # x2
        detections.xyxy[:, 3] += dh  # y2
        if img_shape is not None:
            h, w = img_shape[:2]
            detections.xyxy[:, 0] = np.clip(detections.xyxy[:, 0], 0, w)
            detections.xyxy[:, 1] = np.clip(detections.xyxy[:, 1], 0, h)
            detections.xyxy[:, 2] = np.clip(detections.xyxy[:, 2], 0, w)
            detections.xyxy[:, 3] = np.clip(detections.xyxy[:, 3], 0, h)
        return detections

    # ── Model info ───────────────────────────────────────────────────────────

    @property
    def class_names(self):
        """Return dict mapping our internal class_id → name."""
        return {PLAYER: "Player", BALL: "Ball", REFEREE: "Referee"}

    @property
    def is_finetuned(self):
        """True if using soccana fine-tuned model (3-class)."""
        return len(self._names) == 3
