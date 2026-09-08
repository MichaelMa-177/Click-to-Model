"""Interactive first-frame segmentation with SAM2."""

from __future__ import annotations

import gc
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from click_to_model.config import RuntimePaths


@dataclass(frozen=True)
class MaskPrediction:
    """Accepted SAM2 mask and its point prompts."""

    mask: np.ndarray
    score: float
    points: np.ndarray
    labels: np.ndarray
    valid_depth_fraction: float | None


class Sam2Session:
    """Own a preloaded SAM2 model and expose the interactive prompt UI."""

    def __init__(self, model: Any, predictor: Any, device: str) -> None:
        self.model = model
        self.predictor = predictor
        self.device = device

    @classmethod
    def load(cls, paths: RuntimePaths, device: str) -> Sam2Session:
        if not paths.sam2_checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {paths.sam2_checkpoint}"
            )
        if str(paths.sam2_root) not in sys.path:
            sys.path.insert(0, str(paths.sam2_root))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        started = time.perf_counter()
        print(f"[PRELOAD] Loading SAM2 from {paths.sam2_checkpoint}", flush=True)
        model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            str(paths.sam2_checkpoint),
            device=device,
        )
        session = cls(model, SAM2ImagePredictor(model), device)
        print(
            f"[PRELOAD] SAM2 ready in {time.perf_counter() - started:.2f}s",
            flush=True,
        )
        return session

    def close(self) -> None:
        """Release prompt-model memory before SAM3D inference begins."""
        import torch

        if self.predictor is None:
            return
        reset = getattr(self.predictor, "reset_predictor", None)
        if callable(reset):
            reset()
        self.predictor = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def interactive_mask(
        self,
        image_bgr: np.ndarray,
        depth_raw: np.ndarray | None = None,
        min_depth_coverage: float = 0.25,
    ) -> MaskPrediction:
        """Update the predicted mask after each click until it is accepted."""
        import torch

        points: list[list[int]] = []
        labels: list[int] = []
        current_mask = None
        current_score = None
        current_depth_coverage = None
        dirty = False
        window = "Click-to-Model | left:+ right:- u:undo c:clear enter:accept"

        def mouse(event, x, y, _flags, _param):
            nonlocal dirty
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                points.append([x, y])
                labels.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
                dirty = True

        def redraw():
            canvas = image_bgr.copy()
            if current_mask is not None:
                overlay = np.zeros_like(canvas)
                overlay[current_mask] = (50, 210, 50)
                canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.45, 0.0)
                contours, _ = cv2.findContours(
                    current_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)
            for (x, y), label in zip(points, labels, strict=True):
                cv2.drawMarker(
                    canvas,
                    (x, y),
                    (0, 255, 0) if label else (0, 0, 255),
                    cv2.MARKER_STAR,
                    18,
                    2,
                )

            if current_score is None:
                status = "click object"
                status_color = (255, 255, 255)
            elif current_depth_coverage is None:
                status = f"SAM2 score {current_score:.3f}"
                status_color = (255, 255, 255)
            else:
                status = (
                    f"SAM2 {current_score:.3f} | valid depth "
                    f"{current_depth_coverage:.0%}"
                )
                status_color = (
                    (255, 255, 255)
                    if current_depth_coverage >= min_depth_coverage
                    else (0, 0, 255)
                )
            cv2.putText(
                canvas,
                status,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                status_color,
                2,
            )
            return canvas

        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse)
        try:
            with torch.inference_mode(), amp:
                self.predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                while True:
                    if dirty:
                        if points:
                            masks, scores, _ = self.predictor.predict(
                                point_coords=np.asarray(points, np.float32),
                                point_labels=np.asarray(labels, np.int32),
                                multimask_output=True,
                            )
                            best = int(np.argmax(scores))
                            current_mask = masks[best].astype(bool)
                            current_score = float(scores[best])
                            if depth_raw is not None:
                                current_depth_coverage = float(
                                    np.count_nonzero(depth_raw[current_mask] > 0)
                                    / max(np.count_nonzero(current_mask), 1)
                                )
                        else:
                            current_mask = None
                            current_score = None
                            current_depth_coverage = None
                        dirty = False
                    cv2.imshow(window, redraw())
                    key = cv2.waitKey(20) & 0xFF
                    if key in (13, 32) and current_mask is not None:
                        break
                    if key == ord("u") and points:
                        points.pop()
                        labels.pop()
                        dirty = True
                    elif key == ord("c"):
                        points.clear()
                        labels.clear()
                        dirty = True
                    elif key in (ord("q"), 27):
                        raise RuntimeError("Annotation cancelled")
        finally:
            cv2.destroyWindow(window)

        return MaskPrediction(
            mask=current_mask,
            score=float(current_score),
            points=np.asarray(points, np.float32),
            labels=np.asarray(labels, np.int32),
            valid_depth_fraction=current_depth_coverage,
        )
