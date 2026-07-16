"""Command-line interface for the traffic risk pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from traffic_risk.features.extract import (
    export_features,
    extract_features_from_tracked_frames,
    run_feature_extraction,
)
from traffic_risk.features.visibility import frame_luminance_bgr
from traffic_risk.io.dataset_registry import list_videos
from traffic_risk.io.video_reader import iter_frames
from traffic_risk.perception.detect_yolo import DEFAULT_CLASSES, detect_frames
from traffic_risk.perception.track import TrackedFrame, track_detections
from traffic_risk.perception.types import Detection, DetectionFrame
from traffic_risk.utils.config import PipelineSettings, load_config
from traffic_risk.utils.logging import configure_logging
from traffic_risk.viz.overlays import render_annotated_outputs
from traffic_risk.viz.plots import load_features_table, plot_feature_summaries

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Traffic risk visual feature extraction pipeline"
    )
    parser.add_argument(
        "--config",
        default="./configs/default.yaml",
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG).",
    )

    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list-videos", help="List videos for a dataset")
    list_parser.add_argument("--dataset", required=True, help="Dataset name.")
    list_parser.add_argument(
        "--config-path",
        default="./configs/datasets.yaml",
        help="Path to datasets config.",
    )

    peek_parser = subparsers.add_parser("peek", help="Peek at a video and print frame info")
    peek_parser.add_argument("--video", required=True, help="Path to a video file.")
    peek_parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Max frames to sample for quick checks.",
    )
    peek_parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Sample every N frames.",
    )

    detect_parser = subparsers.add_parser("detect", help="Run YOLOv8 detection on a video")
    detect_parser.add_argument("--video", required=True, help="Path to a video file.")
    detect_parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model weights path or name.",
    )
    detect_parser.add_argument(
        "--out",
        required=True,
        help="Output path for detections (.pkl or .parquet).",
    )
    detect_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    detect_parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    detect_parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Sample every N frames.",
    )
    detect_parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class whitelist.",
    )
    detect_parser.add_argument(
        "--cache-path",
        default=None,
        help="Optional cache path (.pkl or .parquet).",
    )

    track_parser = subparsers.add_parser(
        "track", help="Run lightweight multi-object tracking on cached detections"
    )
    track_parser.add_argument(
        "--detections",
        required=True,
        help="Input detections file (.pkl or .parquet) produced by detect command.",
    )
    track_parser.add_argument("--out", required=True, help="Output tracked frames parquet file.")
    track_parser.add_argument("--iou-threshold", type=float, default=0.3)
    track_parser.add_argument("--max-age", type=int, default=10)

    extract_parser = subparsers.add_parser(
        "extract", help="Extract frame-level traffic features from a video"
    )
    extract_parser.add_argument("--video", required=True, help="Path to a video file.")
    extract_parser.add_argument("--outdir", required=True, help="Output directory for features.")
    extract_parser.add_argument("--sample-every", type=int, default=2)
    extract_parser.add_argument(
        "--format",
        choices=["csv", "parquet", "both"],
        default="both",
        help="Export format.",
    )
    extract_parser.add_argument("--model", default="yolov8n.pt")
    extract_parser.add_argument("--conf", type=float, default=0.25)
    extract_parser.add_argument("--iou", type=float, default=0.45)

    viz_parser = subparsers.add_parser(
        "viz", help="Generate annotated outputs and feature plots"
    )
    viz_parser.add_argument("--video", required=True, help="Path to source video.")
    viz_parser.add_argument("--features", required=True, help="Feature table path (.parquet/.csv).")
    viz_parser.add_argument("--outdir", required=True, help="Directory for visualization outputs.")
    viz_parser.add_argument(
        "--make-video",
        choices=["true", "false"],
        default="true",
        help="Whether to render annotated mp4.",
    )
    viz_parser.add_argument(
        "--tracks",
        default=None,
        help="Tracked detections path (.parquet/.pkl) for drawing boxes/labels.",
    )
    viz_parser.add_argument("--sample-every", type=int, default=1)

    run_parser = subparsers.add_parser(
        "run", help="End-to-end run: detect, track, extract, export, and visualize"
    )
    run_parser.add_argument("--video", required=True, help="Path to source video.")
    run_parser.add_argument("--outdir", required=True, help="Output directory.")
    run_parser.add_argument("--sample-every", type=int, default=2)
    run_parser.add_argument("--conf", type=float, default=0.25)
    run_parser.add_argument("--iou", type=float, default=0.45)
    run_parser.add_argument("--model", default="yolov8n.pt")
    run_parser.add_argument(
        "--format",
        choices=["csv", "parquet", "both"],
        default="both",
        help="Feature export format.",
    )
    run_parser.add_argument("--cache-path", default=None, help="Optional detection cache path.")
    run_parser.add_argument(
        "--annotated-video",
        choices=["true", "false"],
        default="false",
        help="Render annotated video and frames.",
    )
    run_parser.add_argument("--iou-threshold", type=float, default=0.3)
    run_parser.add_argument("--max-age", type=int, default=10)

    return parser


def _serialize_detection_frame(frame: DetectionFrame) -> dict[str, object]:
    return {
        "frame_idx": frame.frame_idx,
        "timestamp": frame.timestamp,
        "detections": [
            {
                "bbox_xyxy": det.bbox_xyxy,
                "confidence": det.confidence,
                "class_id": det.class_id,
                "class_name": det.class_name,
            }
            for det in frame.detections
        ],
    }


def _serialize_tracked_frame(frame: TrackedFrame) -> dict[str, object]:
    return {
        "frame_idx": frame.frame_idx,
        "timestamp": frame.timestamp,
        "detections": [
            {
                "track_id": det.track_id,
                "bbox_xyxy": det.bbox_xyxy,
                "confidence": det.confidence,
                "class_id": det.class_id,
                "class_name": det.class_name,
            }
            for det in frame.detections
        ],
    }


def _load_detection_frames(path: Path) -> list[DetectionFrame]:
    if path.suffix == ".parquet":
        records = pd.read_parquet(path).to_dict("records")
    else:
        records = pd.read_pickle(path)

    frames: list[DetectionFrame] = []
    for row in records:
        detections = [
            Detection(
                bbox_xyxy=tuple(det["bbox_xyxy"]),
                confidence=float(det["confidence"]),
                class_id=int(det["class_id"]),
                class_name=str(det["class_name"]),
            )
            for det in row.get("detections", [])
        ]
        frames.append(
            DetectionFrame(
                frame_idx=int(row["frame_idx"]),
                timestamp=float(row["timestamp"]),
                detections=detections,
            )
        )
    return frames


def _run_end_to_end(args: argparse.Namespace, config: dict[str, object]) -> None:
    video_cfg = config.get("video", {}) if isinstance(config, dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    tracking_cfg = config.get("tracking", {}) if isinstance(config, dict) else {}

    video_path = args.video or video_cfg.get("input_path")
    if not video_path:
        raise ValueError("--video is required")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sample_every = args.sample_every
    conf = args.conf if args.conf is not None else float(model_cfg.get("confidence", 0.25))
    iou = args.iou if args.iou is not None else float(model_cfg.get("iou", 0.45))
    model = args.model or str(model_cfg.get("yolo_weights", "yolov8n.pt"))
    iou_threshold = args.iou_threshold if args.iou_threshold is not None else 0.3
    max_age = args.max_age if args.max_age is not None else int(tracking_cfg.get("max_age", 10))

    LOGGER.info("[run] reading frames: %s", video_path)
    sampled_frames = list(iter_frames(video_path, sample_every=sample_every))
    LOGGER.info("[run] sampled frames: %d", len(sampled_frames))

    visibility = {
        frame_idx: frame_luminance_bgr(frame) for frame_idx, _timestamp, frame in sampled_frames
    }

    class _FrameSource:
        def __init__(self) -> None:
            self.video_path = str(video_path)
            self.sample_every = sample_every
            self.class_whitelist = list(DEFAULT_CLASSES)
            self.cache_path = args.cache_path

        def __iter__(self):
            return iter(sampled_frames)

    LOGGER.info("[run] running detection (model=%s, conf=%.2f, iou=%.2f)", model, conf, iou)
    detection_frames = list(detect_frames(model, _FrameSource(), conf=conf, iou=iou))
    LOGGER.info("[run] detection frames: %d", len(detection_frames))

    detections_out = outdir / "detections.parquet"
    pd.DataFrame([_serialize_detection_frame(frame) for frame in detection_frames]).to_parquet(
        detections_out, index=False
    )
    LOGGER.info("[run] detections saved: %s", detections_out)

    LOGGER.info("[run] tracking (iou_threshold=%.2f, max_age=%d)", iou_threshold, max_age)
    tracked_frames = list(
        track_detections(detection_frames, iou_threshold=iou_threshold, max_age=max_age)
    )
    tracked_out = outdir / "tracked.parquet"
    pd.DataFrame([_serialize_tracked_frame(frame) for frame in tracked_frames]).to_parquet(
        tracked_out, index=False
    )
    LOGGER.info("[run] tracks saved: %s", tracked_out)

    fps = 0.0
    if len(sampled_frames) >= 2:
        dt = sampled_frames[1][1] - sampled_frames[0][1]
        fps = (1.0 / dt) if dt > 0 else 0.0

    LOGGER.info("[run] extracting features")
    features = extract_features_from_tracked_frames(
        tracked_frames,
        visibility,
        video_id=Path(video_path).stem,
        fps=fps,
        sample_every=sample_every,
    )
    exported = export_features(
        features,
        outdir=outdir,
        video_id=Path(video_path).stem,
        file_format=args.format,
    )
    for path in exported:
        LOGGER.info("[run] feature file: %s", path)

    LOGGER.info("[run] generating plots")
    plot_paths = plot_feature_summaries(features, outdir=outdir)
    for path in plot_paths:
        LOGGER.info("[run] plot: %s", path)

    if args.annotated_video.lower() == "true":
        LOGGER.info("[run] rendering annotations")
        outputs = render_annotated_outputs(
            video_path=video_path,
            tracks_path=tracked_out,
            outdir=outdir,
            sample_every=sample_every,
            make_video=True,
        )
        for path in outputs:
            LOGGER.info("[run] annotation output: %s", path)


def main() -> None:
    """Run the (placeholder) pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    settings = PipelineSettings()
    configure_logging(args.log_level or settings.log_level)
    config = load_config(args.config or settings.config_path)

    if args.command == "list-videos":
        videos = list_videos(args.dataset, config_path=args.config_path)
        for video in videos:
            print(video)
        return

    if args.command == "peek":
        for frame_index, timestamp, _frame in iter_frames(
            args.video,
            sample_every=args.sample_every,
            max_frames=args.max_frames,
        ):
            print(f"frame={frame_index} timestamp={timestamp:.3f}s")
        return

    if args.command == "detect":

        class _FrameSource:
            def __init__(self) -> None:
                self.video_path = args.video
                self.sample_every = args.sample_every
                self.class_whitelist = [
                    name.strip() for name in args.classes.split(",") if name.strip()
                ]
                self.cache_path = args.cache_path

            def __iter__(self):
                return iter_frames(self.video_path, sample_every=self.sample_every)

        detections = list(
            detect_frames(
                args.model,
                _FrameSource(),
                conf=args.conf,
                iou=args.iou,
            )
        )
        out_path = Path(args.out)
        rows = [_serialize_detection_frame(frame) for frame in detections]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".parquet":
            pd.DataFrame(rows).to_parquet(out_path, index=False)
        else:
            pd.to_pickle(rows, out_path)
        return

    if args.command == "track":
        detection_path = Path(args.detections)
        output_path = Path(args.out)
        detection_frames = _load_detection_frames(detection_path)
        tracked = list(
            track_detections(
                detection_frames,
                iou_threshold=args.iou_threshold,
                max_age=args.max_age,
            )
        )
        rows = [_serialize_tracked_frame(frame) for frame in tracked]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(output_path, index=False)
        return

    if args.command == "extract":
        table = run_feature_extraction(
            args.video,
            model=args.model,
            conf=args.conf,
            iou=args.iou,
            sample_every=args.sample_every,
        )
        outputs = export_features(
            table,
            outdir=args.outdir,
            video_id=Path(args.video).stem,
            file_format=args.format,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "viz":
        output_dir = Path(args.outdir)
        output_dir.mkdir(parents=True, exist_ok=True)

        features = load_features_table(args.features)
        plot_paths = plot_feature_summaries(features, outdir=output_dir)
        for path in plot_paths:
            print(path)

        make_video = args.make_video.lower() == "true"
        if args.tracks:
            overlay_outputs = render_annotated_outputs(
                video_path=args.video,
                tracks_path=args.tracks,
                outdir=output_dir,
                sample_every=args.sample_every,
                make_video=make_video,
            )
            for path in overlay_outputs:
                print(path)
        elif make_video:
            raise ValueError("--tracks is required when --make-video true to draw boxes and track ids.")
        return

    if args.command == "run":
        _run_end_to_end(args, config)
        return

    print("Loaded config:", config)


if __name__ == "__main__":
    main()
