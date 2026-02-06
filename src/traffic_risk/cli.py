"""Command-line interface for the traffic risk pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from traffic_risk.io.dataset_registry import list_videos
from traffic_risk.io.video_reader import iter_frames
from traffic_risk.perception.detect_yolo import DEFAULT_CLASSES, detect_frames
from traffic_risk.utils.config import PipelineSettings, load_config
from traffic_risk.utils.logging import configure_logging


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

    list_parser = subparsers.add_parser(
        "list-videos", help="List videos for a dataset"
    )
    list_parser.add_argument("--dataset", required=True, help="Dataset name.")
    list_parser.add_argument(
        "--config-path",
        default="./configs/datasets.yaml",
        help="Path to datasets config.",
    )

    peek_parser = subparsers.add_parser(
        "peek", help="Peek at a video and print frame info"
    )
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

    detect_parser = subparsers.add_parser(
        "detect", help="Run YOLOv8 detection on a video"
    )
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
    detect_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    detect_parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold.",
    )
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
    return parser


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
        frames_iter = iter_frames(
            args.video,
            sample_every=args.sample_every,
        )
        setattr(frames_iter, "video_path", args.video)
        setattr(frames_iter, "sample_every", args.sample_every)
        setattr(
            frames_iter,
            "class_whitelist",
            [name.strip() for name in args.classes.split(",") if name.strip()],
        )
        if args.cache_path:
            setattr(frames_iter, "cache_path", args.cache_path)

        detections = list(
            detect_frames(
                args.model,
                frames_iter,
                conf=args.conf,
                iou=args.iou,
            )
        )
        if args.out:
            out_path = Path(args.out)
            rows = [
                {
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
                for frame in detections
            ]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.suffix == ".parquet":
                import pandas as pd

                pd.DataFrame(rows).to_parquet(out_path, index=False)
            else:
                import pandas as pd

                pd.to_pickle(rows, out_path)
        return

    # TODO: wire up video ingestion, detection, tracking, and feature extraction.
    print("Loaded config:", config)


if __name__ == "__main__":
    main()
