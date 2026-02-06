# Traffic Risk – Sub-project 1

Initial scaffold for **visual feature extraction + vehicle tracking from traffic videos**.
This sub-project focuses on ingesting video, running YOLOv8 detection, tracking, feature
extraction, visualization, and exporting artifacts for downstream analysis.

## Quickstart

### 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies (Poetry)

```bash
poetry install --with dev
```

Optional visualization extras:

```bash
poetry install --with dev --extras "viz"
```

If Poetry is unavailable, install with pip:

```bash
pip install -r requirements.txt
```

### 3) Run a placeholder pipeline

```bash
python -m traffic_risk --help
```


### 4) Run end-to-end pipeline

```bash
python -m traffic_risk run --video ./data/raw/sample.mp4 --outdir ./data/processed/run1 --sample-every 2 --conf 0.25
```

This command runs detection, tracking, feature extraction, exports CSV/Parquet, and generates summary plots.
Use `--annotated-video true` to also render an annotated video/frames.

## Project structure

```
configs/                  # Configuration files (YAML/JSON) live here
src/traffic_risk/
  cli.py                  # CLI entrypoint
  io/                     # Video ingestion and dataset registry
  perception/             # YOLOv8 detection + tracking
  features/               # Feature extraction
  viz/                    # Visualizations
  utils/                  # Config, logging, paths
```

## Reproducibility notes

- Use fixed seed values (see TODOs in `utils/config.py`).
- Pin dependencies via Poetry lockfile (`poetry.lock`).
- Capture run configuration and software versions in exports.

## System dependencies

- OpenCV wheels include most codecs, but for full video IO support install FFmpeg:
  - macOS (Homebrew): `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install -y ffmpeg`

## Constraints

- GPU is optional. The default pipeline should run on CPU, with GPU acceleration used
  only when available and configured.


## Tracking limitations (current lightweight baseline)

- Uses simple IoU-based greedy association frame-to-frame (no appearance model).
- Handles short occlusions by keeping unmatched tracks alive for `max_age` frames.
- No motion model/Kalman filtering yet, so identity switches may occur in dense scenes.


## Feature extraction

Extract per-frame features (vehicle count, speed, spacing, luminance):

```bash
python -m traffic_risk extract --video ./data/raw/sample.mp4 --outdir ./data/processed --sample-every 2 --format both
```

Outputs are written with stable schema/order to CSV and/or Parquet:
`video_id, frame_idx, timestamp_s, vehicle_count, mean_speed_px_s, min_distance_px, visibility_luma, fps, sample_every`.


## Visualization

Generate plots from extracted features and optionally render annotated frames/video
(using tracked detections):

```bash
python -m traffic_risk viz \
  --video ./data/raw/sample.mp4 \
  --features ./data/processed/sample_features.parquet \
  --tracks ./data/processed/sample_tracked.parquet \
  --outdir ./data/processed/viz \
  --make-video true
```

Expected outputs:
- `speed_over_time.png`
- `visibility_histogram.png`
- `vehicle_count_over_time.png`
- `annotated_frames/frame_*.jpg`
- `annotated.mp4` (when `--make-video true`)

## Development workflow

Common commands:

```bash
make install
make lint
make format
make test
make run
```

## TODOs

- Implement video ingestion pipeline.
- Integrate YOLOv8 model loading and inference.
- Add tracker integration (e.g., ByteTrack/DeepSORT).
- Design feature extraction schemas.
- Export parquet/csv with metadata.
- Add evaluation metrics and visualization overlays.
