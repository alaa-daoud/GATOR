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

If Poetry is unavailable, install with pip:

```bash
pip install -r requirements.txt
```

### 3) Run a placeholder pipeline

```bash
python -m traffic_risk --help
```

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
