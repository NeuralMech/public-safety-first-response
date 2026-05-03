# Public Safety First Response

This repository is structured around a **public-safety first-response system**, not a single drone-only project.

Top-level system flow:

1. Smart CCTV or an external report recognizes an early risk signal
2. A human-in-the-loop operator reviews the event
3. The drone system performs first-response support
4. Internal drone safety modules reduce the chance of secondary harm

## Repository layout

- `cctv/`: early risk recognition using existing CCTV infrastructure
- `drone/`: safe drone-based first-response execution
- `docs/`: competition, poster, and system overview materials
- `integration/`: future end-to-end orchestration layer between CCTV and drone response
- `presentations/`: poster, slide, and figure assets

## Working rule

1. Keep Python source files in each module's `src/` directory as the source of truth.
2. Treat notebooks in `notebooks/` as demo, validation, or Colab-facing artifacts.
3. Treat `scripts/` as notebook generators or reproducibility helpers.
4. Keep large raw datasets, trained weights, and run outputs out of Git when possible.

## Current scope

- CCTV:
  - lightweight YOLO-based risk-object detection demo for `fire` and `knife`
  - temporal verification and human-in-the-loop dispatch recommendation
- Drone:
  - route planning
  - anomaly detection
  - emergency mitigation stack
- Integration:
  - reserved for the higher-level public-safety orchestration layer

## GitHub note

This repository intentionally excludes large raw datasets and most training artifacts from version control.
If you clone this repository, check each module README for expected local files under:

- `cctv/data/raw/`
- `cctv/models/base/`
- `drone/assets/`
