# SportAI

[![codecov](https://codecov.io/gh/rnouteau/SportAI/graph/badge.svg)](https://codecov.io/gh/rnouteau/SportAI)

SportAI is a sports video analysis pipeline. It runs detection / tracking models on a video, stores every prediction in a typed JSON document (`Analysis`), then computes statistics and non-regression metrics on top of that document.

The first implemented sport is **tennis**, with a TrackNet TorchScript **ball** detector, a YOLO **player** detector, and a YOLO11 pose **court** keypoint model (14 ITF points → frame homography). The architecture is designed so that new sports, models and stats can be added without rewriting the orchestrator.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [Repository layout](#2-repository-layout)
3. [Core concepts](#3-core-concepts)
4. [Pipeline flow](#4-pipeline-flow)
5. [Analysis: dataclasses and JSON](#5-analysis-dataclasses-and-json)
6. [Models](#6-models)
7. [Stats](#7-stats)
8. [Configuration](#8-configuration)
9. [Installation](#9-installation)
10. [Running the pipeline](#10-running-the-pipeline)
11. [Tools](#11-tools)
12. [Training and exporting a model](#12-training-and-exporting-a-model)
13. [Tests and baselines](#13-tests-and-baselines)
14. [Lint and pre-commit](#14-lint-and-pre-commit)
15. [GitHub Actions](#15-github-actions)
16. [Hugging Face assets](#16-hugging-face-assets)
17. [How to extend the project](#17-how-to-extend-the-project)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. What this project does

Given a tennis video and a YAML config, SportAI:

1. Opens the video with OpenCV (`VideoReader`).
2. Builds an empty `TennisAnalysis` object (one frame entry per video frame).
3. Instantiates every model listed in the config (e.g. `TennisBallModel`, `TennisPlayerModel`, `TennisCourtModel`).
4. Runs each model. Models write predictions into the analysis frames (`frame.ball`, `frame.players`, `frame.homography`).
5. Optionally runs stats on the filled analysis.
6. Saves the full analysis as a JSON file, or returns the dict when used from tests.

The important design choice: **the analysis is a tree of typed dataclasses with a stable JSON format**. `TennisAnalysis` holds `TennisFrame` objects, which hold `TennisPlayerAtFrame`, `TennisBall` and `TennisHomography` objects. Models and stats read and write plain attributes (`frame.ball = TennisBall(position=[x, y], confidence=c)`). The dict only exists at the edges: `Analysis.from_dict` / `load_json` build the objects from JSON, `to_dict` / `save_json` turn them back into JSON.

---

## 2. Repository layout

```
SportAI/
├── configs/                          # Production pipeline configs
│   └── tennis_config.yaml
├── pipeline/                         # Inference / analysis runtime
│   ├── run.py                        # Orchestrator (CLI entrypoint)
│   ├── video_reader.py               # OpenCV wrapper (read_frame + iter_frames + iter_triplets)
│   ├── device.py                     # resolve_device (config + SPORTAI_DEVICE + CUDA fallback)
│   ├── models/
│   │   ├── model.py                  # Abstract Model[AnalysisT] base
│   │   └── tennis/
│   │       ├── tennis_ball_model.py    # TrackNet TorchScript ball detector
│   │       ├── tennis_player_model.py  # YOLO player detector
│   │       └── tennis_court_model.py   # YOLO11 pose court (14 kpts → homography)
│   ├── objects/                      # Dataclasses of the analysis (from_dict / to_dict)
│   │   ├── analysis.py               # Analysis[FrameT] base (load_json / save_json)
│   │   ├── frame.py / ball.py / player_at_frame.py / homography.py
│   │   ├── player.py / final_stats.py  # Whole-video results (final_stats.players)
│   │   ├── court_schema.py           # CourtSchema ABC (H + image_to_court + distance)
│   │   ├── keypoint_smoother.py      # Rolling median of court keypoints
│   │   ├── position.py               # CourtPosition / ImagePosition
│   │   └── tennis/
│   │       ├── tennis_analysis.py
│   │       ├── tennis_frame.py
│   │       ├── tennis_ball.py
│   │       ├── tennis_player_at_frame.py  # + side ("near" / "far")
│   │       ├── tennis_player.py / tennis_final_stats.py
│   │       ├── tennis_homography.py
│   │       └── tennis_court_schema.py  # TennisCourtSchema(CourtSchema): ITF + H
│   └── stats/
│       ├── stat.py                   # Abstract Stat[AnalysisT] base
│       └── tennis/
│           ├── tennis_legs_court_stat.py  # player.court via H + analysis.court_schema
│           ├── tennis_side_stat.py        # player.side ("near" / "far") on every frame
│           └── tennis_player_selector_stat.py  # keep the 2 players, player_id + final_stats


├── tests/
│   ├── conftest.py                   # --update-baseline, markers from the folder, fixtures
│   ├── helpers.py                    # Builders: synthetic video, analysis, config, MOT clip
│   ├── fakes.py                      # Fake video reader / models / stat (no weights)
│   ├── assets.py                     # Local files or Hugging Face download
│   ├── baseline.py                   # Build / compare metric baselines
│   ├── fixtures/                     # tennis_analysis_sample.json (real pipeline output)
│   ├── evaluation/                   # Evaluators shared by unit / integration / NRT
│   │   ├── metrics.py                # Counts, box IoU, greedy box matching
│   │   ├── base.py                   # Evaluator / VideoEvaluator: samples → rows → TOTAL
│   │   ├── ball_evaluation.py        # BallEvaluator (TP / FP1 / FP2 / FN / TN)
│   │   ├── player_evaluation.py      # PlayerEvaluator (detection + IDF1 + Role accuracy)
│   │   └── court_evaluation.py       # CourtEvaluator (PCK / OKS / keypoint F1, still images)
│   ├── unit/                         # Fast pure-logic tests (marked `unit` automatically)
│   │   ├── objects/  models/  stats/  render/  pipeline/
│   │   ├── evaluation/  nrt/  tools/
│   ├── integration/                  # Synthetic video + fake models (marked `integration`)
│   ├── baselines/tennis/             # One JSON per NRT suite (<suite id>.json)
│   ├── configs/nrt/tennis/           # One YAML per NRT suite (<suite id>.yaml)
│   └── nrt/                          # Slow model NRT (@pytest.mark.nrt)
│       ├── suites.py                 # The suite registry (also used by the wiki renderer)
│       ├── harness.py                # run_suite: config, evaluate, report, baseline
│       ├── run.py                    # python -m tests.nrt.run [suite ...] [--update-baseline]
│       └── tennis/test_tennis_nrt.py # One parametrized test = one suite
├── models/                           # Local pipeline / annotator weights (gitignored)
│   ├── based_models/                 # e.g. yolo11s.pt for player annotator pre-label
│   └── trained_models/               # Optional stable copies for pipeline `weight:` paths
├── tools/                            # Manual annotation / video helpers
│   ├── ball_anotator.py              # Click ball positions on frames
│   ├── player_annotator.py           # Draw player bboxes (+ YOLO pre-annotate)
│   ├── court_yolo.py                 # Shared YOLO-pose encode/decode for court labels
│   ├── court_annotator.py            # Label court keypoints → .txt (+ CSV) + overlay
│   └── video_cuter.py                # Cut clips / export a still frame
├── trains/                           # Training code (not imported by pipeline)
│   └── tennis/
│       ├── player_detection/         # YOLO player (prepare + train + MLflow)
│       └── court_detection/          # YOLO11 pose court (14 kpts, NDJSON + train)
├── utils/
│   ├── paths.py                      # Project root helper
│   ├── read_yaml.py
│   └── video_downloader.py           # yt-dlp helper
├── scripts/
│   └── render_wiki_home.py           # report.json + nrt-history → Wiki Home.md (F1 charts only)
├── .github/
│   ├── actions/publish-wiki/         # Composite action: render + push wiki
│   └── workflows/
│       ├── pr-tests.yml              # CI: unit/lint 3.11–3.12 + integration/coverage 3.11 + nrt
│       └── nrt-nightly.yml           # Full NRT on main (cron + manual)
├── pyproject.toml                    # Ruff + mypy + coverage config
├── .pre-commit-config.yaml           # Local hooks (ruff, mypy, file hygiene)
├── requirements.txt
├── requirements-test.txt             # pytest, hypothesis, pytest-cov, ruff, mypy, pre-commit, HF
└── pytest.ini
```

**Separation of concerns**

| Area | Role | Imported by production? |
|------|------|-------------------------|
| `pipeline/` | Run models on videos, produce analysis JSON | Yes |
| `trains/` | Train and export model weights | No (produces `.pt` artefacts only) |
| `tools/` | Human annotation / video cutting GUIs | No |
| `tests/unit/` | Fast pure-logic tests (no video / model / HF) | CI job `unit` (matrix 3.11–3.12) |
| `tests/integration/` | Synthetic video + fake model + evaluator flow (no HF) | CI job `integration` (3.11) |
| Coverage | unit + integration via `pytest-cov`, fail under 60 % | CI job `coverage` (3.11) + Codecov |
| `tests/nrt/` | Slow model non-regression vs baseline | CI when inference paths change (3.11) |
| Lint / pre-commit | Ruff + mypy (+ file hygiene hooks) | CI job `lint` (matrix 3.11–3.12) |

Production never imports training code. The only link between `trains/` and `pipeline/` is weight files (TorchScript `.pt` for the ball; Ultralytics `.pt` for players and court). Ball TrackNet training/export lives outside this repo today (pipeline only loads the exported TorchScript).

---

## 3. Core concepts

### VideoReader

`pipeline/video_reader.py` wraps `cv2.VideoCapture`.

| Method / attribute | Meaning |
|--------------------|---------|
| `video_path`, `fps`, `width`, `height`, `frame_count` | Video metadata |
| `info()` | Dict written into the analysis JSON under `"video"` |
| `read_frame(i)` | Seek to frame `i` and return a BGR `np.ndarray` (random access) |
| `iter_frames()` | One forward pass: yield `(i, frame)` without per-frame seek |
| `iter_triplets()` | One forward pass: yield `(i, prev, curr, next)` with edge clamping |
| `release()` | Close the capture |

`TennisBallModel` uses `iter_triplets()`; `TennisPlayerModel` / `TennisCourtModel` use `iter_frames()` so YOLO inference does not seek once per frame (major speedup on long clips / NRT).

### Device (`pipeline/device.py`)

All tennis models call `resolve_device(config["device"])`:

1. If `SPORTAI_DEVICE` is set in the environment, it **overrides** the YAML `device` (same knob as NRT/CI).
2. Otherwise use the config value (default `"cpu"`).
3. If the result starts with `cuda` but CUDA is unavailable → fall back to `"cpu"` with a warning.

### Model

Every model inherits from `pipeline.models.model.Model`, which is generic over the analysis type:

```python
class Model(ABC, Generic[AnalysisT]):
    def __init__(self, config: dict) -> None: ...
    def run(self, video_reader: VideoReader, analysis: AnalysisT) -> None: ...


class TennisBallModel(Model[TennisAnalysis]):
    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None: ...
```

- `__init__(config)` loads weights and stores options (`device`, `threshold`, …).
- `run(...)` reads frames from the video and writes predictions into `analysis.frames`.
- The type parameter ties each model to its sport analysis (`TennisAnalysis`, future `FootballAnalysis`, …) so mypy stays aligned with Liskov.

Models are loaded dynamically from the config by fully-qualified class path:

```yaml
- name: pipeline.models.tennis.TennisBallModel
```

`run.py` does `importlib.import_module(...)` then instantiates the class with the config dict. Adding a model means creating a class and referencing it in YAML — no change to `run.py`.

### Stat

Same generic contract as models:

```python
class Stat(ABC, Generic[AnalysisT]):
    def run(self, video_reader: VideoReader, analysis: AnalysisT) -> None: ...


class TennisSideStat(Stat[TennisAnalysis]):
    def run(self, video_reader: VideoReader, analysis: TennisAnalysis) -> None:
        for frame in analysis.frames:
            for player in frame.players:
                court = player.court
                player.side = side_of(court.y if court is not None else None)
```

Stats read filled frames and write derived fields on players/frames (e.g. `player.court`, `player.side`) or whole-video results in `analysis.final_stats`. They do not run neural nets.

### Analysis

`Analysis` is a generic container:

```python
@dataclass
class Analysis(Generic[FrameT]):
    frame_class: ClassVar[type[Frame]]  # class attribute, not a field
    final_stats_class: ClassVar[type[FinalStats]] = FinalStats
    sport: str
    video: dict
    frames: list[FrameT] = field(default_factory=list)
    final_stats: FinalStats = field(default_factory=FinalStats)
    court_schema: CourtSchema | None = field(default=None, compare=False, repr=False)
```

`TennisAnalysis` sets `frame_class = TennisFrame`, `final_stats_class = TennisFinalStats` (and redefines the `final_stats` field with that type), and creates `self.court_schema = TennisCourtSchema()` in `__post_init__`. Empty frames are just `TennisFrame(id=i)`: every other field has a default.

---

## 4. Pipeline flow

```
CLI / test
   │
   ▼
run_pipeline(sport, video_path, config)
   │
   ├─ VideoReader(video_path)
   ├─ load_class(config["analysis"]).from_video(...)
   │      → frames via frame_class(id=i)  (tennis: players=[], ball=None, homography=None)
   │
   ├─ for each model in config["models"]:
   │      load_class(name)(config)
   │      model.run(video_reader, analysis)   → fills ball / players / homography
   │
   ├─ for each stat in config["stats"] (optional):
   │      load_class(name)(config)
   │      stat.run(video_reader, analysis)    → before video_reader.release()
   │
   └─ analysis.save_json(...)   or return analysis.to_dict() if test=True
```

CLI:

```bash
python -m pipeline.run \
  --sport tennis \
  --video_path path/to/clip.mp4 \
  --config configs/tennis_config.yaml
```

From Python / tests:

```python
from pipeline.run import run_pipeline
from utils import read_yaml

data = run_pipeline(
    "tennis",
    "path/to/clip.mp4",
    read_yaml("configs/tennis_config.yaml"),
    test=True,  # return the dict instead of saving
)
```

---

## 5. Analysis: dataclasses and JSON

### Document shape

```json
{
  "sport": "tennis",
  "video": {
    "video_path": "...",
    "frame_count": 825,
    "fps": 30.0,
    "height": 720,
    "width": 1280
  },
  "frames": [
    {
      "id": 0,
      "players": [],
      "ball": null,
      "homography": null
    },
    {
      "id": 3,
      "players": [
        {
          "bbox": [120.0, 200.0, 340.0, 560.0],
          "confidence": 0.91,
          "track_id": 7,
          "player_id": 1,
          "side": "near",
          "legs": { "court": { "x": 1.2, "y": -4.5 } }
        }
      ],
      "ball": {
        "position": [702, 492],
        "confidence": 0.6467
      },
      "homography": {
        "keypoints": [[120.1, 80.2], "... 14 points ..."],
        "confidences": [0.95, "..."],
        "matrix": [[ "... 3x3 image→metres ..." ]]
      }
    }
  ],
  "final_stats": {
    "players": [
      { "id": 1, "name": "player A", "track_ids": [2, 55] },
      { "id": 2, "name": "player B", "track_ids": [5] }
    ]
  }
}
```

Format rules: `ball` and `homography` are always present (`null` when missing). Inside a player, ball or homography, a field whose value is `None` is **left out** of the JSON (no `"track_id": null`): the tracking evaluator relies on `player.get("player_id", player.get("track_id"))`. Old files that still use `"position"` for a player box are read as `bbox`.

`homography` is written by `TennisCourtModel`: 14 court keypoints + confidences, plus an OpenCV `matrix` (3×3) mapping **image pixels → court metres** (origin = net centre) when ≥4 confident points allow `findHomography`.

### Object mapping

| JSON path | Python class | Access pattern |
|-----------|--------------|----------------|
| root | `TennisAnalysis` | `analysis.sport`, `analysis.video`, `analysis.frames`, `analysis.final_stats` |
| `frames[i]` | `TennisFrame` | `frame.id`, `frame.ball`, `frame.players`, `frame.homography` |
| `frames[i].ball` | `TennisBall` | `ball.position`, `ball.confidence`, `ball.velocity` |
| `frames[i].players[j]` | `TennisPlayerAtFrame` | `player.bbox`, `player.confidence`, `player.track_id`, `player.player_id`, `player.side`, `player.feet` (computed: bbox bottom-centre, pixels) |
| `frames[i].players[j].legs.court` | `CourtPosition` | `player.court.x`, `.y` (m), `.distance` (metres to net centre), `.distance_to(other)` |
| `frames[i].homography` | `TennisHomography` | `keypoints`, `confidences`, `matrix` |
| `final_stats.players[k]` | `TennisPlayer` | `id`, `name`, `track_ids` |

Every class is a `@dataclass` with a `from_dict` classmethod and a `to_dict` method. Objects are built once when the analysis is loaded; `frame.players` is a real list (appending to it changes the frame).

### Writing a ball from a model

```python
analysis.frames[i].ball = TennisBall(position=[x, y], confidence=conf)

# Absence of ball
analysis.frames[i].ball = None
```

Reading later:

```python
for frame in analysis.frames:
    if frame.ball:
        print(frame.id, frame.ball.position, frame.ball.confidence)
```

### Persistence

```python
analysis.save_json("outputs/tennis/stats.json")
analysis = TennisAnalysis.load_json("outputs/tennis/stats.json")
```

`save_json` writes `analysis.to_dict()` (NumPy values are converted). `load_json` reads the file and calls `TennisAnalysis.from_dict`. Tests and visualization tools can reload a previous run with the same API. `tests/unit/test_analysis_round_trip.py` checks that `from_dict(...).to_dict()` gives back the sample file `tests/fixtures/tennis_analysis_sample.json` unchanged.

---

## 6. Models

### TennisBallModel

File: `pipeline/models/tennis/tennis_ball_model.py`

| Option | Default | Meaning |
|--------|---------|---------|
| `weight` | required | Path to a TorchScript `.pt` file |
| `device` | `"cpu"` | Via `resolve_device` (see §3); `SPORTAI_DEVICE` overrides |
| `threshold` | `0.5` | Heatmap binarization threshold |
| `input_size` | `(288, 512)` | Model input `(H, W)` |

**Inference logic**

1. Walk the video once via `video_reader.iter_triplets()` → `(i, i-1, i, i+1)` with edge clamping.
2. Convert BGR → RGB, resize to `512×288`, concatenate channels → tensor `(1, 9, 288, 512)`.
3. Run the TorchScript model → 3 heatmaps; keep the middle one (current frame).
4. Extract the largest contour centroid above the threshold.
5. Rescale coordinates from model space back to original video resolution.
6. Write `TennisBall(position=[x, y], confidence=max_heatmap)` into the frame.

**Why TorchScript?**
Ball training/export lives outside this repo. Production only loads the exported graph (`torch.jit.load` + `map_location` via `resolve_device`), which keeps the deployment surface small and stable.

### TennisPlayerModel

File: `pipeline/models/tennis/tennis_player_model.py`

| Option | Default | Meaning |
|--------|---------|---------|
| `weight` | required | Path to an Ultralytics YOLO `.pt` checkpoint |
| `device` | `"cpu"` | Via `resolve_device` (see §3); `SPORTAI_DEVICE` overrides |
| `input_size` | `(640, 640)` | Stored for reference (Ultralytics uses its own imgsz defaults unless overridden) |

**Inference logic**

1. Walk the video once via `video_reader.iter_frames()`.
2. `YOLO.predict(..., classes=[0], stream=True, device=self.device)` (person / player class).
3. For each box, build `TennisPlayerAtFrame` and assign `frame.players`.

Exported from `pipeline.models.tennis` next to `TennisBallModel`. Production config lists all three models; NRT player suite runs player-only.

### TennisCourtModel

File: `pipeline/models/tennis/tennis_court_model.py`

| Option | Default | Meaning |
|--------|---------|---------|
| `weight` | required | Path to an Ultralytics YOLO11 **pose** `.pt` (14 court keypoints) |
| `device` | `"cpu"` | Via `resolve_device` (see §3); `SPORTAI_DEVICE` overrides |
| `input_size` | `(640, 640)` | Stored for reference (Ultralytics imgsz unless overridden) |

**Inference logic**

1. Walk the video once via `video_reader.iter_frames()`.
2. `YOLO.predict(..., classes=[0], stream=True, device=self.device)` (class `tennis court`).
3. Keep the best instance by mean keypoint confidence; read `keypoints.xy` (14×2).
4. Estimate `H` via `analysis.court_schema.estimate_homography` ; set `homo.matrix` on a `TennisHomography`.

NRT court accuracy uses **still images + YOLO-pose labels** via `CourtEvaluator` (not `run_pipeline` on video). Pipeline court inference on video is what fills `frame.homography` in production runs.

---

## 7. Stats

Stats consume a filled `Analysis` and run **after** models, **before** `video_reader.release()`.

**`TennisLegsCourtStat`** (enabled in production config) projects each player's `feet` (bbox bottom-centre) through `frame.homography.matrix` using `analysis.court_schema`, then sets `player.court` → `CourtPosition` persisted as `legs.court: {x, y}` in metres. Access: `player.court.distance`.

```python
for frame in analysis.frames:
    for player in frame.players:
        if player.court:
            print(frame.id, player.court.distance)
```

List stats under `stats:` like models (FQCN + optional `config`):

```yaml
stats:
  - name: pipeline.stats.tennis.TennisLegsCourtStat
    config: {}
```

---

## 8. Configuration

### Production config — `configs/tennis_config.yaml`

```yaml
analysis: pipeline.objects.tennis.TennisAnalysis
models:
  - name: pipeline.models.tennis.TennisBallModel
    config:
      device: "cuda"
      threshold: 0.5
      weight: "./models/TENNIS_BALL_MODEL_04_11_2026.pt"
  - name: pipeline.models.tennis.TennisPlayerModel
    config:
      device: "cuda"
      weight: "./models/TENNIS_PLAYER_MODEL_17_11_2026.pt"
  - name: pipeline.models.tennis.TennisCourtModel
    config:
      device: "cuda"
      weight: "./models/TENNIS_COURT_MODEL_18_11_2026.pt"
stats:
  - name: pipeline.stats.tennis.TennisLegsCourtStat
    config: {}
outputs:
  path: ./outputs/tennis/stats.json
  video:
    path: ./outputs/tennis/overlay.mp4   # omit / null → JSON only
```

| Key | Meaning |
|-----|---------|
| `analysis` | Fully-qualified `Analysis` class path (required; loaded like models) |
| `models[].name` | Fully-qualified Python class path |
| `models[].config` | Passed to `Model.__init__` |
| `stats[].name` | Optional FQCN stats (same load pattern as models; run before release) |
| `outputs.path` | Where `save_json` writes the analysis |
| `outputs.video.path` | Optional annotated MP4 (ball, players/bbox+feet, court). Absent / empty → JSON only |

### NRT configs — `tests/configs/nrt/tennis/`

One file per suite, named after the suite id (`tennis_ball_detection.yaml`, `tennis_players.yaml`, `tennis_court_detection.yaml`). Each extends the production shape with dataset / weights resolution and regression settings. CI overrides `device` via `SPORTAI_DEVICE=cpu`; baselines store `"device": "cpu"`.

**Players** — `tennis_players.yaml` (the same pipeline as production: detector + court + the 3 player stats):

```yaml
analysis: pipeline.objects.tennis.TennisAnalysis
models:
  - name: pipeline.models.tennis.TennisPlayerModel
    config:
      device: "cpu"
      weight: "./models/yolo11s.pt"
  - name: pipeline.models.tennis.TennisCourtModel   # homography for the selector
    config:
      device: "cpu"
      weight: "./models/TENNIS_COURT_MODEL_18_11_2026.pt"
    weights:                                         # per-model Hugging Face weights
      repo_id: rnouteau/sportai-models
      filename: tennis/court/TENNIS_COURT_MODEL_18_11_2026.pt
stats:
  - name: pipeline.stats.tennis.TennisLegsCourtStat
  - name: pipeline.stats.tennis.TennisSideStat
  - name: pipeline.stats.tennis.TennisPlayerSelectorStat
dataset:
  local_dir: ./datasets/tennis/player_tracking/test_regression
  repo_id: rnouteau/sportai-test-data
  revision: v4
  subset: tennis/player_tracking/test_regression
weights:
  repo_id: rnouteau/sportai-models
  filename: tennis/player/yolo11s.pt
evaluation:
  iou_threshold: 0.5
regression:
  baseline: tests/baselines/tennis/tennis_players.json
  metrics: [Precision, Recall, F1-Score, Mean IoU, IDF1, ID switches, MOTA, Role accuracy]
  tolerance: 0.005
```

**Ball** — `tennis_ball_detection.yaml`: `TennisBallModel` only, `evaluation.min_dist: 10` (pixels), metrics `[Accuracy, Precision, Recall, F1-Score]`, tolerance 0.01.

**Court** — `tennis_court_detection.yaml`: still images through `TennisCourtModel` (no `run_pipeline`, no `analysis:` key), `evaluation: {pck_thresholds: [0.05, 0.1], match_threshold: 0.05, oks_sigma: 0.05}`, metrics `[PCK@0.05, PCK@0.1, OKS, Precision, Recall, F1-Score]`, tolerance 0.01.

Asset resolution (`tests/assets.py`):

1. If the local path exists → use it (dev machine).
2. Otherwise download from Hugging Face Hub (CI).

---

## 9. Installation

### Requirements

- Python 3.11+ (CI matrix: 3.11 / 3.12 on `unit` + `lint`; NRT and local default: 3.11)
- (Optional) CUDA-capable GPU for faster inference
- `ffmpeg` in `PATH` if you use the precise mode of `video_cuter.py`

### Setup

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

For tests, lint, coverage, and CI extras (`pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`, `pre-commit`, HF hub, …).
These pins require **Python ≥ 3.11** (`pre-commit` / `mypy` no longer install on 3.10):

```bash
pip install -r requirements-test.txt
pre-commit install   # once per clone: run hooks on every git commit
```

On CI (and optionally locally for CPU-only), install the CPU build of PyTorch first:

```bash
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-test.txt
```

### Weights

| Role | Path | Notes |
|------|------|--------|
| Ball (TorchScript) | `models/TENNIS_BALL_MODEL_04_11_2026.pt` | Loaded by `TennisBallModel` via `torch.jit.load` |
| Players (YOLO) | `models/TENNIS_PLAYER_MODEL_17_11_2026.pt` | Loaded by `TennisPlayerModel` (HF: `tennis/player/...`) |
| Court (YOLO pose) | `models/TENNIS_COURT_MODEL_18_11_2026.pt` | Loaded by `TennisCourtModel` (HF: `tennis/court/...`) |
| Player YOLO pretrained (annotator) | `models/based_models/yolo11s.pt` | Used by `tools/player_annotator.py` pre-label |
| Player train based / stable | `trains/tennis/player_detection/models/{based_models,trained_models}/` | From `configs/train.yaml` |
| Court train based / stable | `trains/tennis/court_detection/models/{based_models,trained_models}/` | e.g. `yolo11s-pose.pt`, `tennis_court_yolo11s_pose.pt` |

Or point `weight:` in the pipeline config to another path.
`.pt` files and the root `/models/` tree are gitignored (dirs kept via `.gitkeep`); `pipeline/models/` stays tracked. In CI, NRT weights are downloaded from Hugging Face.

---

## 10. Running the pipeline

```bash
python -m pipeline.run \
  --sport tennis \
  --video_path datasets/tennis/ball_detection/test/my_clip.mp4 \
  --config configs/tennis_config.yaml
```

Output: `outputs/tennis/stats.json` plus optional `outputs/tennis/overlay.mp4` when `outputs.video.path` is set (ball circle + conf, player bboxes + ids, feet point + court distance, court lines via H⁻¹ or keypoint crosses).

Useful checks after a run:

```python
from pipeline.objects.tennis.tennis_analysis import TennisAnalysis

analysis = TennisAnalysis.load_json("outputs/tennis/stats.json")
detected = sum(1 for f in analysis.frames if f.ball)
print(f"{detected}/{len(analysis.frames)} frames with a ball")
```

---

## 11. Tools

### Ball annotator — `tools/ball_anotator.py`

GUI to label ball positions frame by frame.

```bash
python tools/ball_anotator.py --video_path path/to/clip.mp4
```

| Key | Action |
|-----|--------|
| Left click | Set ball position on the current frame |
| `←` / `→` | Previous / next frame |
| `d` | Delete annotation on current frame |
| `s` | Export CSV next to the video (`clip.csv`) |
| `q` | Quit |

CSV format:

```
frame,x,y
0,706.0,480.0
1,704.0,484.0
2,,
3,703.0,493.0
```

- Annotated frames have `x,y`.
- Frames without a ball (or not annotated as present) are exported as empty cells → read as `NaN` by pandas.
- The test suite evaluates **all** CSV rows. `NaN` ground truth + prediction = true negative; `NaN` ground truth + prediction present = false positive.

**Coordinate note:** the annotator displays the frame centered in a 1280×720 canvas without resizing. Pixel coordinates match the video only when the video is 1280×720. For other resolutions, convert using the canvas offset `(640 - width/2, 360 - height/2)`.

### Player annotator — `tools/player_annotator.py`

GUI to label player bounding boxes. If no CSV exists next to the video, it pre-annotates with YOLO (`models/based_models/yolo11s.pt`, max 2 boxes per frame).

```bash
python tools/player_annotator.py --video_path path/to/clip.mp4
```

| Key | Action |
|-----|--------|
| Drag (click + release) | Add a bbox on the current frame |
| `←` / `→` | Previous / next frame |
| `d` | Delete the bbox under the mouse |
| `s` | Export CSV next to the video (`clip_players.csv`) |
| `q` | Quit |

CSV format (one row per box):

```
frame,class_id,x0,y0,x1,y1
0,0,120.0,200.0,340.0,560.0
```

### Court annotator — `tools/court_annotator.py`

GUI to label **14 ITF tennis court keypoints** on a still image. Schema / flip map live only in [`pipeline/objects/tennis/tennis_court_schema.py`](pipeline/objects/tennis/tennis_court_schema.py) (shared with train prepare/import and NRT eval). Named points + fixed edges; `v` projects via `TennisCourtSchema().estimate_court_to_image`.

```bash
python tools/court_annotator.py --image_path path/to/frame.png
```

| Key | Action |
|-----|--------|
| Click | Place / replace the active point (image coords; Unicode-safe load) |
| Click near an existing point | Select that point |
| `←` / `→` | Previous / next keypoint id |
| `d` | Delete the active point |
| `s` | Save **YOLO-pose** `{stem}.txt` (primary) + secondary `{stem}_court.csv` |
| `v` | Homography overlay (green lines + crosses) in an OpenCV window |
| `q` | Quit |

Reload prefers `{stem}.txt` if present, else legacy `_court.csv`. Encode/decode helpers: [`tools/court_yolo.py`](tools/court_yolo.py).

Point order (guided, far→near): far baseline (DL/SL/SR/DR) → far service + T → near service + T → near baseline (DL/SL/SR/DR). No net-centre keypoint (net is drawn only in the overlay).

### Video cuter — `tools/video_cuter.py`

```bash
python tools/video_cuter.py --video_path path/to/match.mp4
```

Select a start/end frame with the two sliders, then use the File menu:

- **New cut (fast, no re-encode)** — `ffmpeg -c copy` when available.
- **New cut (precise, re-encode)** — re-encodes with libx264 for frame-accurate cuts.
- **Export current frame as image** — writes `{video}_frame_{n}.png` next to the source (shortcut `e`).

Keyboard: arrows step 1 frame, Shift+arrows ≈ 1 second, Ctrl+arrows ≈ 10 seconds.
### Video downloader — `utils/video_downloader.py`

Thin `yt-dlp` script to download source matches into `downloads/`. Edit `URL` / `MAX_HEIGHT` at the top of the file, then:

```bash
python utils/video_downloader.py
```

---

## 12. Training and exporting a model

Training packages under `trains/` are **not** imported by `pipeline/`.

### Ball detection (TrackNet)

Ball **training / TorchScript export are not in this repo** under `trains/` today. Pipeline inference only needs the exported TorchScript at e.g. `models/TENNIS_BALL_MODEL_04_11_2026.pt` (or the HF path `tennis/ball/...`).

### Player detection (YOLO)

Code: `trains/tennis/player_detection/` (dataset prep + Ultralytics train + MLflow).

1. Put clips and annotations under `datasets/tennis/player_detection/videos/`
   (`clip.mp4` + `clip_players.csv` from the player annotator).
2. Place the pretrained checkpoint at
   `trains/tennis/player_detection/models/based_models/yolo11s.pt`
   (the player annotator may still use `models/based_models/yolo11s.pt` for pre-label).
3. Train from the package directory:

```bash
cd trains/tennis/player_detection
python train.py --config configs/train.yaml
```

What it does:

- Builds a YOLO dataset under `datasets/tennis/player_detection/yolo/` (seeded train/val split by video, or temporal chunks if a single video).
- Fine-tunes from `trains/tennis/player_detection/models/based_models/`, writes Ultralytics run dirs under `.../models/trained_models/<name>/`.
- Copies `best.pt` to the stable path
  `trains/tennis/player_detection/models/trained_models/tennis_player_yolo11s.pt`.
- Logs metrics to SQLite `mlflow.db` when `mlflow.enabled: true` in the config.

Inspect runs:

```bash
# from the SportAI repo root
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

After training, copy / upload the stable weights so `TennisPlayerModel` can load them (local `models/` or Hugging Face `tennis/player/...`).

### Court detection (YOLO11 pose, 14 keypoints)

Code: `trains/tennis/court_detection/` (NDJSON import + optional prepare + Ultralytics pose train + MLflow).

Uses the same **14** ITF keypoints / `FLIP_IDX` from `pipeline.objects.tennis.tennis_court_schema` only (no local copies). Task is **pose** (1 class `tennis court` + 14 kpts), not plain detect.

**Primary path — Ultralytics NDJSON export**

1. Place the NDJSON under `datasets/tennis/court_detection/` (default name in config:
   `tennis-court-keypoint-detection (2).ndjson`).
2. Place pretrained COCO pose weights at
   `trains/tennis/court_detection/models/based_models/yolo11s-pose.pt`.
3. Train (downloads images from CDN URLs into `yolo/images/{train,val,test}/`):

```bash
cd trains/tennis/court_detection
python train.py --config configs/train.yaml
```

`pipeline.import_ndjson: true` writes YOLO labels + `data.yaml` with `kpt_shape: [14, 3]` and a corrected L/R `flip_idx` (the NDJSON identity flip map is not used).

**Optional — extra manual images**

Annotate with the court annotator into `datasets/tennis/court_detection/images/` (writes sibling YOLO `{stem}.txt`). Set `pipeline.prepare_dataset: true` (and usually `import_ndjson: false`). `prepare_dataset` prefers existing `.txt`; falls back to legacy `*_court.csv`.

What training does:

- Instantiates `configs/yolo11s-court-pose.yaml` (14 kpts), loads `yolo11s-pose.pt`, then trains.
- Applies **train-time** augmentation from `train.augment` (HSV / mild translate-scale / `fliplr`; mosaic/mixup/rotation/perspective off).
- Copies `best.pt` to
  `trains/tennis/court_detection/models/trained_models/tennis_court_yolo11s_pose.pt`
  and logs to `mlflow.db`.

Quick visual check after training:

```bash
cd trains/tennis/court_detection
python scripts/predict_overlay.py `
  --weights models/trained_models/tennis_court_yolo11s_pose.pt `
  --image_path ../../../datasets/tennis/court_detection/yolo/images/val/some.jpg `
  --show
```

Copy or upload the stable pose weights to the pipeline path (e.g. `models/TENNIS_COURT_MODEL_18_11_2026.pt` or HF `tennis/court/...`) for `TennisCourtModel`.

### Court NRT (local)

Layout expected by the evaluator (YOLO-pose stills, 14 kpts):

```
datasets/tennis/court_detection/test/
  *.jpg|png          # or images/
  *.txt              # sibling labels, or labels/ with same stems
```

Each label line: `cls xc yc w h` then `x y v` × 14 (normalized), same as the NDJSON import. Metrics are described in §13.

```bash
$env:SPORTAI_DEVICE="cpu"   # or cuda
python -m tests.nrt.run tennis_court_detection
# first time / intentional model change:
python -m tests.nrt.run tennis_court_detection --update-baseline
```

Reports: `outputs/tests/tennis_court_detection/report.md` (+ `.json`). Baseline: `tests/baselines/tennis/tennis_court_detection.json`.

---

## 13. Tests and baselines

Three levels, one marker each (see `pytest.ini`). `unit` and `integration` are added automatically from the folder (`tests/conftest.py`): no decorator to write.

| Marker | Folder | Speed | Needs real video / weights / HF? | Purpose |
|--------|--------|-------|----------------------------------|---------|
| `unit` | `tests/unit/` | Seconds | No | Pure logic: objects and JSON, models with a fake YOLO, stats, evaluators, harness, wiki |
| `integration` | `tests/integration/` | Seconds | No (synthetic OpenCV video + fake models) | `VideoReader`, `run_pipeline` + stats + overlay, evaluators end to end |
| `nrt` | `tests/nrt/` | Minutes on CPU | Yes | Real models on annotated clips / images vs committed baselines |

```bash
python -m pytest tests/unit -m unit -q                 # while coding
python -m pytest tests/integration -m integration -q
$env:SPORTAI_DEVICE="cpu"                               # PowerShell
python -m pytest tests/nrt -m nrt -s -rs               # every NRT suite
```

### Layout of the tests

- `tests/unit/<area>/` mirrors the code: `objects/` (serialization, court schema, keypoint smoother), `models/`, `stats/`, `render/`, `pipeline/`, `evaluation/`, `nrt/` (harness, baseline, wiki), `tools/` (annotators).
- `tests/unit/objects/test_serialization.py` locks the JSON format: lossless round trip of `tests/fixtures/tennis_analysis_sample.json`, empty frames, `None` fields left out of a player, the old `"position"` key read as `bbox`.
- Shared builders are in `tests/helpers.py` (`make_synthetic_video`, `make_tennis_analysis`, `pipeline_config`, `write_mot_clip`) and `tests/fakes.py` (`FakeVideoReader`, `FakeBallModel`, `FakeTrackingModel`, `FakeSceneModel`, `BallCountStat`). Fake models are loaded by `run_pipeline` from their dotted name, e.g. `tests.fakes.FakeTrackingModel`. Fixture `synthetic_video(n_frames=...)` writes a tiny mp4 in `tmp_path`.
- `tests/integration/test_run_pipeline.py` runs the 3 player stats on a fake tracker + homography and checks `player_id`, `side` and `final_stats`.

### Coverage (`pytest-cov`)

Config: [`pyproject.toml`](pyproject.toml). Measured: `pipeline/`, `tests.evaluation`, `tests.baseline`; omitted: `tennis_ball_model.py` (NRT-only TrackNet) and `pipeline/stats/**`. CI threshold: 60 %.

```bash
python -m pytest tests/unit tests/integration -m "unit or integration" `
  --cov=pipeline --cov=tests.evaluation --cov=tests.baseline `
  --cov-report=term-missing --cov-fail-under=60
```

### Philosophy: baseline comparison, not fixed thresholds

The NRT does **not** assert “F1 > 0.80”. It asserts:

> Metrics for every video (and TOTAL) must stay within `tolerance` of the committed baseline.

An improvement that moves a metric outside the tolerance also fails the test. That is intentional: any behaviour change must be reviewed by regenerating and committing the baseline JSON. The git diff of that file shows exactly which videos and metrics moved.

### The NRT suites

Registered once in [`tests/nrt/suites.py`](tests/nrt/suites.py) (id, title, config, evaluator, report columns). The same registry drives the pytest test, `python -m tests.nrt.run` and the wiki.

| Suite id | Evaluator | Annotation | Samples |
|----------|-----------|------------|---------|
| `tennis_ball_detection` | `BallEvaluator` | `<video>.csv` (`frame, x, y`) | clips |
| `tennis_players` | `PlayerEvaluator` | `<clip dir>/gt/gt.txt` (MOT) | clips |
| `tennis_court_detection` | `CourtEvaluator` | YOLO-pose `.txt` | still images |

All evaluators share [`tests/evaluation/base.py`](tests/evaluation/base.py): list the samples, score each one (one row), combine the rows (TOTAL). Counts are summed for TOTAL, so Precision / Recall / F1 of TOTAL are computed on all the samples together.

### Metric definitions — ball (`BallEvaluator`)

For each annotated frame:

| Case | Count |
|------|-------|
| Prediction within `min_dist` px of the annotation | TP |
| Prediction present but too far | FP1 |
| Prediction present, no ball annotated | FP2 |
| No prediction, no ball annotated | TN |
| No prediction, ball annotated | FN |

```
Accuracy  = (TP + TN) / Total        (at most one ball per frame, so TN makes sense)
Precision = TP / (TP + FP)            where FP = FP1 + FP2
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

### Metric definitions — players (`PlayerEvaluator`)

One pipeline run, one MOT annotation (annotated ids: 1 = near player, 2 = far player, the player annotator convention), three groups of metrics:

| Group | Metric | Meaning |
|-------|--------|---------|
| Detection | Precision, Recall, F1-Score | Boxes matched greedily by IoU ≥ `iou_threshold` per frame (TP / FP / FN) |
| Detection | Mean IoU | Average IoU of the matched boxes: how well they fit |
| Identity | IDF1 | Share of the time each player keeps one id (motmetrics) |
| Identity | ID switches, MOTA | Times an id jumps to another person; misses + FP + switches |
| Role | Role accuracy | Share of matched boxes whose `player_id` equals the annotated id |

The id used for the identity metrics is `player_id` (selector) when present, else `track_id`. IDF1 only checks that ids are **stable**: a pipeline calling the near player “2” all along still gets IDF1 = 1. Role accuracy is what checks that P1 really is the near player.

There is no Accuracy (TN would mix frames and boxes) and no mAP (the pipeline output has no score-ranked candidate list).

### Metric definitions — court (`CourtEvaluator`)

Matching **by keypoint id** (0–13) at `match_threshold` × box size: TP / FP / FN → Precision, Recall, F1. Plus `PCK@t` (share of visible keypoints closer than `t` × box size) and `OKS`, averaged over the images for TOTAL.

### Baseline files

`tests/baselines/tennis/<suite id>.json`:

```json
{
  "context": {
    "device": "cpu",
    "dataset_revision": "v4",
    "TennisPlayerModel.weights": "yolo11s.pt",
    "TennisCourtModel.weights": "TENNIS_COURT_MODEL_18_11_2026.pt",
    "stats": "TennisLegsCourtStat, TennisSideStat, TennisPlayerSelectorStat",
    "evaluation.iou_threshold": 0.5
  },
  "videos": {
    "clip.mp4": { "Precision": 0.9912, "IDF1": 0.9912, "Role accuracy": 1.0, "...": "..." },
    "TOTAL": { "...": "..." }
  }
}
```

The `context` lists everything the numbers depend on: the weights **of every model**, their settings (e.g. `TennisBallModel.threshold`), the stats and the evaluation settings. It is compared first: if anything differs, the test fails immediately and asks for a new baseline instead of dumping metric drifts caused by a config change.

**CPU vs GPU:** floating-point results are not bit-identical across devices. The CI baseline is generated on CPU (`SPORTAI_DEVICE=cpu`). Always regenerate baselines on the same device class that will verify them.

### Running the NRT locally

```bash
$env:SPORTAI_DEVICE="cpu"                                   # PowerShell
python -m pytest tests/nrt/tennis -m nrt -s -rs             # every suite
python -m pytest tests/nrt/tennis -m nrt -s -k players      # one suite
python -m tests.nrt.run tennis_players                      # same, without pytest
```

First run without a baseline (or with `--update-baseline`) writes the baseline(s) and **skips** so you cannot mistake “no comparison” for “all green”:

```bash
python -m pytest tests/nrt/tennis -m nrt -s --update-baseline
```

Then commit the files of `tests/baselines/tennis/`. Reports: `outputs/tests/<suite id>/report.md` (current, baseline, delta and status per video and metric, plus the counts) and `report.json` (read by the wiki).

### When the model or post-processing changes

1. The NRT turns red and lists the deltas.
2. If the change is intentional, regenerate:
   - locally: `--update-baseline` with `SPORTAI_DEVICE=cpu`, or
   - on CI: Actions → **Tests** → Run workflow → check **Regenerate the baseline**.
3. Download / copy the new JSON into `tests/baselines/` and commit it in the same PR.
4. Reviewers see the metric diffs in the PR.

---

## 14. Lint and pre-commit

Config lives in [`pyproject.toml`](pyproject.toml) (Ruff + mypy) and [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

| Tool | Scope | What it does |
|------|-------|--------------|
| Ruff check | `pipeline/`, `tests/`, `scripts/`, `utils/` | Lint (E/W/F/I/B/UP); `trains/` and `tools/` excluded |
| Ruff format | same | Consistent formatting (`line-length = 100`) |
| Mypy | `pipeline/` only | Type-check; `ignore_missing_imports = true`; not `--strict` yet |
| pre-commit hooks | staged files | `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files`, then ruff + mypy |

Local commands:

```bash
# Same checks as the CI lint job
ruff check pipeline tests scripts utils
ruff format --check pipeline tests scripts utils
mypy --config-file=pyproject.toml pipeline

# Or via pre-commit (after pre-commit install)
pre-commit run --all-files
```

The CI jobs **`unit`**, **`integration`**, **`coverage`** and **`lint`** run in parallel (`unit`/`lint` on 3.11 and 3.12). The NRT job waits for all of them to succeed — if any matrix cell fails, NRT is skipped even when model paths changed.

---

## 15. GitHub Actions

Two workflows:

| Workflow | File | Role |
|----------|------|------|
| **Tests** | [`.github/workflows/pr-tests.yml`](.github/workflows/pr-tests.yml) | `unit`/`lint` matrix 3.11–3.12; `integration`/`coverage` on 3.11; NRT filtered |
| **NRT Nightly** | [`.github/workflows/nrt-nightly.yml`](.github/workflows/nrt-nightly.yml) | Full NRT on `main` every night (Python 3.11, no path filter) |

Model scores from the latest NRT on `main` are published to the repository **Wiki → Home** page (auto-generated charts from ball + player + court `report.json` files).

### Tests (`pr-tests.yml`)

| Trigger | When |
|---------|------|
| `pull_request` | Path-filtered: `pipeline/`, `tests/`, `configs/`, `utils/`, requirements, `pytest.ini`, `pyproject.toml`, `.pre-commit-config.yaml`, or the workflow itself |
| `push` to `main` | **Any** push (no path filter) — unit/integration/coverage/lint always; NRT still gated by the `changes` path filter |
| `workflow_dispatch` | Manual run; optional `update_baseline` input (NRT job only) |

### Jobs

```mermaid
flowchart LR
    pr[PR_or_push] --> unit[job_unit]
    pr --> integration[job_integration]
    pr --> coverage[job_coverage]
    pr --> lint[job_lint]
    pr --> changes[job_changes]
    unit --> nrt[job_nrt]
    integration --> nrt
    coverage --> nrt
    lint --> nrt
    changes -->|"nrt paths changed?"| nrt
```

**1. `unit` (always, ~seconds)**

- Matrix: Python **3.11 / 3.12** (`fail-fast: false`)
- Install `requirements-test.txt` (no separate CPU torch install)
- `python -m pytest tests/unit -m unit -q`
- Fails the PR quickly if pure logic breaks

**2. `integration` (always, parallel, Python 3.11 only)**

- `python -m pytest tests/integration -m integration -q`
- Synthetic video + fake model; no HF token required

**3. `coverage` (always, parallel, Python 3.11 only)**

- Re-runs unit + integration with `pytest-cov`
- Fails if total coverage < 60 % (omits TrackNet model + stats; see §13)
- Uploads `coverage.xml` artefact + Codecov

**4. `lint` (always, parallel)**

- Matrix: Python **3.11 / 3.12** (`fail-fast: false`)
- `ruff check` + `ruff format --check` on `pipeline/`, `tests/`, `scripts/`, `utils/`
- `mypy --config-file=pyproject.toml pipeline`
- Same install as unit (`requirements-test.txt`)

**5. `changes` (path filter)**

Uses `dorny/paths-filter`. Sets `nrt=true` only when one of these changes:

- `pipeline/models/**`
- `pipeline/run.py`
- `pipeline/video_reader.py`
- `tests/nrt/**`
- `tests/configs/nrt/**`
- `tests/baselines/**`
- `tests/evaluation/**`
- `requirements*.txt`
- `.github/workflows/pr-tests.yml`

A PR that only touches `pipeline/objects/` or `tests/baseline.py` / `tests/unit/` runs **unit + integration + coverage + lint** (no NRT).

**6. `nrt` (slow, conditional, Python 3.11 only)**

Runs only if:

- `unit`, `integration`, `coverage` and `lint` succeeded, **and**
- `workflow_dispatch` **or** the path filter reported NRT-relevant changes

Then:

1. Install CPU `torch` + `torchvision` + `requirements-test.txt` (Python 3.11)
2. Cache `~/.cache/huggingface` (keyed on ball / players / court NRT configs)
3. `pytest tests/nrt/tennis -m nrt` with `SPORTAI_DEVICE=cpu` (ball + players + court)
4. Append ball / players / court `report.md` files to the GitHub Step Summary
5. Upload artefacts: all three report folders + baseline JSON files
6. On `main` only: publish Wiki Home via [`.github/actions/publish-wiki`](.github/actions/publish-wiki) (ball + player + court reports)

`update_baseline` from `workflow_dispatch` is passed only to this job. Timeout is **90** minutes.

Required repository secrets:

| Secret | Purpose |
|--------|---------|
| `HF_TOKEN` | Read-only Hugging Face token to download the private dataset and model weights |
| `WIKI_TOKEN` | PAT that can push to the repo wiki (`repo` classic, or fine-grained with wiki write) |
| `CODECOV_TOKEN` | Codecov upload token (needed for private repos; optional for public) |

If `HF_TOKEN` is missing (e.g. PR from a fork), the NRT step is skipped with a warning instead of failing.

### NRT Nightly (`nrt-nightly.yml`)

Runs the full tennis NRT (ball + players + court) on the default branch (`main`) without the path filter.

| Trigger | When |
|---------|------|
| `schedule` | `0 2 * * *` (02:00 UTC daily) |
| `workflow_dispatch` | Manual run from the Actions tab |

Same setup as the NRT job in **Tests** (Python **3.11**, CPU torch, HF cache, pytest, step summary, artefacts, **wiki Home publish**). Artefact name is `tennis-nrt-reports-nightly` (PR job uses `tennis-nrt-reports`). If `HF_TOKEN` is missing, the workflow **fails** (unlike PR forks, where a skip is acceptable).

### Wiki Home (model results)

After each NRT on `main` (nightly, push, or manual), CI regenerates the wiki from `outputs/tests/<suite id>/report.json` with [`scripts/render_wiki_home.py`](scripts/render_wiki_home.py) and [`.github/actions/publish-wiki`](.github/actions/publish-wiki). The suites are those of [`tests/nrt/suites.py`](tests/nrt/suites.py).

Pages: `Home.md` (one row per suite: key metric, baseline, status, trend), one page per suite (metrics vs baseline, per-video table, context, history chart) and `_Sidebar.md`.

History is stored in the wiki repo as `nrt-history.json` (not in SportAI git). Old suite ids are mapped to the new ones (`tennis_ball_tracking` → `tennis_ball_detection`, `tennis_player_tracking` → `tennis_players`); the history of the former `tennis_player_detection` suite stays in the file but is no longer shown. **PR NRT runs do not update the wiki.**

One-time bootstrap:

1. Repo Settings → Features → enable **Wikis**
2. Create any first page in the Wiki UI (so `*.wiki.git` exists)
3. Add repository secret `WIKI_TOKEN` (PAT with wiki push access)

If `WIKI_TOKEN` is missing or `report.json` was not produced, the publish step warns and skips.

---

## 16. Hugging Face assets

Large binaries are not in git (see `.gitignore`: `*.mp4`, `*.csv`, `*.pt`, `datasets/`).

Expected Hub layout:

```
rnouteau/sportai-test-data          # dataset (private)
├── tennis/ball_detection/test/     # *.mp4 + *.csv
├── tennis/player_detection/test/   # *.mp4 + *_players.csv
└── tennis/court_detection/test/    # images + YOLO-pose *.txt (14 kpts)

rnouteau/sportai-models             # model repo (private)
├── tennis/ball/TENNIS_BALL_MODEL_04_11_2026.pt
├── tennis/player/TENNIS_PLAYER_MODEL_17_11_2026.pt
└── tennis/court/TENNIS_COURT_MODEL_18_11_2026.pt
```

Upload example (Hugging Face CLI `hf` ≥ 2.x):

```bash
pip install -U huggingface_hub
hf auth login

hf repos create rnouteau/sportai-test-data --repo-type dataset --private
hf upload rnouteau/sportai-test-data datasets/tennis/ball_detection/test tennis/ball_detection/test --repo-type dataset
hf upload rnouteau/sportai-test-data datasets/tennis/player_detection/test tennis/player_detection/test --repo-type dataset
hf upload rnouteau/sportai-test-data datasets/tennis/court_detection/test tennis/court_detection/test --repo-type dataset
hf repos tag create rnouteau/sportai-test-data v3 --repo-type dataset

hf repos create rnouteau/sportai-models --private
hf upload rnouteau/sportai-models models/TENNIS_BALL_MODEL_04_11_2026.pt tennis/ball/TENNIS_BALL_MODEL_04_11_2026.pt
hf upload rnouteau/sportai-models models/TENNIS_PLAYER_MODEL_17_11_2026.pt tennis/player/TENNIS_PLAYER_MODEL_17_11_2026.pt
hf upload rnouteau/sportai-models models/TENNIS_COURT_MODEL_18_11_2026.pt tennis/court/TENNIS_COURT_MODEL_18_11_2026.pt
```

NRT configs pin dataset revisions separately: **ball → `v1`**, **players → `v2`**, **court → `v3`**. Tag new dataset snapshots when you change a subset.

For GitHub Actions, create a **read-only** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and store it as the `HF_TOKEN` repository secret.

Keep test clips short (5–15 seconds). TrackNet on CPU is only a few frames per second; YOLO player NRT is lighter per frame but still adds wall time on long clips. Court NRT is image-based (still adds wall time for pose inference over the test set).

---

## 17. How to extend the project

### Add a new model

Follow the existing patterns (`TennisBallModel`, `TennisPlayerModel`, `TennisCourtModel`):

1. Create `pipeline/models/tennis/tennis_<thing>_model.py` inheriting from `Model[TennisAnalysis]`.
2. Implement `run(self, video_reader, analysis: TennisAnalysis)`.
3. In `run`, write into frames (extend `TennisFrame` / analysis JSON as needed).
4. Export the class from `pipeline/models/tennis/__init__.py`.
5. Add an entry under `models:` in the YAML.

No change to `run.py` for a new *tennis* model.

### Add a new stat

1. Create `pipeline/stats/tennis/my_stat.py` inheriting from `Stat[TennisAnalysis]` (`__init__(self, config=None)`).
2. Implement `run(self, video_reader, analysis: TennisAnalysis)`.
3. Prefer writing derived fields on **frames / players** (see `TennisLegsCourtStat` → `player.court`). A whole-video KPI goes in `analysis.final_stats` as a typed field (e.g. a new field on `TennisPlayer` or `TennisFinalStats`, with `from_dict` / `to_dict`), not as a free dict key.
4. Export it and list it under `stats:` in the config (loop already wired in `run.py`).

### Add a new sport

1. Add `pipeline/objects/<sport>/` with `<Sport>Analysis` (`frame_class`, `final_stats_class`), `<Sport>Frame` (extra fields with defaults + `from_dict` / `to_dict`), and domain objects, all as `@dataclass`.
2. Point `analysis:` in `configs/<sport>_config.yaml` at the new Analysis FQCN (`run_pipeline` loads it like models — no edit to `run.py`).
3. Add `pipeline/models/<sport>/` and `pipeline/stats/<sport>/`.
4. Add its NRT suites (see *Add a new NRT suite* below).

### Add a new NRT suite

1. Write the evaluator in `tests/evaluation/`, subclassing `VideoEvaluator` (clips through `run_pipeline`) or `Evaluator` (other samples): `ground_truth_path` / `samples`, `evaluate_frames` / `evaluate`, `total`.
2. Add `tests/configs/nrt/<sport>/<suite id>.yaml` (its `regression.baseline` is `tests/baselines/<sport>/<suite id>.json`).
3. Register the suite in [`tests/nrt/suites.py`](tests/nrt/suites.py): the pytest test, `python -m tests.nrt.run` and the wiki pick it up. `tests/unit/nrt/test_harness.py` checks that config, evaluator and baseline agree.
4. Generate the baseline on **CPU** (`SPORTAI_DEVICE=cpu`, `--update-baseline`) and commit it.

---

## 18. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Want CPU while YAML says `cuda` | Config device vs machine | Set `SPORTAI_DEVICE=cpu` (overrides YAML via `resolve_device` in all tennis models + NRT). Without CUDA, `cuda` already falls back to CPU with a warning. |
| `FileNotFoundError` on weights in CI | Missing / wrong HF path or token | Verify `weights.filename`, repo visibility, and `HF_TOKEN`. |
| Baseline comparison fails after switching GPU ↔ CPU | Different floating-point paths | Regenerate the baseline on the same device class as CI (`cpu`). |
| `Results differ from the baseline` after a tiny code change | Expected (model / post-process / I/O edits) | Inspect `report.md`, then `--update-baseline` on CPU if the new behaviour is wanted. |
| Annotator coordinates disagree with predictions | Video not 1280×720 (ball annotator canvas) | Convert canvas coords using the centering offset, or resize/export clips at 1280×720. |
| NRT still slow on CPU | TrackNet on CPU is heavy (even with `iter_triplets` / `iter_frames`) | Prefer a short clip; GPU locally; CI stays on CPU for baseline stability. |
| NRT job skipped even though model paths changed | Upstream jobs red, or path filter | `unit` / `integration` / `coverage` / `lint` must all be green. Path filter does **not** include `pipeline/objects/**`. |
| `player.court` is `None` | No `homography.matrix`, or stat missing | Need ≥4 confident court kpts for H; enable `pipeline.stats.tennis.TennisLegsCourtStat` in YAML. |
| `No matching distribution for pre-commit==4.7.0` on 3.10 | Pins require Python ≥ 3.11 | Use 3.11+ locally; CI matrix is **3.11 / 3.12** only. |
| `Coverage failure: total … less than fail-under=60` | Real drop, or wrong `--cov` path | Use `--cov=tests.baseline` (module). Omits are in `pyproject.toml`. |
| `from tests...` import fails | Another installed package named `tests` | Run via `python -m pytest` from the repo root (`pytest.ini` sets `pythonpath = .`). |
| pre-commit fails then “files were modified” | Auto-fix hooks (whitespace, ruff format) | Re-stage and commit again; or run `pre-commit run --all-files` once. |
| mypy `[override]` when adding a model/stat | Subclass narrowed `Analysis` without generics | Inherit `Model[TennisAnalysis]` / `Stat[TennisAnalysis]` (see §3). |
| `RuntimeError: operator torchvision::nms does not exist` | `torch` CPU wheel + mismatched `torchvision` from PyPI | Install both from the same index (see §9). |
| Ball NRT fails with small metric drift (~0.005–0.01) | CPU nondeterminism | Inspect `report.md`; ball suite tolerance is `0.01`. Regenerate baseline on CPU only for intentional changes. |

---

## Quick reference

```bash
# Install (CPU torch + torchvision; then test deps)
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-test.txt

# Run inference
python -m pipeline.run --sport tennis --video_path clip.mp4 --config configs/tennis_config.yaml

# Annotate a clip
python tools/ball_anotator.py --video_path clip.mp4

# Fast unit tests (no video / model)
python -m pytest tests/unit -m unit -q

# Integration tests (synthetic video, fake model)
python -m pytest tests/integration -m integration -q

# Coverage (unit + integration, fail under 60%)
python -m pytest tests/unit tests/integration -m "unit or integration" `
  --cov=pipeline --cov=tests.evaluation --cov=tests.baseline `
  --cov-report=term-missing --cov-fail-under=60

# Lint (same as CI job lint)
ruff check pipeline tests scripts utils
ruff format --check pipeline tests scripts utils
mypy --config-file=pyproject.toml pipeline
# or: pre-commit run --all-files

# Non-regression test (CPU, slow)
$env:SPORTAI_DEVICE="cpu"
python -m pytest tests/nrt/tennis -m nrt -s -rs

# Refresh baseline after an intentional model change
python -m pytest tests/nrt/tennis -m nrt -s --update-baseline
```
