# FaceMap 3DMM Landmark & Expression Analyzer

Pipeline to run Qualcomm’s FaceMap 3DMM model locally, render 68‑point landmarks on images/videos, log per‑frame outputs, and analyze expressions to produce a 0–100 score.

What you get
- Ready‑to‑run CLI (`infer.py`) for images/videos, with face ROI cropping, stability guards, overlays, and structured logs
- Expression analyzer (`analyze_expressions.py`) that turns landmarks + pose into per‑frame labels and an overall score
- Clean runs via `--clean`, plus CSVs for raw 3DMM params, landmarks, and meta status

Works with these files in this folder:
- `model.onnx` – FaceMap 3DMM graph
- `model.data` – external weights (must be adjacent to `model.onnx`)

## Prerequisites
- macOS with Python 3.9+ (3.10/3.11 ok)
- A test image file (e.g., `face.jpg`)

## Setup (first time)
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you are on Apple Silicon and hit a wheel issue, try:
```
pip install --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime/pypi/simple/ onnxruntime
```
…otherwise stick with the regular `onnxruntime` listed in `requirements.txt`.

## Quick usage
- Inspect model I/O:
```
python infer.py --inspect
```
- Run inference on one image and save overlay (landmarks drawn if detected):
```
python infer.py --image /path/to/face.jpg --output out
```
This writes an annotated image into `out/face_overlay.jpg` and prints the output tensor shapes/summary.

## Video inference
- Process a `.mov`/`.mp4` and save an annotated video + logs:
```
python infer.py --video "Movie on 10-13-25 at 2.03 PM.mov" --output out --save-video --clean
```
- Keep original resolution is enabled by default for video output. Results:
  - Annotated video: `out/<video>_overlay.mp4`
  - Per-frame summary CSV: `out/logs/<video>_summary.csv` (min/max per frame)
  - Optional raw vector CSV: add `--dump-csv` to write `out/logs/<video>_raw.csv` with the full 265-D vector per frame.

### Landmark rendering via 3DMM decode (optional)
- Enable landmark decoding with Qualcomm’s utilities (downloads small assets on first run):
```
pip install "qai-hub-models[facemap-3dmm]"
python infer.py --video "<your video>.mov" --output out --save-video --decode-3dmm --point-size 3 --line-width 2 --mesh
```
- This draws 68 landmarks per frame decoded from the 1x265 3DMM parameters. Use `--point-size` and `--line-width` to make them more visible. Add `--mesh` for a Delaunay wireframe; add `--no-contours` to hide 68-pt contour lines.

### Robust tracking and meta logs
- To reduce elongation and false overlays on off‑angle frames:
  - Add face ROI detection/cropping: `--face-detect --face-margin 0.25`
  - Skip frames when no face is found: `--skip-if-no-face`
  - Skip unstable jumps in raw params: `--jump-threshold 25.0`
  - Always start clean for a new video: add `--clean` (deletes the `--output` folder before the run)
- Extra logs are written to `out/logs/<video>_meta.csv` with columns: `frame,status,reason,x0,y0,x1,y1`.

### Dumping results for downstream analysis
- Raw param vectors per frame: add `--dump-csv` to get `*_raw.csv`.
- Per-frame 68 landmark pixels: add `--dump-landmarks-csv` to get `*_landmarks.csv`.

## Expression analysis and scoring
- After running inference with `--decode-3dmm --dump-landmarks-csv [--dump-csv --face-detect --skip-if-no-face]`, run:
```
python analyze_expressions.py \
  --landmarks out/logs/<video>_landmarks.csv \
  --meta out/logs/<video>_meta.csv \
  --raw out/logs/<video>_raw.csv \
  --out out/analysis
```
- Outputs:
  - `out/analysis/score.txt` — overall score out of 100
  - `out/analysis/report.json` — summary and class distribution
  - `out/analysis/expressions.csv` — per‑frame label and score



## Notes on mesh & model
- The model outputs 3DMM parameters (shape 1×265), not a dense mesh. We decode to 68 landmarks using Qualcomm’s utils.
- The cyan mesh is a Delaunay wireframe for visualization only; it does not affect inference.

## CLI options
```
python infer.py --help
```
Key flags:
- `--model` Path to ONNX (`model.onnx` by default)
- `--image` Path to one image
- `--output` Output folder for overlays and dumps
- `--normalize` Input normalization (choices: none, 0_1, -1_1, imagenet)
- `--bgr` Use BGR instead of RGB preprocessing
- `--input-name` Force input tensor name (optional)
- `--output-name` Force output tensor(s), comma‑separated (optional)
- `--assume-points N` Assume output is N×2 or N×3 points (try N=68/98/468)
- `--assume-norm` Assume landmark coords are normalized [0..1]
- `--dump-npy` Save raw outputs as .npy
- `--inspect` Print model I/O and exit

## Troubleshooting
- Ensure `model.onnx` and `model.data` stay together in the same directory.
- If ONNX Runtime fails to load, ensure Python and pip are from your venv (`which python`, `which pip`).
- If overlays look wrong, try different `--normalize` or `--assume-*` flags, or pass `--bgr`.
