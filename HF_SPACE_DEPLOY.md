# Hugging Face Space Deployment (Option A)

This project is prepared for a single **Gradio Space** deployment.

## 1) What is already set

- `app.py` is the Space entrypoint.
- `requirements.txt` now includes `requirements-demo.txt`.
- `app.py` supports both:
  - local checkpoint at `checkpoints/best.pt`, or
  - download from a Hugging Face model repo.

## 2) Repository layout for Space

Keep this structure in the Space repo:

```text
<space-root>/
  app.py
  requirements.txt
  requirements-demo.txt
  scripts/demo_gradio.py
  desert_segmentation/
  checkpoints/
    best.pt                # optional (local model mode)
```

## 3) Create the Space

1. Create a new Space on Hugging Face.
2. Select **SDK: Gradio**.
3. Start with **CPU Basic** (free).
4. Push this repository to that Space.

## 4) Choose one model loading mode

### Mode A: Local checkpoint in Space repo

- Add your model file at `checkpoints/best.pt`.
- No extra environment variables needed.

### Mode B: Download checkpoint from HF model repo (recommended)

Set these Space variables:

- `HF_MODEL_REPO` = `your-username/your-model-repo`
- `HF_MODEL_FILENAME` = `best.pt` (optional if file is named `best.pt`)

If the model repo is private, add this as a **Secret**:

- `HF_TOKEN` = your Hugging Face access token

## 5) Optional Space variables

- `CHECKPOINT_PATH` (local path override)
- `ROOT` (auto-set by `app.py`, usually not needed)
- `PORT` (Space runtime sets this automatically)

## 6) Local smoke test (before pushing)

From repo root:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860`.

## 7) Notes

- First startup can be slow while dependencies install.
- CPU inference is slower; switch to GPU hardware later if needed.
- The app binds to `0.0.0.0` in Space automatically.
