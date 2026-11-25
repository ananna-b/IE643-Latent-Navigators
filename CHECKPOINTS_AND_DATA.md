# Checkpoints and dataset access

## Files NOT included in this GitHub package

- PASCAL VOC 2012 image datasets and XML annotations (retrieve from the benchmark source).
- Full RTTS / DAWN / ExDark imagery and annotations (obtain from original sources, observe their terms).
- The exact trained checkpoint of the selected novelty experiment. The recorded `selected_candidate.json` references:

  `/content/drive/MyDrive/IE643_LatentNavigators/novelty_2026/yolo_runs/minority_lowlight/weights/best.pt`

- The original YOLO baseline / earlier combined checkpoints needed to rerun novelty fine-tuning. The novelty notebook searches for them in `MyDrive/IE643_LatentNavigators/checkpoints/`, e.g. `det_baseline.pt` and `det_combined.pt`.

## How to add a demo checkpoint later

1. Copy the *correct* file from the original Google Drive training run (verify its run folder and training configuration).
2. Record source run, model name, train data, class map, Ultralytics version if known, and validation split.
3. If publishing the file, check rights and use Git LFS when appropriate. Do not simply rename an unknown `best.pt` file to the selected novelty checkpoint.
4. Make the notebook/demo path configurable; the current experiment notebooks contain absolute Colab/Drive paths.

The repository includes the exact recorded **metric CSVs** and **qualitative exports** as evidence, so viewers can inspect results without having access to private Drive folders.
