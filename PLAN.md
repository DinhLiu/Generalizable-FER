# Kaggle Experiment Plan: CAFE on RAF-DB + Cross-Dataset Evaluation

## Summary

- Run a report-feasible reproduction of CAFE: train on RAF-DB, save checkpoint/logs, then evaluate zero-shot on CK+48 plus the paper datasets FERPlus, AffectNet, SFEW2.0, and MMAFEDB.
- Target environment: Kaggle Notebook with GPU enabled.
- Kaggle path convention:
  - Working code/output: `/kaggle/working/Generalizable-FER`
  - Writable compatibility paths for the original code: `/kaggle/working/data`, `/kaggle/working/resnet18_msceleb.pth`, `/kaggle/working/outputs`
  - Read-only Kaggle datasets/checkpoints: `/kaggle/input/...`

## Kaggle Setup

1. Create a Kaggle Notebook and enable GPU:
   - Settings -> Accelerator -> GPU.
   - Internet can be enabled if you want CLIP to download automatically; otherwise upload the CLIP cache/checkpoint separately.

2. Put the repo in `/kaggle/working/Generalizable-FER`:
   - Option A: upload this repo as a Kaggle Dataset, then copy it:
     ```bash
     cp -r /kaggle/input/generalizable-fer/Generalizable-FER /kaggle/working/Generalizable-FER
     ```
   - Option B: if Internet is enabled, clone/pull the repo into `/kaggle/working/Generalizable-FER`.

3. Attach/upload datasets and checkpoints as Kaggle Datasets, then create writable compatibility links:
   ```bash
   mkdir -p /kaggle/working/data /kaggle/working/outputs
   ln -s /kaggle/input/raf-db/raf-basic /kaggle/working/data/raf-basic
   ln -s /kaggle/input/ckplus48 /kaggle/working/data/ckplus48
   ln -s /kaggle/input/ferplus /kaggle/working/data/ferplus
   ln -s /kaggle/input/affectnet /kaggle/working/data/affectnet
   ln -s /kaggle/input/datasetsfew /kaggle/working/data/sfew
   ln -s /kaggle/input/mma-facial-expression /kaggle/working/data/mma
   ln -s /kaggle/input/cafe-checkpoints/resnet18_msceleb.pth /kaggle/working/resnet18_msceleb.pth
   ```

4. Verify the expected RAF-DB layout:
   ```text
   /kaggle/working/data/raf-basic/EmoLabel/list_patition_label.txt
   /kaggle/working/data/raf-basic/Image/aligned/*_aligned.jpg
   ```

5. Install dependencies conservatively:
   ```bash
   cd /kaggle/working/Generalizable-FER
   python - <<'PY'
   import importlib.util
   for name in ["torch", "torchvision", "cv2", "pandas", "sklearn", "matplotlib", "seaborn", "ftfy", "regex", "tqdm"]:
       print(name, importlib.util.find_spec(name) is not None)
   PY
   pip install -r requirements.txt
   ```
   If Kaggle already has PyTorch, do not separately reinstall a different CUDA wheel unless import/CUDA checks fail.

6. Check GPU:
   ```bash
   python - <<'PY'
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
   PY
   ```

## Run CAFE on RAF-DB

Run from the `code` directory because the original script uses relative paths:

```bash
cd /kaggle/working/Generalizable-FER/code
python ours_CAFE.py --raf_path ../../data/raf-basic --label_path list_patition_label.txt --batch_size 32 --workers 2 --gpu 0 --epochs 1
```

If the 1-epoch smoke test succeeds, run the main training:

```bash
python ours_CAFE.py --raf_path ../../data/raf-basic --label_path list_patition_label.txt --batch_size 32 --workers 2 --gpu 0 --epochs 60
```

If CUDA runs out of memory, rerun with `--batch_size 16` and record this difference in the report.

Save artifacts:

```bash
mkdir -p /kaggle/working/outputs/rafdb_cafe_seed3407
cp ours_best.pth ours_final.pth results.txt /kaggle/working/outputs/rafdb_cafe_seed3407/
```

Paper reference for train-on-RAF-DB CAFE:

| Test set | Paper accuracy |
|---|---:|
| RAF-DB | 88.72 |
| FERPlus | 73.16 |
| AffectNet | 45.86 |
| SFEW2.0 | 52.86 |
| MMA | 56.80 |
| Mean | 63.48 |

## Evaluate CK+48

- Create a separate `evaluate_ckplus48.py`; do not change the training logic in `ours_CAFE.py`.
- The script must scan `/kaggle/working/data/ckplus48` after extraction to determine the real class folders.
- Class handling:
  - Drop `contempt`, because RAF-DB/CAFE has no contempt class.
  - If CK+48 has `neutral`, map it to CAFE label `6` and report `CK+48-7cls-with-neutral`.
  - If CK+48 has no neutral, report `CK+48-6cls`; predictions of label `6` count as wrong naturally because no ground-truth neutral exists.
- Label mapping:
  ```text
  surprise -> 0
  fear     -> 1
  disgust  -> 2
  happy    -> 3
  sadness/sad -> 4
  anger/angry -> 5
  neutral  -> 6, only if present
  ```
- CK+48 transform is mandatory:
  - Convert grayscale or RGB input to RGB 3-channel.
  - Resize/upscale from 48x48 to 224x224.
  - Normalize with ImageNet statistics:
    ```python
    transforms.Resize((224, 224))
    transforms.ToTensor()
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ```
- Metrics:
  - Overall accuracy.
  - Mean Accuracy: average of per-class accuracy over classes with ground-truth samples.
  - Macro-F1.
  - Per-class accuracy.
  - Confusion matrix.
- Outputs:
  ```text
  /kaggle/working/outputs/ckplus48_eval/predictions.csv
  /kaggle/working/outputs/ckplus48_eval/metrics.json
  /kaggle/working/outputs/ckplus48_eval/confusion_matrix.png
  ```

## Paper Dataset Structures from Kaggle Inputs

The added Kaggle inputs use folder-based class labels. All evaluations use CAFE/RAF's 7-class order: `surprise=0`, `fear=1`, `disgust=2`, `happy=3`, `sadness=4`, `anger=5`, `neutral=6`. Any `contempt` folder is skipped because the trained CAFE classifier has no contempt output.

| Dataset | Kaggle structure | Evaluation split | Notes |
|---|---|---|---|
| FERPlus | `/kaggle/input/ferplus/{test,train,validation}/{angry,contempt,disgust,fear,happy,neutral,sad,suprise}` | `test` | Folder name `suprise` is mapped to `surprise`; `contempt` is skipped. |
| AffectNet | `/kaggle/input/affectnet/archive (3)/{Test,Train}/{Anger,Contempt,Disgust,Fear,Happy,Neutral,Sad,Surprise}` plus `labels.csv` | `Test` | The image folders already provide labels; `Contempt` is skipped. |
| SFEW2.0 | `/kaggle/input/datasetsfew/{Train,Val}/{Angry,Disgust,Fear,Happy,Neutral,Sad,Surprise}` and `/Test/Test_Aligned_Faces` | `Val` | `Test/Test_Aligned_Faces` is unlabeled in the visible structure, so use `Val` for measured accuracy. |
| MMAFEDB | `/kaggle/input/mma-facial-expression/MMAFEDB/{test,train,valid}/{angry,disgust,fear,happy,neutral,sad,surprise}` | `test` | Direct 7-class folder layout. |

## Evaluate Paper Datasets

Use `code/evaluate_cross_dataset.py` from the `code` directory after RAF-DB training has produced `/kaggle/working/outputs/rafdb_cafe_seed3407/ours_best.pth`:

```bash
cd /kaggle/working/Generalizable-FER/code
python evaluate_cross_dataset.py --dataset ferplus --data_path /kaggle/working/data/ferplus --checkpoint ../../outputs/rafdb_cafe_seed3407/ours_best.pth --output_dir ../../outputs/ferplus_eval --batch_size 64 --workers 2 --gpu 0
python evaluate_cross_dataset.py --dataset affectnet --data_path /kaggle/working/data/affectnet --checkpoint ../../outputs/rafdb_cafe_seed3407/ours_best.pth --output_dir ../../outputs/affectnet_eval --batch_size 64 --workers 2 --gpu 0
python evaluate_cross_dataset.py --dataset sfew --data_path /kaggle/working/data/sfew --split Val --checkpoint ../../outputs/rafdb_cafe_seed3407/ours_best.pth --output_dir ../../outputs/sfew_val_eval --batch_size 64 --workers 2 --gpu 0
python evaluate_cross_dataset.py --dataset mma --data_path /kaggle/working/data/mma --checkpoint ../../outputs/rafdb_cafe_seed3407/ours_best.pth --output_dir ../../outputs/mma_eval --batch_size 64 --workers 2 --gpu 0
```

Each run writes:

```text
/kaggle/working/outputs/<dataset>_eval/predictions.csv
/kaggle/working/outputs/<dataset>_eval/metrics.json
/kaggle/working/outputs/<dataset>_eval/confusion_matrix.png
```

## Acceptance Checks for the Report

- Import and CUDA checks pass in Kaggle.
- RAF-DB label file and aligned images are found through `/kaggle/working/data/raf-basic`.
- One forward pass returns logits of shape `[batch, 7]`.
- Training creates `ours_best.pth`, `ours_final.pth`, and `results.txt`.
- CK+48 and paper-dataset evaluation detect the class mode, resize images to 224x224, and write CSV/JSON/PNG outputs.
- Report chapter 5 states:
  - Kaggle GPU environment, Python/PyTorch/CUDA versions, batch size, epochs.
  - RAF-DB as source dataset; CK+48 as outside-paper dataset; FERPlus, AffectNet, SFEW2.0, and MMAFEDB as paper cross-dataset evaluations.
  - RAF-DB paper-vs-run result and each evaluation dataset's overall accuracy, Mean Accuracy, macro-F1.
  - Differences caused by Kaggle GPU, batch size if changed, seed, checkpoint availability, CK+48 48x48 upscaling, and neutral/contempt handling.
