# Data

The dataset (~7 GB of `.tif` image patches) is not committed to this repository. Download it from the [Kaggle Histopathologic Cancer Detection competition](https://www.kaggle.com/competitions/histopathologic-cancer-detection) after accepting the competition rules:

```bash
kaggle competitions download -c histopathologic-cancer-detection -p data/
unzip -q data/histopathologic-cancer-detection.zip -d data/histopathologic-cancer-detection
```

Expected layout after extraction:

```
data/histopathologic-cancer-detection/
├── train/              # 220,025 labeled 96×96 .tif patches
├── test/               # 57,458 unlabeled patches
├── train_labels.csv    # id → label (0 = normal, 1 = tumor in center 32×32)
└── sample_submission.csv
```

On Kaggle, the same content is mounted read-only at `/kaggle/input/histopathologic-cancer-detection`, which is the path the notebook uses by default.
