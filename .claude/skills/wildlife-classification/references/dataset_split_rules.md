# Dataset Split Rules for Wildlife Classification

## Core Rule

**GroupShuffleSplit by site is mandatory for spatial data.** Random splitting causes massive data leakage when images come from the same location or time sequence.

---

## Why Random Splits Fail

| Data Source | Leakage Risk | Grouping Variable |
|-------------|-------------|-------------------|
| Camera trap | Same station sees same individuals | Station ID |
| Drone tiles | Adjacent tiles share background/animals | Source orthomosaic filename |
| Burst/sequence | Temporal autocorrelation | Sequence ID or timestamp window |
| Multiple visits | Same site, same individuals | Site ID |

If tiles from the same orthomosaic appear in both train and val, the model memorizes background texture, not animal features. Reported F1 will be inflated by 10-30%.

---

## GroupShuffleSplit Implementation

```python
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# images: list of file paths
# labels: list of class indices
# groups: list of group IDs (site, station, orthomosaic)

# Step 1: Split into train (70%) and temp (30%)
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(gss.split(images, labels, groups=groups))

# Step 2: Split temp into val (15%) and test (15%)
temp_images = np.array(images)[temp_idx]
temp_labels = np.array(labels)[temp_idx]
temp_groups = np.array(groups)[temp_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx_rel, test_idx_rel = next(gss2.split(temp_images, temp_labels, groups=temp_groups))

val_idx = temp_idx[val_idx_rel]
test_idx = temp_idx[test_idx_rel]
```

---

## Extracting Group IDs

### From drone tiles (patcher output)
```python
# Tile filename format: orthomosaic_001_tile_003.jpg
# Group = everything before last "_tile_"
import re

def extract_group(filename):
    match = re.match(r'(.+)_tile_\d+', filename)
    return match.group(1) if match else filename
```

### From camera trap stations
```python
# Filename format: StationA_2024-01-15_08-30-22.jpg
# Group = station name (first part before date)
def extract_station(filename):
    return filename.split('_')[0]
```

### From temporal sequences
```python
# Group images within N-minute windows at same station
from datetime import datetime, timedelta

def group_by_time_window(timestamps, stations, window_minutes=5):
    groups = []
    for ts, station in zip(timestamps, stations):
        window_id = int(ts.timestamp() / (window_minutes * 60))
        groups.append(f"{station}_{window_id}")
    return groups
```

---

## Default Split Ratios

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model training |
| Val | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final evaluation (touch ONCE) |

---

## Cross-Validation for Small Datasets

When total dataset < 500 images, use GroupKFold instead:

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(images, labels, groups)):
    print(f"Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")
    # Train and evaluate per fold
    # Report mean +/- std across folds
```

---

## Iguana Training Split Pattern

The iguana training codebase uses predefined site-based splits:
- `Floreana_classification/classification_train_0_512` and `classification_val_0_512`
- `Fernandina_s_classification/classification_train_0_518` and `classification_val_0_518`
- Each island's data stays entirely within one split (no cross-island leakage)

---

## Data Leakage Validation

Run this check after splitting:

```python
def validate_no_leakage(train_groups, val_groups, test_groups):
    train_set = set(train_groups)
    val_set = set(val_groups)
    test_set = set(test_groups)

    assert train_set.isdisjoint(val_set), \
        f"Leakage! Groups in both train and val: {train_set & val_set}"
    assert train_set.isdisjoint(test_set), \
        f"Leakage! Groups in both train and test: {train_set & test_set}"
    assert val_set.isdisjoint(test_set), \
        f"Leakage! Groups in both val and test: {val_set & test_set}"
    print("No data leakage detected")
```

---

## Special Case: Tiled Drone Imagery

When using manual tiling or patcher tools:
1. All tiles from one orthomosaic must go to the same split
2. Adjacent tiles may share the same animal at borders -- same-split handles this
3. If using overlapping tiles for training augmentation, the group is the source orthomosaic, NOT the tile

---

## Common Mistakes

1. **Random split on tiled data** -- adjacent tiles share background; model memorizes terrain
2. **Splitting by image but not by sequence** -- camera trap burst captures the same animal 5 times in 3 seconds
3. **Using test set for hyperparameter tuning** -- test set must be touched exactly once
4. **Forgetting temporal correlation** -- images from the same day at the same site are not independent
