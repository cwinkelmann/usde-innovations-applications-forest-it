# Wildlife Adapted Notebook Template

## Usage

This template defines the structure for adapting any existing PCV notebook to use wildlife datasets. The `wildlife_adapter_agent` uses this template to produce adapted notebook documents. The template ensures that all adaptations are systematic, documented, and preserve the original pedagogical flow.

---

## Adaptation Document Structure

### Header Section

```markdown
# [Original Notebook Title] -- Wildlife Adaptation

## Adaptation Summary

| Attribute | Value |
|-----------|-------|
| Original notebook | [filename.ipynb] |
| Original dataset | [dataset name] |
| Wildlife dataset | [replacement dataset name] |
| Original cell count | [N] |
| Modified cells | [N of M] |
| Added cells | [N] |
| Removed cells | [0 -- never remove, only replace] |

## Learning Objectives (Preserved)

1. [Same as original objective 1]
2. [Same as original objective 2]
3. [Same as original objective 3]

## Learning Objectives (Added)

1. [Wildlife-specific objective, e.g., "Interpret precision and recall for rare species detection"]

## Prerequisites (Updated)

- [Original prerequisites]
- **New:** [Any additional prerequisites for wildlife context]
```

### Adaptation Specification Table

```markdown
## Cell-by-Cell Adaptation

| Cell # | Type | Original Content | Change Type | Adaptation Notes |
|--------|------|-----------------|-------------|-----------------|
| 1 | markdown | Title: "MNIST Classification" | text swap | -> "Wildlife Species Classification with iNaturalist" |
| 2 | markdown | Learning objectives | extend | Add wildlife-specific objective |
| 3 | code | `import torchvision; MNIST(...)` | code swap | Replace with ImageFolder + wildlife download |
| 4 | markdown | "MNIST contains 70,000 images..." | text swap | Describe wildlife dataset characteristics |
| 5 | code | `transforms.Compose([...])` | extend | Add rotation (nadir invariance), wildlife-specific augmentation |
| 6 | code | DataLoader creation | unchanged | Same code, different data |
| 7 | code | Model definition | unchanged | Same architecture |
| 8 | code | Training loop | unchanged | Same training loop |
| 9 | code | Evaluation | extend | Add per-species accuracy analysis |
| 10 | markdown | "Results discussion" | text swap | Wildlife-specific interpretation |
| 11 | markdown | (NEW) | addition | "Wildlife Context: Class imbalance in ecological data" |
```

---

## Cell Templates by Change Type

### Change Type: `text swap`

Replace dataset-specific text while keeping the same structure and teaching style.

**Original:**
```markdown
## Dataset: MNIST

MNIST contains 70,000 handwritten digit images (60,000 train, 10,000 test).
Each image is 28x28 pixels, grayscale, with labels 0-9.
```

**Adapted:**
```markdown
## Dataset: iNaturalist Mini (10 Wildlife Species)

This subset of iNaturalist contains 5,000 wildlife images (4,000 train, 1,000 test).
Each image is a color photograph of one of 10 common species, resized to 224x224 pixels.

> **Wildlife context:** Unlike MNIST where classes are balanced, ecological datasets
> often have severe class imbalance. Common species may have 10x more images than rare species.
```

### Change Type: `code swap`

Replace data loading code while keeping the same variable names and downstream interface.

**Original:**
```python
# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)
```

**Adapted:**
```python
# Download and load wildlife dataset
from torchvision.datasets import ImageFolder
import urllib.request
import zipfile

# Download wildlife mini dataset (if not already present)
DATA_URL = "[URL to wildlife dataset zip]"
DATA_DIR = Path("./data/wildlife_mini")
if not DATA_DIR.exists():
    print("Downloading wildlife dataset...")
    urllib.request.urlretrieve(DATA_URL, "wildlife_mini.zip")
    with zipfile.ZipFile("wildlife_mini.zip", "r") as z:
        z.extractall("./data")
    print("Download complete!")

# Define transforms (note: wildlife images are RGB, not grayscale)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root=DATA_DIR / "train", transform=transform)
test_dataset = ImageFolder(root=DATA_DIR / "test", transform=transform)

print(f"Training images: {len(train_dataset)}")
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")
```

**Key rules for code swap:**
1. Keep the same variable names (`train_dataset`, `test_dataset`) so downstream cells work unchanged
2. Use `ImageFolder` for classification datasets (standard torchvision pattern)
3. Use ImageNet normalization stats for RGB images
4. Print dataset statistics for student verification
5. Include download code with existence check (idempotent)

### Change Type: `extend`

Add lines to an existing cell without removing any original code.

**Original:**
```python
# Evaluate model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test accuracy: {accuracy:.4f}")
```

**Adapted (extended):**
```python
# Evaluate model
correct = 0
total = 0
all_preds = []    # ADDED: collect for per-species analysis
all_labels = []   # ADDED: collect for per-species analysis
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())    # ADDED
        all_labels.extend(labels.cpu().numpy())       # ADDED

accuracy = correct / total
print(f"Test accuracy: {accuracy:.4f}")

# ADDED: Per-species accuracy analysis
from sklearn.metrics import classification_report
print("\nPer-species classification report:")
print(classification_report(all_labels, all_preds,
                            target_names=test_dataset.classes))
```

**Key rules for extend:**
1. Mark all added lines with `# ADDED` comment
2. Do not modify existing lines
3. Keep additions at the end of the cell where possible
4. If additions must go in the middle (e.g., collecting predictions), add the collection lines and the analysis at the end

### Change Type: `addition` (New Cell)

Insert a new cell that does not exist in the original notebook.

**Template for wildlife context cells:**

```markdown
### Wildlife Context: [Topic]

> This section is new -- it does not appear in the original PCV notebook.

[1-2 paragraphs explaining why this concept matters specifically for wildlife/ecology]

**Example from the case study:** [Reference to Miesner thesis if applicable]

**Discussion question:** [Prompt for student reflection]
```

**Placement rules for additions:**
- Wildlife context cells go AFTER the related code cell, not before
- Never add more than 2 new cells per original section (keep additions proportional)
- Never exceed 30% additional cells (e.g., for a 15-cell notebook, add at most 4-5 cells)

### Change Type: `unchanged`

The cell is identical to the original. Copy it verbatim. This typically applies to:
- Model architecture definitions (same CNN regardless of data)
- Training loops (same optimizer, same loss function)
- Generic utility functions (plotting, timing)

---

## Adaptation Quality Checklist

Before finalizing any adapted notebook, verify:

```markdown
## Quality Checklist

- [ ] All original learning objectives are preserved
- [ ] Dataset download is automated and idempotent
- [ ] Variable names match original (no downstream breaks)
- [ ] Image dimensions are correct for the new dataset (28x28 -> 224x224 for RGB)
- [ ] Normalization stats are updated (MNIST stats -> ImageNet stats)
- [ ] Number of classes is updated in model definition
- [ ] All assertions/validation cells still pass with wildlife data
- [ ] Wildlife context cells are clearly marked as additions
- [ ] Total added cells < 30% of original cell count
- [ ] No PCV theory content is removed or rewritten (only data-specific text)
- [ ] Import dependencies are listed and pip-installable
- [ ] Cross-notebook references are updated if applicable
```

---

## Common Pitfalls to Avoid

1. **Changing the model architecture to "improve" it** -- The adapted notebook teaches the same architecture. If you want to use a better model for wildlife, create a new exercise instead.
2. **Adding too much wildlife context** -- The lesson is about the PCV concept, not about wildlife ecology. Keep additions proportional.
3. **Breaking normalization** -- MNIST uses mean=0.1307, std=0.3081. Wildlife RGB images use ImageNet stats. Forgetting to update normalization is a common silent error.
4. **Forgetting to update num_classes** -- If the model's final layer has `nn.Linear(features, 10)` for MNIST, it must be updated to match the wildlife dataset's class count.
5. **Hardcoded image sizes** -- MNIST is 28x28 grayscale. Wildlife images are typically 224x224 RGB. Check all reshape/view operations.
