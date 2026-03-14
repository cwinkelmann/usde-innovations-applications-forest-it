# PCV Course Inventory -- Complete Materials Reference

## Overview

**Course:** Practical Computer Vision with PyTorch
**Location:** `/Users/christian/PycharmProjects/hnee/practical-computer-vision/`
**Modules:** 8
**Notebooks:** 18
**Slide Decks:** 10 (6 PCV + 4 Image Dataset Curation)
**Documentation:** Module overview PDF + project specification

---

## Module Inventory

### Module 1: Foundations (Lessons 1-3)

**Topics covered:**
- Computer vision task taxonomy (classification, detection, segmentation)
- Digital image representation: pixels, channels, color spaces
- PIL/NumPy image manipulation
- PyTorch tensors for image data

**Notebooks:**
| Notebook | Description | Hands-On Depth | Reusability for Wildlife |
|----------|-------------|----------------|--------------------------|
| `Digital_Image_Representation_PIL_NumPy_PyTorch.ipynb` | Load, display, manipulate images with PIL, NumPy, PyTorch. Channel ordering (HWC vs CHW). Normalization. | Guided walkthrough + student exercises | HIGH -- directly applicable; swap stock images for wildlife images |

**Slides:**
- PCV Workshop 1 slides (Lessons 1-3)

**Key concepts for gap analysis:**
- Pixel representation -> directly applicable to GSD concept (pixel = physical area)
- Tensor shapes -> same for aerial images
- Color spaces -> relevant for thermal + RGB fusion

---

### Module 2: Neural Networks (Lessons 4-6)

**Topics covered:**
- Introduction to neural networks
- Multi-layer perceptron (MLP) for regression
- Matrix multiplication and forward pass mechanics

**Notebooks:**
| Notebook | Description | Hands-On Depth | Reusability for Wildlife |
|----------|-------------|----------------|--------------------------|
| `Training_a_Perceptron_for_Image_based_Regression.ipynb` | Train a simple perceptron on image data for regression. Loss functions, gradient descent basics. | Guided walkthrough + exercises | MEDIUM -- standard NN content, wildlife swap is cosmetic |

**Slides:**
- PCV Workshop 2 slides (Lessons 4-6)

**Key concepts for gap analysis:**
- Standard neural network foundations -- no wildlife-specific gap
- Gradient descent fundamentals needed for all downstream modules

---

### Module 3: Training & Evaluation (Lessons 7-9)

**Topics covered:**
- Classification vs regression tasks
- Evaluation metrics: accuracy, precision, recall, F1, confusion matrix
- DataLoaders: train/val/test splits, batching, shuffling

**Notebooks:**
| Notebook | Description | Hands-On Depth | Reusability for Wildlife |
|----------|-------------|----------------|--------------------------|
| `Starter_Create_Dataloaders_Train_Val_Test.ipynb` | Create custom Dataset class, apply transforms, build DataLoaders with proper splitting. | Student exercises with TODOs | HIGH -- swap dataset for wildlife; add stratified splitting for rare species |
| `Kaggle_Competition_LeNet5_Digit_Recognition.ipynb` | LeNet-5 on MNIST for Kaggle digit recognition. End-to-end training pipeline. | Full student project | HIGH -- replace MNIST with iNaturalist mini (10 species) |

**Slides:**
- PCV Workshop 3 slides (Lessons 7-9)

**Key concepts for gap analysis:**
- Precision/recall -> critical for rare species detection (needs wildlife context)
- DataLoaders -> need stratified splitting for imbalanced ecological data
- Confusion matrix -> directly applicable to species confusion analysis

---

### Module 4: CNNs (Lessons 10-12)

**Topics covered:**
- Convolution operation (filters, stride, padding)
- Pooling layers (max, average)
- Upsampling and transposed convolutions
- LeNet-5 architecture deep-dive

**Notebooks:**
| Notebook | Description | Hands-On Depth | Reusability for Wildlife |
|----------|-------------|----------------|--------------------------|
| `Looking_into_LeNet5_with_Random_Weights.ipynb` | Visualize LeNet-5 feature maps at each layer with random weights. Understand what convolutions "see." | Guided visualization | MEDIUM -- useful for understanding feature extraction from aerial images |

**Slides:**
- PCV Workshop 3 slides (continued, Lessons 10-12)

**Key concepts for gap analysis:**
- Convolution filters -> relevant to spatial feature extraction from overhead imagery
- Receptive field -> determines minimum object size detectable (connects to GSD)
- No aerial-specific convolution patterns covered

---

### Module 5: Training Techniques (Lessons 13-15)

**Topics covered:**
- Batch normalization, layer normalization
- Binary cross-entropy for multi-label classification
- Skip connections and residual learning
- Transfer learning and fine-tuning pretrained models

**Notebooks:**
| Notebook | Description | Hands-On Depth | Reusability for Wildlife |
|----------|-------------|----------------|--------------------------|
| `Pet_Classification.ipynb` | Fine-tune a CNN on Oxford Pets dataset. Multi-class classification. | Full student project | VERY HIGH -- direct swap to camera trap species classification |
| `Finetuning_a_Resnet_for_Multilabel_Classification.ipynb` | Fine-tune ResNet for multi-label classification task. | Student exercises | HIGH -- applicable to multi-species camera trap frames |

**Slides:**
- PCV Workshop 4 slides (Lessons 13-15)

**Key concepts for gap analysis:**
- Transfer learning -> directly applicable to wildlife; needs domain-shift discussion
- Fine-tuning -> same mechanics, different data
- Pet classification -> immediate swap target for wildlife classification

---

### Module 6: Optimization & Interpretability (Lessons 16-19)

**Topics covered:**
- Data augmentation strategies
- Regularization (dropout, weight decay)
- Transfer learning (feature extraction vs fine-tuning)
- Class Activation Maps (CAM) for interpretability

**Notebooks:**
| Notebook | Description | Hands-On Depth | Reusability for Wildlife |
|----------|-------------|----------------|--------------------------|
| `Labeling_Images_with_a_Pretrained_Resnet.ipynb` | Use pretrained ResNet to label images. Explore ImageNet class predictions on novel images. | Guided walkthrough | HIGH -- apply to wildlife images; discuss domain gap from ImageNet |

**Slides:**
- PCV Workshop 5 slides (Lessons 16-19)

**Key concepts for gap analysis:**
- Augmentation -> needs wildlife-specific additions (rotation for nadir, flipping for aerial)
- CAM -> useful for understanding what model sees on camouflaged wildlife
- Transfer learning theory -> solid foundation; needs aerial domain gap discussion

---

### Module 7: Embeddings (Lessons 20-22)

**Topics covered:**
- Feature embeddings from CNNs
- Vision Transformers (ViT) overview
- CLIP: Contrastive Language-Image Pre-training
- Zero-shot classification with CLIP

**Notebooks:**
| Notebook | Description | Hands-On Depth | Reusability for Wildlife |
|----------|-------------|----------------|--------------------------|
| `Creating_Embeddings_from_Resnet34.ipynb` | Extract embeddings from ResNet-34, visualize with t-SNE, measure similarity. | Student exercises | VERY HIGH -- species embedding similarity, re-identification |
| `Intro_to_CLIP_ZeroShot_Classification.ipynb` | CLIP zero-shot classification on custom categories. | Guided + exercises | VERY HIGH -- zero-shot wildlife species classification |

**Note:** There may be additional notebooks in Module 7 (ViT-related). Verify against actual repo contents.

**Slides:**
- PCV Workshop 5/6 slides (Lessons 20-22)

**Key concepts for gap analysis:**
- Embeddings -> directly applicable to species similarity and re-identification
- CLIP -> strong bridge to wildlife zero-shot classification
- ViT -> relevant as DINOv2 (used in some wildlife models) is ViT-based

---

### Module 8: Detection & Segmentation (Lessons 23-24)

**Topics covered:**
- Object detection overview (R-CNN family, YOLO family, anchor boxes)
- Semantic segmentation overview
- Instance segmentation overview
- **CONCEPTUAL ONLY -- NO HANDS-ON NOTEBOOKS**

**Notebooks:**
| Notebook | Description | Hands-On Depth | Reusability for Wildlife |
|----------|-------------|----------------|--------------------------|
| (none) | No implementation notebooks for Module 8 | N/A | N/A -- CRITICAL GAP |

**Slides:**
- PCV Workshop 6 slides (Lessons 23-24)

**Key concepts for gap analysis:**
- **CRITICAL GAP:** Detection is the core task for wildlife survey automation. Students learn concepts but cannot implement.
- Anchor boxes explained conceptually but no code
- No mAP/AP50/AP75 computation
- No NMS implementation
- No YOLO training or inference
- No point-based detection (HerdNet, P2PNet)
- No counting/density estimation

---

## Slide Decks

### PCV Series (6 decks)

| Deck | Lessons | Key Content |
|------|---------|-------------|
| PCV Workshop 1 | 1-3 | CV tasks, image representation, PyTorch tensors |
| PCV Workshop 2 | 4-6 | Neural networks, perceptron, matrix math |
| PCV Workshop 3 | 7-12 | Classification, metrics, DataLoaders, CNNs, pooling |
| PCV Workshop 4 | 13-15 | Normalization, BCE, skip connections, transfer learning |
| PCV Workshop 5 | 16-19 | Augmentation, regularization, transfer learning, CAM |
| PCV Workshop 6 | 20-24 | Embeddings, ViT, CLIP, detection overview, segmentation overview |

### Image Dataset Curation Series (4 decks)

| Deck | Key Content |
|------|-------------|
| IDC Deck 1 | Image dataset fundamentals, labeling strategies |
| IDC Deck 2 | Annotation tools, quality control |
| IDC Deck 3 | Dataset bias, class imbalance |
| IDC Deck 4 | Dataset versioning, documentation |

**Relevance to wildlife:** The IDC series covers annotation best practices that are directly applicable to wildlife dataset curation (bounding box vs point annotation, inter-annotator agreement).

---

## Documentation

### Module Overview PDF

**Location:** `docs/Modules - Practical Computer Vision with PyTorch.pdf`

**Content summary:**
- Course learning objectives
- Module-by-module topic list (Lessons 1-24)
- Recommended textbooks and resources
- Assessment criteria

### Project Specification

**Location:** `docs/project_task.md`

**Content summary:**
- Final project requirements
- Deliverable format (notebook + report)
- Grading rubric
- Suggested project topics (none are wildlife-specific)

---

## Reusability Assessment Summary

| Category | Count | Notebooks |
|----------|-------|-----------|
| VERY HIGH (direct wildlife swap) | 4 | Pet_Classification, Finetuning_Resnet, Embeddings_Resnet34, CLIP_ZeroShot |
| HIGH (swap with minor adaptation) | 4 | Image_Representation, DataLoaders, LeNet5_Kaggle, Labeling_Pretrained |
| MEDIUM (standard content, cosmetic swap) | 2 | Perceptron_Regression, LeNet5_Random_Weights |
| N/A (critical gap, no notebook exists) | 1 | Module 8 Detection & Segmentation |

**Total adaptable notebooks:** 10 of 18 (assess remaining 8 against actual repo contents)
**Critical gaps requiring new content:** Module 8 hands-on detection, aerial imagery fundamentals, tile inference, counting/density estimation
