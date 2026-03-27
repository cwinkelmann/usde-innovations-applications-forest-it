# PCV-to-Wildlife Bridge -- Concept Mapping Reference

## Purpose

This document maps every major concept taught in PCV Modules 1-8 to its corresponding wildlife detection application. It serves two agents:
- **curriculum_mapper_agent:** Uses this mapping to identify which PCV concepts are directly applicable, which need extension, and which have no wildlife counterpart.
- **wildlife_adapter_agent:** Uses this mapping to add domain-relevant commentary when adapting notebooks.

---

## Concept Mappings

### Module 1: Foundations

| PCV Concept | Wildlife Application | Bridge Explanation |
|-------------|---------------------|-------------------|
| Pixel representation (RGB values) | Aerial image pixel = physical area on ground | In PCV, a pixel is abstract. In aerial imaging, each pixel represents a GSD-determined area (e.g., 1 cm x 1 cm at 40 m altitude). This makes pixel resolution a biological measurement. |
| Image resolution (width x height) | Image footprint determines survey coverage | A 5472x3648 image at 0.94 cm/px GSD covers 51.3 m x 34.2 m on the ground. Resolution is no longer just "quality" -- it determines how much habitat you survey per image. |
| Color channels (RGB) | Multi-spectral and thermal channels | PCV teaches RGB. Wildlife sensing adds thermal (single-channel heat map), NIR (near-infrared for vegetation health), and multispectral bands. The tensor concept generalizes directly. |
| Tensor operations (CHW format) | Same format for aerial tiles | PyTorch CHW format applies identically to aerial image tiles. No conceptual change needed, just awareness that C might be 1 (thermal) or 4+ (multispectral). |
| Image normalization (0-255 to 0-1) | Sensor-specific normalization | Camera trap images and drone images come from different sensors with different dynamic ranges. Normalization must be adapted per sensor, not assumed universal. |

### Module 2: Neural Networks

| PCV Concept | Wildlife Application | Bridge Explanation |
|-------------|---------------------|-------------------|
| Perceptron / MLP | Baseline for species prediction from features | A simple MLP on hand-crafted features (body length, color histogram) was the pre-deep-learning approach to species ID. Understanding MLPs contextualizes why CNNs were a breakthrough. |
| Loss functions (MSE, cross-entropy) | Task-specific losses for wildlife | MSE for counting (regression), cross-entropy for species classification, focal loss for class-imbalanced camera trap data. Same math, different ecological motivation. |
| Gradient descent | Same optimization, different convergence challenges | Wildlife datasets are smaller and noisier than benchmark datasets. Gradient descent may converge to poor solutions without proper augmentation and regularization. |

### Module 3: Training & Evaluation

| PCV Concept | Wildlife Application | Bridge Explanation |
|-------------|---------------------|-------------------|
| Train/val/test split | Spatial and temporal splits in ecology | Standard random splitting is invalid for ecological data where images from the same camera location are correlated. Splits must be by location or by time (temporal hold-out). |
| Accuracy metric | Misleading for imbalanced wildlife data | A camera trap dataset with 90% empty frames gets 90% accuracy by predicting "empty." Accuracy is nearly useless. Use precision, recall, F1. |
| Precision and recall | Critical for rare species conservation | For an endangered species, a false negative (missed individual) is costlier than a false positive (false alarm). Recall-oriented evaluation is standard in conservation. |
| F1 score | Primary metric in wildlife detection | The thesis reports F1 = 0.934 (Floreana) and F1 = 0.843 (Fernandina). F1 balances precision and recall, which is what ecologists need. |
| Confusion matrix | Species confusion analysis | Which species does the model confuse? Confusion matrices reveal systematic misclassifications (e.g., juvenile iguanas confused with lava rocks). |
| DataLoaders and batching | Same mechanics, wildlife-specific transforms | DataLoader code is identical. The difference is in transforms (rotation invariance for nadir imagery) and sampling strategy (oversampling rare species). |

### Module 4: CNNs

| PCV Concept | Wildlife Application | Bridge Explanation |
|-------------|---------------------|-------------------|
| Convolution filters | Spatial feature extraction from aerial imagery | Convolution filters detect edges, textures, and shapes. In aerial imagery, early filters detect body outlines; deeper filters detect species-specific patterns. |
| Stride and padding | Control receptive field for small object detection | Small wildlife in aerial images requires careful stride/padding to avoid down-sampling too aggressively. Objects that are 15 pixels wide can be lost by stride-2 pooling. |
| Pooling layers | Trade-off: invariance vs. localization | Max pooling provides translation invariance but loses spatial precision. For counting (where exact location matters), excessive pooling is harmful. |
| Receptive field | Minimum detectable object size | An object must span at least the receptive field of the deepest feature map. If an iguana is 30 pixels and the receptive field is 50 pixels, the network sees the iguana + context. |
| Feature maps | What the network "sees" in wildlife images | Visualizing feature maps on aerial wildlife images shows what the CNN considers discriminative: body shape, shadow, texture contrast against substrate. |

### Module 5: Training Techniques

| PCV Concept | Wildlife Application | Bridge Explanation |
|-------------|---------------------|-------------------|
| Batch normalization | Stabilizes training on variable wildlife data | Camera trap images have extreme lighting variation (day/night, flash/ambient). Batch norm helps the network adapt to this input distribution. |
| Skip connections / ResNets | Backbone for wildlife detection models | ResNet-50 and DLA-34 (used in HerdNet) rely on skip connections. Understanding residual learning is prerequisite to understanding the thesis model. |
| Transfer learning (fine-tuning) | Pretrained models for wildlife species | Fine-tuning ImageNet-pretrained models on wildlife data is the most common approach. PCV teaches the mechanics; the wildlife application is direct. |
| Multi-label classification | Multi-species camera trap frames | Camera trap images often contain multiple species. Multi-label classification (sigmoid per class) is more appropriate than single-label (softmax). |

### Module 6: Optimization & Interpretability

| PCV Concept | Wildlife Application | Bridge Explanation |
|-------------|---------------------|-------------------|
| Data augmentation | Rotation invariance for nadir imagery | Standard augmentations (flip, crop, color jitter) apply. For nadir imagery, 90-degree rotation invariance is especially important since the camera is looking straight down. |
| Regularization (dropout, weight decay) | Preventing overfitting on small wildlife datasets | Wildlife datasets are typically small (hundreds to low thousands of images). Regularization is essential to prevent memorization. |
| Transfer learning (feature extraction) | Frozen backbone + wildlife head | Using a pretrained backbone as a fixed feature extractor with only the classification head trained is effective when wildlife training data is very limited (<500 images per species). |
| Class Activation Maps (CAM) | Understanding wildlife model decisions | CAM on wildlife images reveals whether the model focuses on the animal body or on contextual cues (habitat, camera trap housing). This is critical for model validation. |
| Learning rate scheduling | Critical for fine-tuning on small datasets | Learning rate schedules (cosine, step decay) matter more with small datasets. Too high and the model forgets pretrained features; too low and it never adapts to wildlife domain. |

### Module 7: Embeddings

| PCV Concept | Wildlife Application | Bridge Explanation |
|-------------|---------------------|-------------------|
| Feature embeddings (ResNet) | Species similarity and re-identification | Embedding vectors from wildlife images can measure inter-species similarity (which species look alike?) and intra-species similarity (individual re-identification). |
| t-SNE visualization | Species cluster analysis | t-SNE on wildlife embeddings reveals natural species groupings and highlights which species the model cannot distinguish. |
| Cosine similarity | Species confusion prediction | High cosine similarity between two species' embeddings predicts high confusion rate. This informs annotation priorities and model architecture choices. |
| Vision Transformers (ViT) | DINOv2 backbone for wildlife models | ViT is the foundation for DINOv2, which is used as an alternative backbone in some wildlife detection models (including a comparison in the thesis). |
| CLIP zero-shot classification | Wildlife species without training data | CLIP can classify wildlife species using text prompts ("a photo of a marine iguana") without any training images. This is transformative for rapid surveys of new species. |

### Module 8: Detection & Segmentation (Conceptual)

| PCV Concept | Wildlife Application | Bridge Explanation |
|-------------|---------------------|-------------------|
| Object detection task definition | Core task for wildlife survey automation | Detection (locate + classify animals) is the central task. PCV defines it; the wildlife bridge implements it. |
| Anchor boxes (conceptual) | Anchor-free detection in YOLOv8 | PCV introduces anchor boxes conceptually. Modern detectors (YOLOv8) are anchor-free. The bridge module covers this evolution. |
| R-CNN family (conceptual) | Two-stage detection for high accuracy | R-CNN concepts help understand MegaDetector (YOLOv5) and the trade-off between speed and accuracy in wildlife monitoring. |
| Segmentation (conceptual) | Instance segmentation for dense colonies | Segmentation is mentioned conceptually. For dense iguana colonies, instance segmentation could delineate individuals. Not used in thesis but relevant. |

---

## Unmapped Wildlife Requirements (No PCV Equivalent)

These wildlife detection skills have no counterpart in PCV and require entirely new content:

| Requirement | Why PCV Cannot Cover It | Generating Agent |
|-------------|------------------------|-----------------|
| GSD calculation | No concept of physical pixel size in PCV | aerial_concepts_agent |
| Nadir vs oblique geometry | No camera geometry beyond pinhole model | aerial_concepts_agent |
| Orthomosaic generation | No multi-image stitching concept | aerial_concepts_agent |
| Motion blur assessment | No exposure/motion model | aerial_concepts_agent |
| YOLOv8 implementation | Module 8 is conceptual only | detection_bridge_agent |
| mAP/AP50/AP75 computation | Metrics mentioned but not coded | detection_bridge_agent |
| NMS implementation | Described but not implemented | detection_bridge_agent |
| Point-based detection (HerdNet) | Not covered at all | detection_bridge_agent |
| Tile-based inference (SAHI) | Not covered at all | exercise_generator_agent |
| Counting/density estimation | Not covered at all | detection_bridge_agent |
| HITL verification workflow | Not covered at all | exercise_generator_agent |

---

## Bridge Priority Matrix

| Priority | PCV Concept | Wildlife Extension | Effort |
|----------|-------------|-------------------|--------|
| CRITICAL | Module 8 detection (conceptual) | Hands-on YOLOv8 + metrics | HIGH (new module) |
| CRITICAL | None | GSD + aerial fundamentals | HIGH (new module) |
| HIGH | Module 5 fine-tuning | Wildlife species fine-tuning | LOW (notebook adaptation) |
| HIGH | Module 7 CLIP | Zero-shot wildlife classification | LOW (notebook adaptation) |
| HIGH | Module 3 metrics | Precision/recall for rare species | LOW (commentary addition) |
| MEDIUM | Module 7 embeddings | Species similarity analysis | LOW (notebook adaptation) |
| MEDIUM | Module 6 augmentation | Nadir-specific augmentation | LOW (commentary addition) |
| MEDIUM | Module 6 CAM | CAM on camouflaged wildlife | LOW (notebook adaptation) |
| LOW | Module 4 feature maps | Feature maps on aerial tiles | LOW (visualization swap) |
| LOW | Module 2 MLP | Historical context for wildlife CV | NONE (commentary only) |
