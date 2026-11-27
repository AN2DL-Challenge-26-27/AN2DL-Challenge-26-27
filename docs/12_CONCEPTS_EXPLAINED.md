# 12 · Concepts and techniques explained

Previous: [11 · Training in the cloud](11_CLOUD_TRAINING.md)

This page explains every idea and technique used in the project, for readers new to machine learning. Each section says **what the idea is**, **why we needed it here**, and **where it is used**, with numbers from our own experiments. It can be read on its own; the other documents give the details. A glossary is at the end.

**Contents**

1. [The task and how it is scored](#1-the-task-and-how-it-is-scored)
2. [Understanding and cleaning the data](#2-understanding-and-cleaning-the-data)
3. [Measuring progress honestly: validation](#3-measuring-progress-honestly-validation)
4. [Neural networks and pretrained models](#4-neural-networks-and-pretrained-models)
5. [How a model is trained](#5-how-a-model-is-trained)
6. [Training techniques we used](#6-training-techniques-we-used)
7. [Data augmentation and the test-domain shift](#7-data-augmentation-and-the-test-domain-shift)
8. [Making predictions: TTA, ensembles, robustness](#8-making-predictions-tta-ensembles-robustness)
9. [Engineering practices](#9-engineering-practices)
10. [Planned techniques](#10-planned-techniques)
11. [Glossary](#11-glossary)

---

## 1. The task and how it is scored

### Image classification

A **classifier** looks at an image and answers "which of these categories is it?". Here there are 10 categories of galaxy shape: smooth round galaxies, spirals with tight or loose arms, edge-on discs, merging pairs, irregular galaxies, and so on (full list in [01](01_DATASET_AND_TASK.md)). The model does not output just one answer. It outputs ten **probabilities**, one per class, that add up to 1, for example "Gnocchi 70%, Ravioli 25%, others 5%". The class with the highest probability is the prediction.

We have 4,328 labelled training images (we know the correct class) and 14,736 test images (we do not). Kaggle compares our test predictions with the hidden correct labels.

### Precision, recall, and F1

For one class, for example "loose spiral":

* **Precision:** of the images we *called* loose spiral, how many really are? High precision = few false alarms.
* **Recall:** of the images that *really are* loose spirals, how many did we find? High recall = few misses.
* **F1** combines both into one number: F1 = 2 · precision · recall / (precision + recall). It is high only when both are high. A model that calls everything "loose spiral" has perfect recall but terrible precision, so its F1 is low.

### Weighted F1 (the competition metric)

The competition computes F1 for each of the 10 classes and averages them, **weighting each class by how many test images belong to it**. Large classes (round galaxies, spirals) therefore count much more than tiny ones (cigar-shaped galaxies, about 1.4% of images). This matters for our choices: improving a big class by 1 point helps the score far more than improving a rare one. It is also why we eventually removed the class weights from the loss (section 6).

### Public and private leaderboard

Kaggle splits the test set in two hidden parts. The **public** leaderboard shows the score on one part right away. The **final ranking** uses the other, **private** part. A model tuned too closely to the public part can drop on the private part. So we make decisions with our own validation (section 3) and use the public leaderboard only as a check. Our best public score so far is **0.8441**.

---

## 2. Understanding and cleaning the data

Before training anything we studied the data (exploratory data analysis, [02](02_EDA_FINDINGS.md)) and cleaned it ([03](03_PREPROCESSING.md)). Three surprises were found.

### Artifacts: alien overlays and blank frames

332 training images had a cartoon alien face pasted on top. It appears in every class equally, never in the test set, and hides the galaxy, so a model would only learn something useless from it. These images were removed.

**How they were found:** colours can be described by hue (which colour), saturation (how strong), and value (how bright), the **HSV** colour model. The alien's skin is a narrow range of pale beige. Some bright yellow galaxies have the same colour, so the detector also requires dark "eye" pixels inside the beige region. This simple rule matched, image for image, a check with a neural network. Three images contained only grey noise (**blank frames**); the one in the training set was removed.

### Near-duplicates and the "star-field fingerprint"

Many training images show **the same patch of sky** two to six times: one clean original plus brighter, noisier copies. Exact file comparison cannot find them, because the noise makes every copy different. Comparing whole images does not work either, since any two centred galaxies look alike.

The trick: ignore the galaxy in the middle and compare the **background stars** around it, whose pattern is unique to each patch of sky, like a fingerprint. Technically:

* **Band-pass filter (difference of Gaussians):** blurring an image a little and subtracting a stronger blur keeps medium-sized details (stars) and removes both pixel noise and the large smooth galaxy.
* **Correlation:** a number from −1 to 1 that says how similar two patterns are; 1 = identical.
* **The 8 orientations (D8 symmetry):** the comparison is repeated for the 4 rotations and their mirror images, in case a copy was rotated.

A threshold was set from data: real copies match without rotation, while chance matches spread evenly over all 8 orientations. This found 721 groups of copies.

### Label noise

Almost every copy carries a **different label** than its original, always the neighbouring class: round ↔ in-between round, or barred ↔ tight spiral. One galaxy cannot be two classes, so these are wrong labels (**label noise**). The labels of this kind of dataset come from volunteers voting on galaxy shapes (Galaxy Zoo), so borderline galaxies are genuinely ambiguous. We tried three **label policies**: use the original's label for copies, drop the copies, or keep the labels as given. They were within noise of each other.

### Class imbalance

The largest class has about 9 times more images than the smallest (Penne/cigar-shaped: 56 kept images). Rare classes are hard to learn and their scores jump around. Section 6 describes how we dealt with this.

---

## 3. Measuring progress honestly: validation

### Training data vs validation data

A model must be judged on images it **did not see during training**. Otherwise it can simply memorise the answers. So part of the labelled data is held out as a **validation set**: the model never trains on it, and its score there estimates how well it will do on new images. This is our main tool for comparing ideas.

### Overfitting

**Overfitting** means the model memorises the training images instead of learning general rules. You see it when training accuracy keeps rising but validation accuracy stops or falls. Example from our runs: ConvNeXt-Small reached 97% training accuracy but only about 84% on validation. Several techniques in section 6 exist to fight this.

### K-fold cross-validation

Instead of one fixed split, the labelled data is divided into **5 parts (folds)**. Each fold takes a turn as the validation set while the other four are used for training:

```text
Fold 0: [val ][train][train][train][train]
Fold 1: [train][val ][train][train][train]   ... and so on
```

Every image is validated exactly once, by a model that did not see it. These are called **out-of-fold (OOF)** predictions, and they give a score over all about 3,000 images instead of about 600, which is far more reliable. Most experiments so far used fold 0 only, to save time; 5-fold training is the next step ([11](11_CLOUD_TRAINING.md)).

### Stratified and grouped folds (preventing leakage)

* **Stratified:** every fold gets the same share of each class, so no fold is unusually easy or hard.
* **Grouped:** all copies of the same patch of sky stay in the same fold. If a copy were in training and its original in validation, the model would recognise the scene, not the galaxy shape, and the validation score would be falsely high. This is called **leakage**. We use scikit-learn's `StratifiedGroupKFold`, which does both.

### How precise is a validation score?

With about 600 validation images, a score has roughly **±1.5 points of random noise** (statistical standard error). Differences smaller than that may be luck. This is why we did not over-interpret small gains, and why the 5-fold out-of-fold score matters.

### Validation vs the leaderboard, and "test-like" validation

Our Kaggle scores were consistently about 1.8 points below validation, but the **ranking** of submissions was the same. Part of the gap is **optimism**: we pick the best epoch and the best ensemble using the same validation images, which flatters them. The other part is that the test images are corrupted differently (section 7).

We therefore added **test-like validation**: the same validation images, but with test-like corruptions applied before scoring. It predicted the leaderboard better. For example, it correctly showed that adding DINOv3 helps, while clean validation said the opposite. We now choose models by both scores together.

---

## 4. Neural networks and pretrained models

### Neural networks, in one paragraph

A neural network is a large mathematical function with millions of adjustable numbers called **weights** (or parameters). An image goes in as numbers (pixel values); ten class scores come out. Training adjusts the weights so that the outputs match the correct labels. For images, networks are built from layers that detect simple patterns first (edges, dots), then combinations (arcs, blobs), then whole structures (spiral arms, a central bulge).

### Convolutional networks (CNNs) and ConvNeXt

A **convolution** slides a small filter over the image and responds wherever a certain local pattern appears, like a stamp that lights up on matching spots. A **convolutional neural network (CNN)** stacks many such layers. CNNs are good at local texture and shape, and they need relatively little data. **ConvNeXt** (2022) is a modern CNN design; we used the Tiny (28 million weights) and Small (50 million weights) versions.

### Vision Transformers (ViT)

A **Vision Transformer** cuts the image into small square **patches** (for example 14 × 14 pixels), treats them like words in a sentence, and uses **attention**: every patch can directly look at every other patch and decide what is relevant. This lets the model compare distant parts of a galaxy, for example whether both spiral arms look alike, from the first layer on. Transformers usually need more data than CNNs, which pretrained models solve.

### Pretraining and transfer learning (fine-tuning)

Training a network from scratch needs millions of images; we have about 4,000. The standard solution is **transfer learning**: start from a network that was already **pretrained** on a huge general image collection (for example ImageNet, 14 million photos). Its early layers already know edges, textures, and shapes. We replace its last layer (the **head**) with a new 10-class head and continue training on our galaxies. This is called **fine-tuning**, and it needs far less data and time. The competition allows only general-purpose pretrained models, not models trained on astronomy images.

### Self-supervised pretraining: DINOv2 and DINOv3

Normal pretraining needs labels ("this photo is a dog"). **Self-supervised** pretraining learns from unlabelled images: a "student" network learns to produce the same description for different crops and versions of an image as a "teacher" network. **DINOv2** (Meta, 2023, 142 million images) and **DINOv3** (Meta, 2025, about 1.7 billion images) are Vision Transformers trained this way. They produce very general features.

In our project, DINOv2 became the best single model (validation 0.852). DINOv3 was weaker on clean validation but the most robust to corruptions (section 8).

### The `timm` library

`timm` (PyTorch Image Models) is a library with hundreds of pretrained networks under one interface. Our **model presets** (`MODEL=dinov2_base` in a config file) map short names to exact `timm` checkpoints, plus suitable settings for an 8 GB GPU ([08](08_TRAINING_AND_RESULTS.md)).

---

## 5. How a model is trained

### Batches, steps, and epochs

Images are processed in small groups called **batches** (for example 32 images). After each batch the weights are adjusted once; that is one **step**. One full pass over all training images is an **epoch**. Our runs use 25–30 epochs.

### Softmax, loss, and cross-entropy

The network's raw outputs are turned into probabilities by **softmax**. The **loss** is a number that measures how wrong the predictions are; training tries to make it small. For classification we use **cross-entropy**: it is small when the model gives high probability to the correct class, and large when it is confidently wrong.

### Gradient descent and AdamW

To reduce the loss, the training program computes, for every weight, in which direction a small change would reduce the loss. This set of directions is the **gradient**, computed by **backpropagation**. The weights then take a small step in that direction (**gradient descent**). **AdamW** is a popular, robust variant that adapts the step size for each weight and includes weight decay (section 6).

### Learning rate, warm-up, and cosine schedule

The **learning rate** is the size of each step. Too large and training becomes unstable; too small and it is slow or stuck. We use a **schedule**:

* **Warm-up:** start very small for the first epoch, so the new random head does not damage the pretrained weights.
* **Cosine decay:** then lower the rate smoothly towards almost zero by the end, which lets the model settle into a good solution.

### Choosing the best epoch and checkpoints

After every epoch the model is scored on validation. The weights of the best epoch are saved as **`best.pt`**, a **checkpoint** (a file with all the weights). A second file, **`last.pt`**, stores the complete training state after every epoch, so an interrupted run can **resume** instead of starting over.

---

## 6. Training techniques we used

| Technique | The idea in one sentence | How it helped here |
| --- | --- | --- |
| **Class weights** | Make mistakes on rare classes count more in the loss. | Used at first (1/√class size). Later removed: weighted F1 rewards the big classes, so the research recipe trains unweighted. |
| **Label smoothing** | Train towards "90% sure" instead of "100% sure" of the label. | Reduces over-confidence; useful because some labels are wrong or ambiguous. |
| **Weight decay** | Gently pull all weights towards zero. | A standard brake against overfitting. |
| **Drop path (stochastic depth)** | Randomly skip whole layers during training. | The network cannot rely on any single layer, which reduces overfitting. |
| **MixUp** | Blend two images and their labels (e.g. 70% galaxy A + 30% galaxy B). | Teaches calmer, less over-confident predictions. **CutMix** (pasting a square of one image into another) was deliberately not used: it would put spiral arms onto a smooth galaxy and create fake "merging" galaxies. |
| **EMA of weights** | Keep a slowly moving average of the weights and evaluate that. | Smooths out noise between epochs; works like a cheap ensemble of checkpoints. |
| **Layer-wise learning-rate decay** | Early layers learn slowly, later layers and the head faster. | Protects the general features the pretrained early layers already know. |
| **Mixed precision (bfloat16 / float16)** | Compute with 16-bit instead of 32-bit numbers. | About 2× faster and half the memory. The laptop uses bfloat16; Colab/Kaggle T4 GPUs need float16 with a **gradient scaler**, which prevents very small gradients from becoming zero. |
| **Gradient accumulation** | Add up gradients over several small batches before one step. | Lets an 8 GB GPU train large models with an effective batch of 32 (8 images × 4). |
| **Gradient checkpointing** | Store fewer intermediate results and recompute them when needed. | Saves memory for large models at the cost of some speed. |

The first four techniques plus MixUp, EMA, and layer decay form what docs/10 calls the **research recipe**. It did not help ConvNeXt (R1) but worked well for DINOv2 and DINOv3 (R2, N2).

---

## 7. Data augmentation and the test-domain shift

### Data augmentation

**Augmentation** shows the model a slightly changed version of each training image every time it sees it. That makes 4,000 images feel like many more, and it teaches the model which changes do *not* matter.

* **Geometric:** galaxies have no "up", so images are rotated by any angle, mirrored, and slightly scaled. Corners created by rotation are filled with a mirror image of the border (**reflection padding**) instead of black.
* **Photometric:** brightness, contrast, colour casts, gamma, noise, and blur are changed at random.
* **Synthetic artifacts:** occasionally a coloured rectangle, a streak, a colour flood, or a bright band is added.

A preview grid is in [05](05_AUGMENTATION_AND_TRANSFORMS.md).

### Domain shift

**Domain shift** means the test images differ systematically from the training images. Our test set has colour casts, darkening, noise, and overlays. By fitting the few test images that duplicate training images, we measured how strong these changes are and set the augmentation ranges to match. Training on corrupted images teaches the model to ignore the corruptions.

### Normalisation

Before entering the network, pixel values are shifted and scaled to the same range the pretrained model saw during pretraining (ImageNet mean and standard deviation). This is a fixed step, identical for training and test images.

---

## 8. Making predictions: TTA, ensembles, robustness

### Test-time augmentation (TTA)

At prediction time, the model also looks at the 4 rotations and their mirror images (8 views) and averages the 8 probability sets. Galaxies have no orientation, so all views should agree, and averaging reduces random errors. In our runs TTA helped some models (DINOv3: 0.832 → 0.843) and hurt others (DINOv2: 0.851 → 0.839), so it is decided per model.

### Ensembles

An **ensemble** averages the probabilities of several models. If the models make *different* mistakes, the average corrects many of them. If they make the *same* mistakes, nothing is gained. So **diversity** matters more than the number of models:

* Three similar ConvNeXt models shared about 82% of their errors and gained little from each other.
* Adding DINOv2, a different family, raised the Kaggle score from 0.8371 to 0.8424.
* Adding DINOv3 as well raised it to **0.8441**.

To check diversity we count **shared errors**: the share of one model's mistakes that the other model also makes.

### Robustness

A model is **robust** if its predictions stay correct when the input is disturbed. We measured robustness by scoring validation images with test-like corruptions. DINOv3 lost almost nothing (−0.1 to −0.9 points); DINOv2 and ConvNeXt lost 1–3 points. That explained why DINOv3 lost much less from validation to the real test set (0.7 instead of about 1.8 points), and why it helps the ensemble.

### Irreducible error

Some validation images are wrong for *every* model, about 10%. Many look ambiguous or mislabelled. No model can fix a wrong label, which is why a weighted F1 near 0.90 is close to the realistic ceiling for this task.

---

## 9. Engineering practices

* **Configuration files.** Every training run is described by a small text file (`configs/experiments/<name>.env`). Each run also saves the final settings with the source of every value. Anyone can see exactly what produced a result and repeat it ([08](08_TRAINING_AND_RESULTS.md)).
* **Reproducibility and seeds.** Random choices (shuffling, augmentation) start from a fixed **seed** (42), so runs can be repeated. GPU arithmetic is not perfectly deterministic, so repeats match very closely but not bit for bit.
* **Checkpoints and resume.** `last.pt` after every epoch means a crash or a cloud disconnect loses at most one epoch.
* **Checksums.** The shared weights zip records a **SHA-256** fingerprint of every model file, so a corrupted download is detected ([09](09_MODEL_WEIGHTS_AND_REPRODUCTION.md)).
* **GPU temperature guard.** The laptop GPU reaches 85–86 °C under full load. `experiments/gpu_guard.py` pauses training at 82 °C and resumes it at 72 °C. Results are unchanged; runs take about twice as long.
* **Cloud training.** Google Colab and Kaggle offer free GPUs without the laptop's heat problem. The notebooks run a list of config files and save everything to Google Drive ([11](11_CLOUD_TRAINING.md)).
* **Version control.** Code, configs, manifests, and lightweight results are kept in git on GitHub; large files (data, weights) are on Google Drive.

---

## 10. Planned techniques

* **Pseudo-labelling.** Use the current best ensemble to label the 14,736 unlabelled test images, keep only the confident predictions, and add them to training. The test set contains most of the unique galaxies, so this could help a lot. It must be done carefully, because wrong pseudo-labels teach wrong things.
* **Calibration (temperature scaling).** Some models are over-confident. Dividing their raw outputs by a fitted "temperature" makes the probabilities honest before averaging models.
* **Per-class multipliers.** Slightly raise or lower each class's probability to directly maximise weighted F1, fitted on out-of-fold predictions so that it does not overfit.

---

## 11. Glossary

| Term | Meaning |
| --- | --- |
| Accuracy | Share of images classified correctly. |
| Attention | Mechanism in transformers that lets every image patch weigh information from every other patch. |
| Augmentation | Random changes to training images (rotation, colour, noise, …) so the model learns what does not matter. |
| Batch | A small group of images processed together before one weight update. |
| Checkpoint | A saved file with a model's weights (`best.pt`) or full training state (`last.pt`). |
| CNN | Convolutional neural network: built from filters that detect local patterns. |
| Cross-entropy | The loss used for classification: small when the correct class gets high probability. |
| Cross-validation (k-fold) | Splitting data into k parts, each used once for validation. |
| Domain shift | Systematic difference between training and test images. |
| Ensemble | Combining several models, here by averaging their probabilities. |
| Epoch | One full pass over the training images. |
| F1 / weighted F1 | Combination of precision and recall per class; weighted by class size for the final score. |
| Fine-tuning | Continuing to train a pretrained model on a new task. |
| Fold | One of the k parts in cross-validation. |
| Gradient | For every weight, the direction that reduces the loss. |
| Head | The last layer of a network, which outputs the class scores; replaced when fine-tuning. |
| Label noise | Training labels that are wrong or ambiguous. |
| Leakage | Information from the validation set reaching training, which makes validation scores look too good. |
| Learning rate | Size of each weight update. |
| Loss | Number measuring how wrong the model's predictions are; training minimises it. |
| Out-of-fold (OOF) | Predictions for each training image from the fold model that did not see it. |
| Overfitting | Memorising training images instead of learning general rules. |
| Precision / recall | Share of predicted members that are correct / share of true members that are found. |
| Pretrained model | A network already trained on a large dataset, used as a starting point. |
| Pseudo-labelling | Using a model's confident predictions on unlabelled images as extra training labels. |
| Robustness | How well predictions survive disturbed inputs (colour casts, noise, overlays). |
| Seed | Starting value for random number generation; makes runs repeatable. |
| Self-supervised learning | Pretraining on unlabelled images by solving tasks built from the images themselves. |
| Softmax | Turns raw network outputs into probabilities that sum to 1. |
| Stratified / grouped split | Same class mix in every fold / related images (copies) kept in the same fold. |
| Test-like validation | Validation images scored after applying test-like corruptions. |
| TTA | Test-time augmentation: averaging predictions over several transformed views of an image. |
| Validation set | Labelled images held out from training, used to measure progress honestly. |
| Vision Transformer (ViT) | Network that splits an image into patches and relates them with attention. |
| Weights | The adjustable numbers inside a network that training changes. |
