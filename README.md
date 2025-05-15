# Vision Transformers (ViT) and Masked Autoencoders (MAE) for Image Classification and Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project explores the integration of Vision Transformers (ViT) and Masked Autoencoders (MAE) to address core computer vision tasks, namely image classification and image segmentation. By leveraging the self-supervised learning capabilities of MAE, a ViT model is pretrained on large amounts of unlabeled data to overcome challenges related to data scarcity. The pretrained ViT-MAE model is then fine-tuned for multiple downstream tasks, demonstrating strong performance and generalization across datasets including MNIST, PASCAL VOC, and CIFAR-10.

The project aims to highlight the synergy between ViTs and MAEs, providing a unified framework that can perform both coarse-grained classification and fine-grained segmentation without relying on task-specific architectures.

---

## Table of Contents

- [Introduction](#introduction)  
- [Background](#background)  
- [Datasets](#datasets)  
- [Methodology](#methodology)  
- [Results](#results)  
- [Implementation Details](#implementation-details)  
- [Challenges](#challenges)  
- [Future Work](#future-work)  
- [References](#references)  

---

## Introduction

Vision Transformers (ViT) have emerged as a promising architecture for visual tasks, inspired by the success of transformers in natural language processing. However, ViTs require large-scale labeled datasets and significant computational resources. Masked Autoencoders (MAE) address these issues by enabling self-supervised pretraining through reconstructing masked portions of the input images, thus reducing dependency on labeled data.

This project combines ViT with MAE pretraining to build a robust framework capable of excelling in multiple vision tasks, including image classification and segmentation.

---

## Background

- **Vision Transformers (ViT):**  
  - Treat images as sequences of patches (tokens).  
  - Use global self-attention mechanisms to capture long-range dependencies.  
  - Require large-scale datasets for training.

- **Masked Autoencoders (MAE):**  
  - Self-supervised learning method.  
  - Mask a large portion of input image patches (e.g., 75%) and train the model to reconstruct the missing parts.  
  - Use asymmetric architecture: a heavy encoder and a lightweight decoder.  
  - Improve training efficiency and representation learning.

- **CNNs vs Transformers:**  
  - CNNs excel at local feature extraction but are limited in capturing global context.  
  - Transformers capture global relationships but are computationally expensive.  
  - Hybrid models combine the strengths of both.

---

## Datasets

| Dataset    | Task(s)                 | Images                   | Classes        | Training Size | Test Size  |
|------------|-------------------------|--------------------------|----------------|---------------|------------|
| MNIST      | Classification          | 28x28 grayscale digits   | 10 digits      | 60,000        | 10,000     |
| PASCAL VOC | Detection, Segmentation | ~500x500 varied size     | 20 object classes + background | 5,000         | 5,000      |
| CIFAR-10   | Classification          | 32x32 RGB images         | 10 classes     | 50,000        | 10,000     |

---

## Methodology

### Vision Transformer (ViT)

- Splits images into fixed-size patches (tokens).  
- Embeds and feeds tokens into a transformer encoder.  
- Employs self-attention to capture global context across patches.  
- Pretrained on large datasets like ImageNet-21k.

### Masked Autoencoders (MAE)

- Randomly masks a large percentage of input patches (typically 75%).  
- Encoder processes visible patches only.  
- Lightweight decoder reconstructs missing patches.  
- Trained using reconstruction loss (Mean Squared Error).

### Training Workflow

1. **Pretraining (Self-supervised):**  
   - Train MAE model to reconstruct masked image patches from visible ones.

2. **Fine-tuning:**  
   - Add classification or segmentation head to the pretrained encoder.  
   - Fine-tune on labeled datasets (MNIST, PASCAL VOC, CIFAR-10).

3. **Evaluation:**  
   - Classification accuracy for classification tasks.  
   - Intersection-over-Union (IoU) and pixel accuracy for segmentation.

---

## Results

### MNIST

- **Classification Accuracy:** ~99.28% after fine-tuning.  
- **Segmentation Accuracy:** ~94.42% (on segmentation tasks with adapted MNIST images).  

### PASCAL VOC

- Classification accuracy reached ~98.19%.  
- Segmentation accuracy around ~76.09%.  
- Demonstrated robustness and scalability on complex real-world data.

### CIFAR-10

- Achieved classification accuracy of ~96.72%.  
- Showed consistent performance with the ViT-MAE model.

---

## Implementation Details

- **Model Architecture:** ViT encoder combined with MAE pretraining.  
- **Optimization:** Adam optimizer with learning rate 1e-4.  
- **Batch Size:** 16 for training and validation.  
- **Epochs:** 3 for MNIST tasks; up to 15 for PASCAL VOC and CIFAR-10.  
- **Loss Functions:**  
  - MSE loss for MAE pretraining.  
  - Cross-Entropy loss for classification and segmentation fine-tuning.  
- **Data Augmentation:** Applied standard augmentations such as resizing, normalization, horizontal flips.

---

## Challenges

- High computational resource requirements for training ViT-MAE models.  
- Tuning hyperparameters like mask ratio, patch size, and learning rate critical for stable training.  
- Data augmentation essential to improve generalization.  
- CNN models sometimes converge faster and achieve higher accuracy on certain datasets (e.g., PASCAL VOC classification).

---

## Future Work

- Extending ViT-MAE frameworks to 3D medical imaging tasks.  
- Exploring adversarial training for improved segmentation robustness in noisy environments.  
- Optimizing model architectures for deployment on resource-constrained devices.  
- Investigating hybrid models combining convolution and transformer layers for efficiency gains.  
- Further studies on fine-tuning large-scale transformer models.

---

## References

1. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," 2020.  
2. He et al., "Masked Autoencoders Are Scalable Vision Learners," 2022.  
3. Krizhevsky and Hinton, "Learning multiple layers of features from tiny images," 2009.  
4. Everingham et al., "The PASCAL Visual Object Classes Challenge 2012," 2012.  
5. LeCun et al., "MNIST handwritten digit database," 2010.

---



If you want, I can also help generate example code snippets or usage instructions for the project.  
Would you like that?
