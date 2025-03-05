# Image Classification Alzheimer

## About

This project consists of implementing the AlexNet architecture using PyTorch to train a classification model.
This is an extension of a first project using LeNet, in search of a more robust architecture for higher resolution images.

The implementation serves as a technical showcase for medical image analysis workflows with MLOps integration.

## Dataset
- **Alzheimer's MRI Image Collection**
- Class distribution: 4 stages of Alzheimer's progression

#### Class Distribution

- **Total Class Distribution:**
  - Very Mild Demented: 2,629
  - Moderate Demented: 1,862
  - Mild Demented: 2,635
  - Non Demented: 2,774

Class distribution between splits table:

<img src="data/notes/Figure_2.png" alt="Confusion Matrix" width="500"/>
<br></br>

- Preprocessing pipeline:
  - Grayscale (As MRI images ideally consists in grayscale images (tested))
  - Normalization (mean=0.5, std=0.5)
  - Class-folder-based dataset organization
  - Data splitted by 70/15/15 for train, test validation

## Architecture Details
**AlexNet Adaptation for MRI Analysis**

<img src="data/notes/alexnet_arch.png" alt="AlexNet architecture" width="500"/>
<p><small>(Original AlexNet architecture)</small></p>
- Input: 32x32 grayscale MRI slices (1 channel)
- Feature extractor:
  - 5 convolutional blocks (Conv2D + BatchNorm + ReLU + MaxPool)
  - Kernel progression: 11→5→3→3→3 with stride/padding optimization
- Classifier:
  - 3 fully-connected layers (4096→4096→4) with dropout (p=0.5)
  - Output: 4-class probabilities (Alzheimer's stages)

## Baseline

We started with the architecture described above and the following parameters:
    LEARNING_RATE = 0.0001
    EPOCHS = 100
    BATCH_SIZE = 128

    OPTIMIZER = "SGD"
    WEIGHT_DECAY = 0.0001
    SGD_MOMENTUM = 0.9

From these parameters, I obtained the following result in the best trained model:
f1_score 0.89278
val_acc 0.88421
val_loss 0.3704
<img src="wandb/offline-run-20250304_195409-cad-alexnet-classification-RGB-2025-03-04-19-48-33-470-a3dh41-algo-1/files/media/images/confusion_matrix_image_100_34ec8d483df2e72a2f4b.png" alt="Confusion Matrix" width="500"/>


## Goals

The initial idea is to achieve at least 91% accuracy and 0.91 f1-score.

## Technologies Used
- **PyTorch**: Core framework for CNN implementation (AlexNet), DataLoader creation, and training loop management with CUDA acceleration
- **Sagemaker**: Orchestrates distributed training jobs with GPU instances, manages model artifacts/outputs, and handles hyperparameter configuration
- **WandB**: Tracks experiment metrics in real-time, logs confusion matrices, and stores training visualizations for performance analysis
- **MLOps**: Implements automated model versioning, S3-based checkpointing, and SageMaker pipelines for reproducible workflows
- **Training Optimization**: Features early stopping, best model selection by F1-score, and incremental checkpoint saving
- **AWS**: Leverages S3 for data storage/checkpoints and EC2 GPU instances via SageMaker for scalable model training
- **Albumentations**: Used to apply some transformations in random samples to apply more diversity in the original data


## Results

After implementing **Albumentations**, I got f1_score = 0.92564, val_acc 0.91579, val_loss 0.26711
I decided to apply this strategies in the transformation:
- **Albumentations**:
  - Normalize: 
  - ShiftScaleRotate:
  - <s>RandomBrightnessContrast: tried as a possibility to diversify the dataset, but it break the logic behind MRI images.</s>
  - <s>ElasticTransform: removed after some bad results. Distortions seems to affect the MRI quality.</s>


### Next steps
- Implement data augmentation techniques
  - On-the-fly or generate augmented data
- Document the code using docstrings





