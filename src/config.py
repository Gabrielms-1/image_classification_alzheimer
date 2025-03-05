import os

class Config:
    PROJECT_NAME = "cad-alexnet-classification"

    LOCAL_ROOT_DIR = "file://data/raw"
    LOCAL_OUTPUT_DIR = "file://results/"

    NUM_CLASSES = 4
    LEARNING_RATE = 0.0005
    EPOCHS = 100
    BATCH_SIZE = 128
    RESIZE = 224

    OPTIMIZER = "SGD"
    WEIGHT_DECAY = 0.0001
    SGD_MOMENTUM = 0.85

    AUGMENTATION_TRANSFORMATIONS = {
        "p":0.2,
        
        "shift_limit":0.05, 
        "scale_limit":0.05, 
        "rotate_limit":15, 
        "border_mode":"constant", 
        
        # "alpha":1, 
        # "sigma":50, 
        # "alpha_affine":50, 
        # "interpolation":"linear", 
    }  

    EVAL_DIR = "data/raw/valid"
    
    S3_BUCKET = "cad-brbh-datascience"
    S3_INPUT_DIR = "s3://cad-brbh-datascience/alzheimer_images/"
    S3_TRAIN_DIR = "s3://cad-brbh-datascience/alzheimer_images/train/"
    S3_VAL_DIR = "s3://cad-brbh-datascience/alzheimer_images/valid/"
    S3_CHECKPOINT_DIR = "s3://cad-brbh-datascience/alzheimer_images/checkpoints"
    S3_OUTPUT_DIR = "s3://cad-brbh-datascience/alzheimer_images/models"

