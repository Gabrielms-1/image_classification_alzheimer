import os

class Config:
    PROJECT_NAME = "cad-alexnet-classification"

    LOCAL_ROOT_DIR = "file://data/raw"
    LOCAL_OUTPUT_DIR = "file://results/"

    NUM_CLASSES = 4
    LEARNING_RATE = 0.0001
    EPOCHS = 50
    BATCH_SIZE = 64
    RESIZE = 224

    #OPTIMIZER = "Adam"
    OPTIMIZER = "SGD"
    WEIGHT_DECAY = 0.0001
    SGD_MOMENTUM = 0.9
    

    EVAL_DIR = "data/raw/valid"
    
    S3_BUCKET = "cad-brbh-datascience"
    S3_INPUT_DIR = "s3://cad-brbh-datascience/alzheimer_images/"
    S3_TRAIN_DIR = "s3://cad-brbh-datascience/alzheimer_images/train/"
    S3_VAL_DIR = "s3://cad-brbh-datascience/alzheimer_images/valid/"
    S3_CHECKPOINT_DIR = "s3://cad-brbh-datascience/alzheimer_images/checkpoints"
    S3_OUTPUT_DIR = "s3://cad-brbh-datascience/alzheimer_images/models"

