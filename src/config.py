import os

class Config:
    LOCAL_ROOT_DIR = "data/raw"

    NUM_CLASSES = 4
    LEARNING_RATE = 0.0001
    EPOCHS = 100
    BATCH_SIZE = 16
    RESIZE = 224

    OPTIMIZER = "Adam"

    EVAL_DIR = "data/raw/valid"
    
    S3_BUCKET = "cad-brbh-datascience"
    S3_INPUT_DIR = "s3://cad-brbh-datascience/alzheimer_images/"
    S3_TRAIN_DIR = "s3://cad-brbh-datascience/alzheimer_images/train/"
    S3_VAL_DIR = "s3://cad-brbh-datascience/alzheimer_images/valid/"
    S3_CHECKPOINT_DIR = "s3://cad-brbh-datascience/alzheimer_images/checkpoints/"
    S3_OUTPUT_DIR = "s3://cad-brbh-datascience/alzheimer_images/models/"