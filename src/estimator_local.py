import sagemaker
from sagemaker.pytorch import PyTorch
import configparser
from datetime import datetime
from zoneinfo import ZoneInfo
import os

from config import Config

network_config = configparser.ConfigParser()
network_config.read("sagemaker/credentials.ini")

sagemaker_session_bucket = Config.S3_BUCKET

session = sagemaker.Session(
    default_bucket=sagemaker_session_bucket
)

role = sagemaker.get_execution_role()

trainin_dir = Config.LOCAL_ROOT_DIR + "/train"
val_dir = Config.LOCAL_ROOT_DIR + "/valid"

timestamp = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d-%m-%T_%H-%M")

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    instance_type="local",
    instance_count=1,
    role=role,
    pytorch_version="2.2",
    framework_version="2.2",
    py_version="py310",
    hyperparameters={
        "model_name": Config.PROJECT_NAME,
        "epochs": Config.EPOCHS,
        "batch_size": Config.BATCH_SIZE,
        "learning_rate": Config.LEARNING_RATE,
        "num_classes": Config.NUM_CLASSES,
        "resize": Config.RESIZE,
    },
    base_job_name=f"{Config.PROJECT_NAME}_RGB",
    output_path=os.path.join(Config.LOCAL_OUTPUT_DIR, timestamp),
    environment={
        "WANDB_API_KEY": network_config["WANDB"]["wandb_api_key"],
        "WANDB_MODE": "online",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    },
    metric_definitions=[
        {'Name': 'train:loss', 'Regex': 'Loss: ([0-9\\.]+)'},
        {'Name': 'train:accuracy', 'Regex': 'Accuracy: ([0-9\\.]+)'},
    ],
    requirements_file="requirements.txt"
)

estimator.fit(
    inputs={
        "train": trainin_dir,
        "val": val_dir
    }
)
