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

trainin_dir = Config.S3_TRAIN_DIR
val_dir = Config.S3_VAL_DIR

timestamp = datetime.now(ZoneInfo('America/Sao_Paulo')).strftime('%Y%m%d_%H-%M-%S')

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    instance_type="ml.g5.4xlarge",
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
    base_job_name=f"{Config.PROJECT_NAME}-RGB",
    tags=[
        {"Key": "Application", "Value": network_config['TAGS']['application']},
        {"Key": "Cost Center", "Value": network_config['TAGS']['cost_center']}
    ],
    subnets=network_config['NETWORK']['subnets'].split(','),
    security_group_ids=network_config['NETWORK']['security_group_ids'].split(','),
    checkpoint_s3_uri=os.path.join(Config.S3_CHECKPOINT_DIR, f"{timestamp}"),
    checkpoint_local_path="/opt/ml/checkpoints",
    output_path=Config.S3_OUTPUT_DIR + f"/{timestamp}",
    environment={
        "WANDB_API_KEY": network_config["WANDB"]["wandb_api_key"],
        "WANDB_MODE": "offline",
        "WANDB_DIR": "/opt/ml/model/",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    },
    metric_definitions=[
        {'Name': 'epoch', 'Regex': "EPOCH: ([0-9\\.]+)"},
        {'Name': 'train:loss', 'Regex': 'train_loss: ([0-9\\.]+)'},
        {'Name': 'train:accuracy', 'Regex': 'train_accuracy: ([0-9\\.]+)'},
        {'Name': 'val:loss', 'Regex':  'val_loss: ([0-9\\.]+)'},
        {'Name': 'val:accuracy', 'Regex':  'val_accuracy: ([0-9\\.]+)'}
    ],
    enable_sagemaker_metrics=True,
    requirements_file="requirements.txt"
)

estimator.fit(
    inputs={
        "train": trainin_dir,
        "val": val_dir
    }
)
