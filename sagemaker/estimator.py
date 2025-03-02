import sagemaker
from sagemaker.pytorch import PyTorch
import configparser
from datetime import datetime
from zoneinfo import ZoneInfo
from config import Config

network_config = configparser.ConfigParser()
network_config.read("sagemaker/credentials.ini")

sagemaker_session_bucket = Config.S3_BUCKET
