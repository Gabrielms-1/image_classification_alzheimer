import os
import boto3
import subprocess
from datetime import datetime
import uuid 

s3_bucket = "cad-brbh-datascience"
s3_prefix = "alzheimer_images/checkpoints/wandb"
local_dir = "wandb"

if not os.path.exists(local_dir):
    os.makedirs(local_dir)

s3 = boto3.resource('s3')
bucket = s3.Bucket(s3_bucket)

def sync_wandb_from_s3(obj):
    target = os.path.join(local_dir, os.path.relpath(obj.key, s3_prefix))
    if not os.path.exists(os.path.dirname(target)):
        os.makedirs(os.path.dirname(target))
    
    bucket.download_file(obj.key, target)

def sync_wandb_to_s3(checkpoint):
    if "offline" in checkpoint:
        print(f"Syncing {checkpoint} to wandb remote")
        subprocess.run(["wandb", "sync", os.path.join(local_dir, checkpoint)])


if __name__ == "__main__":
    print("Syncing wandb from s3 to local")
    for obj in bucket.objects.filter(Prefix=s3_prefix):
        sync_wandb_from_s3(obj)
    
    print("Syncing local wandb to s3")    
    for checkpoint in os.listdir(local_dir):
        sync_wandb_to_s3(checkpoint)



print("Syncing complete")
    

