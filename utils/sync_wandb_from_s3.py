import os
import boto3
import subprocess

s3_bucket = "cad-brbh-datascience"
s3_prefix = "alzheimer_images/checkpoints/wandb"
local_dir = "wandb"

if not os.path.exists(local_dir):
    os.makedirs(local_dir)

s3 = boto3.resource('s3')
bucket = s3.Bucket(s3_bucket)


for obj in bucket.objects.filter(Prefix=s3_prefix):
    target = os.path.join(local_dir, os.path.relpath(obj.key, s3_prefix))
    if "offline" not in target:
        continue
    if not os.path.exists(os.path.dirname(target)):
        os.makedirs(os.path.dirname(target))
    print(f"Downloading {obj.key} from s3 to {target}")
    bucket.download_file(obj.key, target)

for checkpoint in os.listdir(local_dir):
    if "offline" in checkpoint:
        print(f"Syncing {checkpoint} to wandb remote")
        subprocess.run(["wandb", "sync", os.path.join(local_dir, checkpoint)])

print("Syncing complete")
    

