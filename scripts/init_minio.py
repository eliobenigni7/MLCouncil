"""Create required MinIO buckets for MLflow artifacts."""
import boto3
from botocore.exceptions import ClientError

ENDPOINT_URL = "http://localhost:9000"
ACCESS_KEY = "mlcouncil"
SECRET_KEY = "mlcouncil_secret"
BUCKET_NAME = "mlflow-artifacts"


def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    try:
        s3.create_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' created.")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"Bucket '{BUCKET_NAME}' already exists.")
        else:
            raise
    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    print(f"Buckets available: {buckets}")


if __name__ == "__main__":
    main()
