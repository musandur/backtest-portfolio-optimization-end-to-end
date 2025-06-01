import boto3
from botocore.config import Config
import pandas as pd
import requests
import io


def load_data_from_aws_s3_csv(bucket_name, object_key, region="eu-central-1", expiry_seconds=600):
    """
    Load a CSV file from AWS S3 using a pre-signed URL.

    Parameters:
    - bucket_name: str, name of my  S3 bucket
    - object_key: str, key (path/filename) of the object in the bucket
    - region: str, AWS region (default: "eu-central-1")
    - expiry_seconds: int, time in seconds until the pre-signed URL expires (default: 600)

    Returns:
    - pandas.DataFrame containing the CSV data
    """
    s3 = boto3.client(
        "s3",
        region_name=region,
        config=Config(
            signature_version="s3v4"
        )
    )

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": object_key},
        ExpiresIn=expiry_seconds
    )

    #print(f"Pre-signed URL generated (valid for {expiry_seconds}s).")

    response = requests.get(url)
    response.raise_for_status()

    return pd.read_csv(io.StringIO(response.text), 
                       index_col=[0, 1],
                       parse_dates=[1])

