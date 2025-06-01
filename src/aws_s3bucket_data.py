# import boto3
# from botocore.config import Config
# import pandas as pd
# import requests
# import io
# import os


# def load_data_from_aws_s3_csv(bucket_name, object_key, region="eu-central-1", expiry_seconds=600):
#     """
#     Load a CSV file from AWS S3 using a pre-signed URL.

#     Parameters:
#     - bucket_name: str, name of my  S3 bucket
#     - object_key: str, key (path/filename) of the object in the bucket
#     - region: str, AWS region (default: "eu-central-1")
#     - expiry_seconds: int, time in seconds until the pre-signed URL expires (default: 600)

#     Returns:
#     - pandas.DataFrame containing the CSV data
#     """
#     s3 = boto3.client(
#         "s3",
#         region_name=region,
#         config=Config(
#             signature_version="s3v4"
#         )
#     )

#     url = s3.generate_presigned_url(
#         "get_object",
#         Params={"Bucket": bucket_name, "Key": object_key},
#         ExpiresIn=expiry_seconds
#     )

#     #print(f"Pre-signed URL generated (valid for {expiry_seconds}s).")

#     response = requests.get(url)
#     response.raise_for_status()

#     return pd.read_csv(io.StringIO(response.text), 
#                        index_col=[0, 1],
#                        parse_dates=[1])



import boto3
from botocore.config import Config
import pandas as pd
import requests
import io
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env


def load_data_from_aws_s3_csv(expiry_seconds=600):
    """
    Load a CSV file from AWS S3 using a pre-signed URL.
    Bucket name, object key, and region are taken from .env file.

    Returns:
    - pandas.DataFrame containing the CSV data
    """
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4")
    )

    url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": os.getenv("S3_BUCKET"),
            "Key": os.getenv("S3_OBJECT_KEY")
        },
        ExpiresIn=expiry_seconds
    )

    response = requests.get(url)
    response.raise_for_status()

    return pd.read_csv(io.StringIO(response.text),
                       index_col=[0, 1],
                       parse_dates=[1])


