import boto3
from botocore.config import Config
import pandas as pd
import requests
import io
import os
from dotenv import load_dotenv
from datetime import datetime  # needed for date_parser

load_dotenv()  # Load environment variables from .env


def load_csv_from_aws_s3(object_key, expiry_seconds=600, index_type=None):
    """
    Load a CSV file from AWS S3 using a pre-signed URL.

    Parameters:
    - object_key (str): Path to file in the S3 bucket (e.g. "data/stock_prices.csv")
    - expiry_seconds (int): Validity period of the pre-signed URL
    - index_type (str): Index format; one of {"multi", "date", "plain", None}
        - "multi": MultiIndex with [0, 1] and parse_dates=[1]
        - "date": Single date index_col=[0], parse_dates=[0]
        - "plain": index_col=[0] without date parsing
        - None: No index_col, read as-is

    Returns:
    - pandas.DataFrame
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
            "Key": object_key
        },
        ExpiresIn=expiry_seconds
    )

    response = requests.get(url)
    response.raise_for_status()

    if index_type == "multi":
        return pd.read_csv(io.StringIO(response.text),
                           index_col=[0, 1],
                           parse_dates=[1])
    elif index_type == "date":
        # dateparse = lambda x: datetime.strptime(x, "%Y-%m-%d") # # Define explicit date parser
        return pd.read_csv(io.StringIO(response.text),
                           index_col=[0],
                           parse_dates=[0],
                           date_format="%Y-%m-%d")
    elif index_type == "plain":
        return pd.read_csv(io.StringIO(response.text),
                           index_col=[0])
    else:
        return pd.read_csv(io.StringIO(response.text))
