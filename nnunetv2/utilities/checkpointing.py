import io
import boto3
import torch
from botocore.exceptions import ClientError


def save_checkpoint_s3(state_dict, epoch, bucket_name, mlflow_run_id, aws_region="eu-central-1"):
    """
    Save a PyTorch model's state_dict as a checkpoint to an S3 bucket.

    This function serializes the given PyTorch state_dict into an in-memory buffer
    and uploads it directly to an S3 bucket. The checkpoint is stored with a key
    that includes the MLflow run ID and epoch number for traceability. It includes
    error handling for common issues such as missing buckets or network errors.

    Args:
        state_dict (dict): The PyTorch model's state_dict to be saved.
        epoch (int): The epoch number associated with the checkpoint.
        bucket_name (str): The name of the S3 bucket where the checkpoint will be stored.
        mlflow_run_id (str): The MLflow run ID associated with the checkpoint for traceability.
        aws_region (str): The AWS region where the S3 bucket is located. Defaults to "eu-central-1".

    Returns:
        str: The S3 key (path) of the uploaded checkpoint file.

    Raises:
        FileNotFoundError: If the specified S3 bucket does not exist.
        botocore.exceptions.ClientError: If there is an error during the upload process.
        RuntimeError: For unexpected errors during serialization or upload.

    Example:
        >>> state_dict = {"layer1.weight": torch.randn(10, 10)}
        >>> save_checkpoint_s3(state_dict, epoch=1, bucket_name="my-bucket", mlflow_run_id="12345")
        'checkpoints/run_12345/epoch_1.pt'
    """
    # Create an in-memory bytes buffer
    buffer = io.BytesIO()

    try:
        # Save the model's state_dict to the buffer
        torch.save(state_dict, buffer)

        # Reset the buffer's position to the beginning
        buffer.seek(0)

        # Define the S3 path for the checkpoint
        checkpoint_path = f'checkpoints/run_{mlflow_run_id}/epoch_{epoch}.pt'

        # Initialize S3 resource
        s3_client = boto3.resource('s3', region_name=aws_region)

        # Check if the bucket exists
        bucket = s3_client.Bucket(bucket_name)
        if not bucket.creation_date:
            raise FileNotFoundError(f"The specified bucket '{bucket_name}' does not exist.")

        # Upload the buffer directly to S3
        bucket.put_object(Key=checkpoint_path, Body=buffer)

        return checkpoint_path

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            raise FileNotFoundError(f"The specified bucket '{bucket_name}' does not exist.")
        else:
            raise e  # Re-raise other unexpected errors

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while saving the checkpoint: {e}")


def load_checkpoint_s3(epoch, bucket_name, mlflow_run_id, aws_region="eu-central-1"):
    """
    Load a PyTorch model's state_dict from a checkpoint stored in an S3 bucket.

    This function retrieves the specified checkpoint file from an S3 bucket,
    loads it into memory, and deserializes it into a PyTorch state_dict.
    It includes error handling for common issues such as missing files or
    network errors.

    Args:
        epoch (int): The epoch number of the checkpoint to load.
        bucket_name (str): The name of the S3 bucket containing the checkpoint.
        mlflow_run_id (str): The MLflow run ID associated with the checkpoint.
        aws_region (str): The AWS region where the S3 bucket is located. Defaults to "eu-central-1".

    Returns:
        dict: The PyTorch state_dict loaded from the checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist in the S3 bucket.
        botocore.exceptions.ClientError: If there is an error during the download process.

    Example:
        >>> state_dict = load_checkpoint_s3(epoch=1, bucket_name="my-bucket", mlflow_run_id="12345")
        >>> print(state_dict)
    """
    # Define the S3 path for the checkpoint
    checkpoint_path = f'checkpoints/run_{mlflow_run_id}/epoch_{epoch}.pt'

    # Initialize S3 resource
    s3_client = boto3.resource('s3', region_name=aws_region)

    # Create an in-memory bytes buffer
    buffer = io.BytesIO()

    try:
        # Download the checkpoint file from S3 into the buffer
        s3_client.Bucket(bucket_name).download_fileobj(checkpoint_path, buffer)

        # Reset the buffer's position to the beginning
        buffer.seek(0)

        # Load and return the state_dict from the buffer
        state_dict = torch.load(buffer)
        return state_dict

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} in bucket {bucket_name}")
        else:
            raise e  # Re-raise other unexpected errors

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading the checkpoint: {e}")
