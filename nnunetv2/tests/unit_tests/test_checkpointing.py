import unittest
import boto3
from botocore.exceptions import ClientError
import io
import os
import sys
import torch

# Import the function to be tested
from nnunetv2.utilities.checkpointing import save_checkpoint_s3, load_checkpoint_s3


class TestSaveCheckpointS3RealConnection(unittest.TestCase):
    def setUp(self):
        """Set up an S3 bucket, deleting it first if it already exists."""
        self.s3 = boto3.resource("s3", region_name=os.environ.get("AWS_REGION"))
        self.bucket_name = "checkpointing-test-bucket"

        # Check if the bucket exists and delete it if necessary
        bucket = self.s3.Bucket(self.bucket_name)
        try:
            if bucket.creation_date:
                # Delete all objects in the bucket
                bucket.objects.all().delete()
                # Delete the bucket itself
                bucket.delete()
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code != "404":
                print(f"Unexpected error when checking or deleting bucket: {e}")
                raise

        # Create a new bucket
        self.s3.create_bucket(
            Bucket=self.bucket_name,
            CreateBucketConfiguration={
                "LocationConstraint": os.environ.get("AWS_REGION")
            }
        )

    def tearDown(self):
        """Clean up the S3 bucket after tests."""
        bucket = self.s3.Bucket(self.bucket_name)
        try:
            # Delete all objects in the bucket
            bucket.objects.all().delete()
            # Delete the bucket itself
            bucket.delete()
        except ClientError as e:
            print(f"Error during teardown: {e}")
            raise

    def test_save_and_load_checkpoint_s3(self):
        # Mock inputs
        state_dict = {"layer1.weight": torch.randn(10, 10)}  # Example state_dict
        epoch = 37
        mlflow_run_id = "12345"

        # Call the function
        checkpoint_path = save_checkpoint_s3(state_dict, epoch, self.bucket_name, mlflow_run_id, os.getenv("AWS_REGION"))

        # Verify the result
        expected_path = f"checkpoints/run_{mlflow_run_id}/epoch_{epoch}.pt"
        self.assertTrue(expected_path in checkpoint_path)

        # Verify that the object exists in the real S3 bucket
        obj = self.s3.Object(self.bucket_name, f"checkpoints/run_{mlflow_run_id}/epoch_{epoch}.pt")
        obj_body = obj.get()["Body"].read()

        # Verify that the object content matches the serialized state_dict
        buffer = io.BytesIO(obj_body)
        loaded_state_dict = torch.load(buffer)
        self.assertEqual(state_dict.keys(), loaded_state_dict.keys())
        for key in state_dict:
            self.assertTrue(torch.equal(state_dict[key], loaded_state_dict[key]))

        # Load checkpoint from S3
        loaded_state_dict = load_checkpoint_s3(epoch, self.bucket_name, mlflow_run_id)

        # Verify that the loaded state_dict matches the original one
        self.assertEqual(state_dict.keys(), loaded_state_dict.keys())
        for key in state_dict:
            self.assertTrue(torch.equal(state_dict[key], loaded_state_dict[key]))


if __name__ == "__main__":
    unittest.main()
