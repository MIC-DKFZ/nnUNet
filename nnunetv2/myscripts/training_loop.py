import subprocess

def train_nnunet(dataset_id, unet_config, folds):
    """
    Automates the training process for nnU-Net by looping through folds.

    Parameters:
    - dataset_id: int or str
      The ID or name of the dataset to train on.
    - unet_config: str
      The configuration of the nnU-Net (e.g., '3d_fullres').
    - folds: list of int
      The folds to train sequentially.

    Returns:
    - None
    """
    for fold in folds:
        print(f"Starting training for fold {fold}...")
        
        # Construct the training command
        command = [
            "nnUNetv2_train",
            str(dataset_id),
            unet_config,
            str(fold),
            "-device",
            "cuda"
        ]
        
        try:
            # Execute the command
            subprocess.run(command, check=True)
            print(f"Training for fold {fold} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Training for fold {fold} failed with error: {e}")
            break  # Exit loop if a fold fails to train

# Example usage
if __name__ == "__main__":
    dataset_id = 250
    unet_config = "3d_fullres"
    folds = [0, 1, 2, 3, 4]

    train_nnunet(dataset_id, unet_config, folds)
