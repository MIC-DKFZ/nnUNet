import subprocess

def predict_nnunet(dataset, input_folder, output_folder_base, folds, trainer, unet_config, planner):
    """
    Automates the prediction process for nnUNet by looping through specified folds.

    Parameters:
    - dataset: str
      The dataset name (e.g., 'Dataset350_LymphNodes').
    - input_folder: str
      The path to the folder containing test images.
    - output_folder_base: str
      The base path for the output folder. It should end with a directory prefix like '.../fold_'.
    - folds: list of int
      The list of fold numbers to process (e.g., [1, 2, 3, 4]).
    - trainer: str
      The trainer to use (e.g., 'nnUNetTrainer_100epochs').
    - unet_config: str
      The nnUNet configuration (e.g., '3d_fullres').
    - planner: str
      The planner to use (e.g., 'nnUNetResEncUNetLPlans').
    """
    for fold in folds:
        print(f"Starting prediction for fold {fold}...")
        
        # Construct the output folder path by appending the fold number
        output_folder = f"{output_folder_base}{fold}"
        
        # Construct the prediction command
        command = [
            "nnUNetv2_predict",
            "-d", dataset,
            "-i", input_folder,
            "-o", output_folder,
            "-f", str(fold),
            "-tr", trainer,
            "-c", unet_config,
            "-p", planner
        ]
      
        try:
            # Execute the command
            subprocess.run(command, check=True)
            print(f"Prediction for fold {fold} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Prediction for fold {fold} failed with error: {e}")
            break  # Exit loop if a fold fails


def postprocessing_nnunet(input_folder_base, output_folder_base, folds, postprocessing_pkl, num_processes, plans_json):

    for fold in folds:
        print(f"Starting postprocessing for fold {fold}...")
        
        # Construct the input and output folder paths by appending the fold number
        input_folder = f"{input_folder_base}{fold}"
        output_folder = f"{output_folder_base}{fold}"
        
        # Construct the postprocessing command
        command = [
            "nnUNetv2_apply_postprocessing",
            "-i", input_folder,
            "-o", output_folder,
            "-pp_pkl_file", postprocessing_pkl,
            "-np", str(num_processes),
            "-plans_json", plans_json
        ]
      
        try:
            # Execute the command
            subprocess.run(command, check=True)
            print(f"Postprocessing for fold {fold} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Postprocessing for fold {fold} failed with error: {e}")
            break


if __name__ == "__main__":
    # Define parameters for the prediction process
    dataset = "Dataset350_LymphNodes"
    input_inference = r"C:\Users\Test\Desktop\Bart\nnUNet\nnUNet_raw\Dataset350_LymphNodes\imagesTs"
    output_inference_base = r"C:\Users\Test\Desktop\Bart\nnUNet\nnUNet_results\Dataset350_LymphNodes\nnUNetTrainer_100epochs__nnUNetResEncUNetLPlans__3d_fullres\Inference_testresults\fold_"
    output_postprocessing_base = r"C:\Users\Test\Desktop\Bart\nnUNet\nnUNet_results\Dataset350_LymphNodes\nnUNetTrainer_100epochs__nnUNetResEncUNetLPlans__3d_fullres\Inference_postprocessing\fold_"
    folds_i = [1, 2, 3, 4]  # Folds for which predictions will be performed
    folds_pp = [0, 1, 2, 3, 4]  # Folds for which postprocessing will be performed
    trainer = "nnUNetTrainer_100epochs"
    unet_config = "3d_fullres"
    planner = "nnUNetResEncUNetLPlans"
    postprocessing_pkl = r"C:\Users\Test\Desktop\Bart\nnUNet\nnUNet_results\Dataset350_LymphNodes\nnUNetTrainer_100epochs__nnUNetResEncUNetLPlans__3d_fullres\crossval_results_folds_0_1_2_3_4\postprocessing.pkl" 
    num_processes = 8
    plans_json = r"C:\Users\Test\Desktop\Bart\nnUNet\nnUNet_results\Dataset350_LymphNodes\nnUNetTrainer_100epochs__nnUNetResEncUNetLPlans__3d_fullres\crossval_results_folds_0_1_2_3_4\plans.json"


    predict_nnunet(dataset, input_inference, output_inference_base, folds_i, trainer, unet_config, planner)
    postprocessing_nnunet(output_inference_base, output_postprocessing_base, folds_pp, postprocessing_pkl, num_processes, plans_json)
