from nnunetv2.inference.bottleneck_predictor import BottleneckEnsemblePredictor

def main():
    predictor = BottleneckEnsemblePredictor()
    predictor.initialize_from_trained_model_folder(
        model_folder,  # Path to your model folder
        use_folds=(0,1,2,3,4),  # Use all folds for ensemble
        checkpoint_name="checkpoint_final.pth"  # Will fall back to checkpoint_best.pth if not found
    )
    
    embeddings = predictor.predict_from_files(
        input_folder="path/to/input/images",
        output_folder="path/to/output/embeddings",
    )

if __name__ == "__main__":
    main()
