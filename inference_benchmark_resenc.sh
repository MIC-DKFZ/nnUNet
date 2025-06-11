#!/bin/bash

# nnUNet Validation Benchmark Runner Script
# This script runs the Python benchmark with your exact configuration

# Set your paths and configuration here
NNUNET_BASE_PATH="/mnt/data/gpu-server/nnUNet_modified/nnunet_data"
DATASET_ID="001"
CONFIGURATION="3d_fullres"
FOLD=0
TRAINER="nnUNetTrainer"
PLANS="nnUNetResEncUNetMPlans"
CHECKPOINT="checkpoint_best.pth"
DEVICE=0
UV_COMMAND="uv run --extra cu124"

# Benchmark parameters
FULL_SET_RUNS=3              # Number of times to run the full validation set
INDIVIDUAL_MAX_IMAGES=10     # Number of individual images to benchmark in detail
INDIVIDUAL_RUNS_PER_IMAGE=3  # Number of runs per individual image

# Output settings
OUTPUT_DIR="${NNUNET_BASE_PATH}/nnUNet_results/Dataset001_PancreasSegClassification/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/benchmark_predictions"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${NNUNET_BASE_PATH}/nnUNet_results/Dataset001_PancreasSegClassification/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/validation_benchmark_${TIMESTAMP}.json"

echo "🎯 Starting nnUNet Validation Benchmark"
echo "========================================"
echo "Base Path: ${NNUNET_BASE_PATH}"
echo "Dataset: ${DATASET_ID}"
echo "Configuration: ${CONFIGURATION}"
echo "Model: ${TRAINER}__${PLANS}"
echo "Fold: ${FOLD}"
echo "Device: ${DEVICE}"
echo "========================================"

# Set environment variables
export nnUNet_raw="${NNUNET_BASE_PATH}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE_PATH}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_BASE_PATH}/nnUNet_results"

echo "Environment variables set:"
echo "  nnUNet_raw: ${nnUNet_raw}"
echo "  nnUNet_preprocessed: ${nnUNet_preprocessed}"
echo "  nnUNet_results: ${nnUNet_results}"
echo ""

# Check if benchmark script exists
BENCHMARK_SCRIPT="inference_benchmark.py"

if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo "❌ Error: Benchmark script '$BENCHMARK_SCRIPT' not found!"
    echo "Please ensure the Python benchmark script is in the current directory."
    exit 1
fi

echo "✅ Found benchmark script: $BENCHMARK_SCRIPT"
echo ""

# Run the benchmark
echo "🚀 Starting comprehensive benchmark..."
echo "This may take 15-30 minutes depending on your validation set size."
echo ""

uv run --extra cu124 python "$BENCHMARK_SCRIPT" \
    --nnunet_base_path "$NNUNET_BASE_PATH" \
    --dataset_id "$DATASET_ID" \
    --configuration "$CONFIGURATION" \
    --fold "$FOLD" \
    --trainer "$TRAINER" \
    --plans "$PLANS" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --uv_command "$UV_COMMAND" \
    --output_dir "$OUTPUT_DIR" \
    --full_set_runs "$FULL_SET_RUNS" \
    --individual_max_images "$INDIVIDUAL_MAX_IMAGES" \
    --individual_runs_per_image "$INDIVIDUAL_RUNS_PER_IMAGE" \
    --output_file "$OUTPUT_FILE"

# Check if benchmark completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Benchmark completed successfully!"
    echo "📊 Results saved to: $OUTPUT_FILE"
    echo "📄 Report saved to: ${OUTPUT_FILE%.json}.txt"
    echo ""
    echo "📈 Quick Summary:"
    echo "=================="

    # Extract key metrics from the JSON file if jq is available
    if command -v jq &> /dev/null; then
        echo "Full validation set timing:"
        jq -r '.benchmark_results.full_validation_set | "  Total images: \(.total_images)", "  Mean time per image: \(.per_image_time_mean | tostring | .[0:6])s", "  Total time: \(.total_time_mean | tostring | .[0:6])s"' "$OUTPUT_FILE" 2>/dev/null || echo "  (Install jq for detailed summary)"

        echo ""
        echo "Individual image analysis:"
        jq -r '.benchmark_results.individual_images.overall_individual_stats | "  Mean time: \(.mean_time | tostring | .[0:6])s", "  Median time: \(.median_time | tostring | .[0:6])s", "  Range: \(.min_time | tostring | .[0:6])s - \(.max_time | tostring | .[0:6])s"' "$OUTPUT_FILE" 2>/dev/null || echo "  (Install jq for detailed summary)"
    else
        echo "Install 'jq' for detailed JSON parsing, or check the .txt report file."
    fi

else
    echo ""
    echo "❌ Benchmark failed! Check the error messages above."
    echo "Common issues:"
    echo "- Check that all paths exist"
    echo "- Verify nnUNet installation"
    echo "- Ensure CUDA device is available"
    echo "- Check disk space for temporary files"
    exit 1
fi