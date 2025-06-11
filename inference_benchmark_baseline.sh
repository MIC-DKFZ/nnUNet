#!/bin/bash

# nnUNet BASELINE Validation Benchmark Runner Script
# This script runs benchmarks with the standard/baseline nnUNet configuration
# for comparison against your custom ResEncUNet model

# Set your paths and BASELINE configuration here
NNUNET_BASE_PATH="/mnt/data/gpu-server/nnUNet_modified/nnunet_data"
DATASET_ID="001"
CONFIGURATION="3d_fullres"
FOLD=0
TRAINER="nnUNetTrainer"              # Standard baseline trainer
PLANS="nnUNetPlans"                  # Standard baseline plans (NOT ResEncUNetMPlans)
CHECKPOINT="checkpoint_best.pth"
DEVICE=0
UV_COMMAND="uv run --extra cu124"

# Benchmark parameters
FULL_SET_RUNS=3              # Number of times to run the full validation set
INDIVIDUAL_MAX_IMAGES=10     # Number of individual images to benchmark in detail
INDIVIDUAL_RUNS_PER_IMAGE=3  # Number of runs per individual image

# Output settings for BASELINE
OUTPUT_DIR="${NNUNET_BASE_PATH}/nnUNet_results/Dataset001_PancreasSegClassification/nnUNetTrainer__nnUNetPlans__3d_fullres/baseline_benchmark_predictions"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${NNUNET_BASE_PATH}/nnUNet_results/Dataset001_PancreasSegClassification/nnUNetTrainer__nnUNetPlans__3d_fullres/baseline_benchmark_${TIMESTAMP}.json"

echo "🎯 Starting nnUNet BASELINE Validation Benchmark"
echo "================================================="
echo "Base Path: ${NNUNET_BASE_PATH}"
echo "Dataset: ${DATASET_ID}"
echo "Configuration: ${CONFIGURATION}"
echo "Model: ${TRAINER}__${PLANS} (BASELINE)"
echo "Fold: ${FOLD}"
echo "Device: ${DEVICE}"
echo ""
echo "⚠️  NOTE: This is the BASELINE benchmark using standard nnUNet"
echo "   Compare results against your ResEncUNet model to measure improvements"
echo "================================================="

# Set environment variables
export nnUNet_raw="${NNUNET_BASE_PATH}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE_PATH}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_BASE_PATH}/nnUNet_results"

echo "Environment variables set:"
echo "  nnUNet_raw: ${nnUNet_raw}"
echo "  nnUNet_preprocessed: ${nnUNet_preprocessed}"
echo "  nnUNet_results: ${nnUNet_results}"
echo ""

# Check if baseline model exists
BASELINE_MODEL_DIR="${NNUNET_BASE_PATH}/nnUNet_results/Dataset001_PancreasSegClassification/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_${FOLD}"
if [ ! -d "$BASELINE_MODEL_DIR" ]; then
    echo "❌ Error: Baseline model directory not found!"
    echo "   Expected: ${BASELINE_MODEL_DIR}"
    echo ""
    echo "🔧 To create baseline model, you need to train with standard nnUNet:"
    echo "   nnUNetv2_train 001 3d_fullres 0 --npz"
    echo ""
    echo "💡 Or if you want to benchmark against existing baseline results,"
    echo "   ensure the standard nnUNet model is trained and available."
    exit 1
fi

# Check if checkpoint exists
CHECKPOINT_PATH="${BASELINE_MODEL_DIR}/${CHECKPOINT}"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Baseline checkpoint not found!"
    echo "   Expected: ${CHECKPOINT_PATH}"
    echo ""
    echo "Available checkpoints in baseline model directory:"
    ls -la "${BASELINE_MODEL_DIR}"/*.pth 2>/dev/null || echo "   No .pth files found"
    exit 1
fi

echo "✅ Found baseline model: $BASELINE_MODEL_DIR"
echo "✅ Found baseline checkpoint: $CHECKPOINT_PATH"
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

# Run the baseline benchmark
echo "🚀 Starting BASELINE comprehensive benchmark..."
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
    echo "🎉 BASELINE Benchmark completed successfully!"
    echo "📊 Results saved to: $OUTPUT_FILE"
    echo "📄 Report saved to: ${OUTPUT_FILE%.json}.txt"
    echo ""
    echo "📈 BASELINE Quick Summary:"
    echo "=========================="

    # Extract key metrics from the JSON file if jq is available
    if command -v jq &> /dev/null; then
        echo "Baseline validation set timing:"
        jq -r '.benchmark_results.full_validation_set | "  Total images: \(.total_images)", "  Mean time per image: \(.per_image_time_mean | tostring | .[0:6])s", "  Total time: \(.total_time_mean | tostring | .[0:6])s"' "$OUTPUT_FILE" 2>/dev/null || echo "  (Install jq for detailed summary)"

        echo ""
        echo "Baseline individual image analysis:"
        jq -r '.benchmark_results.individual_images.overall_individual_stats | "  Mean time: \(.mean_time | tostring | .[0:6])s", "  Median time: \(.median_time | tostring | .[0:6])s", "  Range: \(.min_time | tostring | .[0:6])s - \(.max_time | tostring | .[0:6])s"' "$OUTPUT_FILE" 2>/dev/null || echo "  (Install jq for detailed summary)"
    else
        echo "Install 'jq' for detailed JSON parsing, or check the .txt report file."
    fi

    echo ""
    echo "🔄 Next Steps for Comparison:"
    echo "============================"
    echo "1. Run your ResEncUNet benchmark using the other script"
    echo "2. Compare the per_image_time_mean values"
    echo "3. Calculate improvement percentage:"
    echo "   Improvement% = (baseline_time - custom_time) / baseline_time * 100"
    echo ""
    echo "📊 For your Master's project, you need >10% improvement:"
    echo "   If baseline = 2.0s and custom = 1.8s, then improvement = 10%"
    echo ""
    echo "💾 Save this baseline result for comparison with optimized models!"

else
    echo ""
    echo "❌ BASELINE Benchmark failed! Check the error messages above."
    echo "Common issues:"
    echo "- Baseline model not trained (need to run standard nnUNet training)"
    echo "- Check that all paths exist"
    echo "- Verify nnUNet installation"
    echo "- Ensure CUDA device is available"
    echo "- Check disk space for temporary files"
    echo ""
    echo "🔧 To train baseline model if needed:"
    echo "   nnUNetv2_train 001 3d_fullres 0 --npz"
    exit 1
fi

echo ""
echo "📝 IMPORTANT NOTES:"
echo "=================="
echo "• This benchmark used STANDARD nnUNet (nnUNetPlans)"
echo "• Compare against your ResEncUNet model results"
echo "• For Master's requirement: need 10% speed improvement"
echo "• Use this baseline to validate your optimizations"
echo ""
echo "🔗 Useful comparison script:"
echo "python3 -c \""
echo "import json"
echo "baseline = json.load(open('$OUTPUT_FILE'))"
echo "# Load your custom model results and compare"
echo "baseline_time = baseline['benchmark_results']['full_validation_set']['per_image_time_mean']"
echo "print(f'Baseline per-image time: {baseline_time:.4f}s')"
echo "\""