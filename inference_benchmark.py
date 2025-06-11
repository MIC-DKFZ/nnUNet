#!/usr/bin/env python3
"""
Comprehensive nnUNet Validation Set Benchmarking Script
Uses subprocess to call nnUNet CLI commands for reliable benchmarking
"""

import os
import time
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import psutil
import shutil
import tempfile


class nnUNetValidationBenchmark:
    """
    Benchmark nnUNet inference speed on validation dataset using CLI subprocess calls.
    """

    def __init__(self,
                 nnunet_base_path: str,
                 dataset_id: str = "001",
                 configuration: str = "3d_fullres",
                 fold: int = 0,
                 trainer: str = "nnUNetTrainer",
                 plans: str = "nnUNetResEncUNetMPlans",
                 checkpoint: str = "checkpoint_best.pth",
                 device: int = 0,
                 uv_command: str = "uv run --extra cu124"):
        """
        Initialize benchmark with your nnUNet configuration.

        Args:
            nnunet_base_path: Base path to nnUNet data directories
            dataset_id: Dataset ID (e.g., "001")
            configuration: Configuration name (e.g., "3d_fullres")
            fold: Fold number
            trainer: Trainer name
            plans: Plans name
            checkpoint: Checkpoint filename
            device: CUDA device number
            uv_command: UV command prefix for running nnUNet
        """
        self.nnunet_base_path = Path(nnunet_base_path)
        self.dataset_id = dataset_id
        self.configuration = configuration
        self.fold = fold
        self.trainer = trainer
        self.plans = plans
        self.checkpoint = checkpoint
        self.device = device
        self.uv_command = uv_command

        # Setup paths
        self.setup_paths()

        # Results storage
        self.results = {
            'config': {
                'dataset_id': dataset_id,
                'configuration': configuration,
                'fold': fold,
                'trainer': trainer,
                'plans': plans,
                'checkpoint': checkpoint,
                'device': device,
                'uv_command': uv_command
            },
            'system_info': self.get_system_info(),
            'benchmark_results': {}
        }

    def setup_paths(self):
        """Setup nnUNet paths and verify they exist."""
        self.nnunet_raw = self.nnunet_base_path / "nnUNet_raw"
        self.nnunet_preprocessed = self.nnunet_base_path / "nnUNet_preprocessed"
        self.nnunet_results = self.nnunet_base_path / "nnUNet_results"

        # Dataset specific paths
        dataset_name = f"Dataset{self.dataset_id}_PancreasSegClassification"
        self.dataset_path = self.nnunet_raw / dataset_name
        self.validation_images = self.dataset_path / "imagesTs"

        # Model path
        model_folder_name = f"{self.trainer}__{self.plans}__{self.configuration}"
        self.model_folder = self.nnunet_results / dataset_name / model_folder_name / f"fold_{self.fold}"

        # Verify paths exist
        required_paths = [self.nnunet_raw, self.nnunet_preprocessed, self.nnunet_results,
                         self.validation_images, self.model_folder]

        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required path does not exist: {path}")

        print(f"✅ All paths verified:")
        print(f"  Validation images: {self.validation_images}")
        print(f"  Model folder: {self.model_folder}")

    def get_system_info(self) -> Dict:
        """Collect system information for the benchmark report."""
        try:
            # Basic system info
            system_info = {
                'timestamp': datetime.now().isoformat(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'python_version': subprocess.check_output(['python', '--version']).decode().strip()
            }

            # GPU info using nvidia-smi
            try:
                gpu_output = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                    '--format=csv,noheader,nounits'
                ]).decode().strip()

                gpu_info = []
                for line in gpu_output.split('\n'):
                    if line.strip():
                        name, memory, driver = line.split(', ')
                        gpu_info.append({
                            'name': name.strip(),
                            'memory_total_mb': int(memory),
                            'driver_version': driver.strip()
                        })
                system_info['gpus'] = gpu_info
            except Exception as e:
                system_info['gpu_error'] = str(e)

            return system_info

        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def get_validation_images(self) -> List[Path]:
        """Get list of validation images to benchmark."""
        # Look for images with _0000.nii.gz suffix (nnUNet naming convention)
        image_files = list(self.validation_images.glob("*_0000.nii.gz"))

        if not image_files:
            # Fallback to any .nii.gz files
            image_files = list(self.validation_images.glob("*.nii.gz"))

        if not image_files:
            raise ValueError(f"No validation images found in {self.validation_images}")

        # Sort for consistent ordering
        image_files.sort()
        print(f"Found {len(image_files)} validation images")
        return image_files

    def setup_environment(self) -> Dict[str, str]:
        """Setup environment variables for nnUNet."""
        env = os.environ.copy()
        env.update({
            'nnUNet_raw': str(self.nnunet_raw),
            'nnUNet_preprocessed': str(self.nnunet_preprocessed),
            'nnUNet_results': str(self.nnunet_results),
            'CUDA_VISIBLE_DEVICES': str(self.device)
        })
        return env

    def benchmark_single_image(self, image_path: Path, num_runs: int = 5) -> Dict:
        """
        Benchmark inference on a single image by running nnUNet multiple times.
        """
        print(f"\n🔍 Benchmarking: {image_path.name}")

        # Create temporary directories for each run
        times = []
        successful_runs = 0

        for run_idx in range(num_runs):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input_dir = Path(temp_dir) / "input"
                temp_output_dir = Path(temp_dir) / "output"
                temp_input_dir.mkdir()
                temp_output_dir.mkdir()

                # Copy single image to temporary input directory
                temp_image_path = temp_input_dir / image_path.name
                shutil.copy2(image_path, temp_image_path)

                # Prepare nnUNet command
                cmd = [
                    *self.uv_command.split(),
                    'nnUNetv2_predict',
                    '-i', str(temp_input_dir),
                    '-o', str(temp_output_dir),
                    '-d', self.dataset_id,
                    '-c', self.configuration,
                    '-f', str(self.fold),
                    '-p', self.plans,
                    '-chk', self.checkpoint
                ]

                # Setup environment
                env = self.setup_environment()

                try:
                    # Time the inference
                    start_time = time.perf_counter()

                    result = subprocess.run(
                        cmd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout per run
                    )

                    end_time = time.perf_counter()

                    if result.returncode == 0:
                        elapsed_time = end_time - start_time
                        times.append(elapsed_time)
                        successful_runs += 1
                        print(f"  Run {run_idx + 1}/{num_runs}: {elapsed_time:.4f}s ✅")
                    else:
                        print(f"  Run {run_idx + 1}/{num_runs}: FAILED ❌")
                        print(f"    Error: {result.stderr}")

                except subprocess.TimeoutExpired:
                    print(f"  Run {run_idx + 1}/{num_runs}: TIMEOUT ⏰")
                except Exception as e:
                    print(f"  Run {run_idx + 1}/{num_runs}: ERROR - {e}")

        if not times:
            raise RuntimeError(f"All runs failed for {image_path.name}")

        # Calculate statistics
        times_array = np.array(times)
        stats = {
            'image_name': image_path.name,
            'successful_runs': successful_runs,
            'total_runs': num_runs,
            'mean_time': float(np.mean(times_array)),
            'std_time': float(np.std(times_array)),
            'min_time': float(np.min(times_array)),
            'max_time': float(np.max(times_array)),
            'median_time': float(np.median(times_array)),
            'all_times': times
        }

        print(f"  📊 Results: {stats['mean_time']:.4f}±{stats['std_time']:.4f}s (median: {stats['median_time']:.4f}s)")
        return stats

    def benchmark_full_validation_set(self, output_dir: str = None, num_runs: int = 1) -> Dict:
        """
        Benchmark inference on the full validation set.

        Args:
            output_dir: Directory to save predictions (optional)
            num_runs: Number of runs for timing (default 1 for full set)
        """
        if output_dir is None:
            output_dir = self.model_folder / "benchmark_predictions"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Running full validation set benchmark...")
        print(f"Output directory: {output_dir}")

        # Get all validation images
        validation_images = self.get_validation_images()

        # Prepare nnUNet command for full dataset
        cmd = [
            *self.uv_command.split(),
            'nnUNetv2_predict',
            '-i', str(self.validation_images),
            '-o', str(output_dir),
            '-d', self.dataset_id,
            '-c', self.configuration,
            '-f', str(self.fold),
            '-p', self.plans,
            '-chk', self.checkpoint,
            '--verbose'
        ]

        # Setup environment
        env = self.setup_environment()

        times = []

        for run_idx in range(num_runs):
            print(f"\n📏 Timing run {run_idx + 1}/{num_runs}...")

            # Clear output directory for clean run
            if run_idx > 0:
                shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True)

            try:
                start_time = time.perf_counter()

                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minute timeout
                )

                end_time = time.perf_counter()

                if result.returncode == 0:
                    elapsed_time = end_time - start_time
                    times.append(elapsed_time)
                    print(f"✅ Run {run_idx + 1} completed in {elapsed_time:.4f}s")
                else:
                    print(f"❌ Run {run_idx + 1} failed:")
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")

            except subprocess.TimeoutExpired:
                print(f"⏰ Run {run_idx + 1} timed out")
            except Exception as e:
                print(f"💥 Run {run_idx + 1} error: {e}")

        if not times:
            raise RuntimeError("All benchmark runs failed")

        # Calculate statistics
        times_array = np.array(times)
        per_image_times = times_array / len(validation_images)

        full_set_stats = {
            'total_images': len(validation_images),
            'successful_runs': len(times),
            'total_runs': num_runs,
            'total_time_mean': float(np.mean(times_array)),
            'total_time_std': float(np.std(times_array)),
            'total_time_min': float(np.min(times_array)),
            'total_time_max': float(np.max(times_array)),
            'per_image_time_mean': float(np.mean(per_image_times)),
            'per_image_time_std': float(np.std(per_image_times)),
            'all_total_times': times,
            'output_directory': str(output_dir)
        }

        return full_set_stats

    def benchmark_individual_images(self, max_images: int = 10, num_runs_per_image: int = 3) -> Dict:
        """
        Benchmark individual images for detailed analysis.

        Args:
            max_images: Maximum number of images to benchmark individually
            num_runs_per_image: Number of runs per image for statistics
        """
        print(f"\n🔬 Running individual image benchmarks...")

        validation_images = self.get_validation_images()

        # Limit number of images for individual benchmarking
        if len(validation_images) > max_images:
            validation_images = validation_images[:max_images]
            print(f"Limiting to first {max_images} images for individual benchmarking")

        individual_results = {}
        all_times = []

        for image_path in validation_images:
            try:
                stats = self.benchmark_single_image(image_path, num_runs_per_image)
                individual_results[image_path.name] = stats
                all_times.extend(stats['all_times'])
            except Exception as e:
                print(f"❌ Failed to benchmark {image_path.name}: {e}")
                continue

        if not all_times:
            raise RuntimeError("No individual image benchmarks completed successfully")

        # Overall statistics across all individual runs
        all_times_array = np.array(all_times)
        overall_stats = {
            'individual_results': individual_results,
            'overall_individual_stats': {
                'total_successful_runs': len(all_times),
                'mean_time': float(np.mean(all_times_array)),
                'std_time': float(np.std(all_times_array)),
                'min_time': float(np.min(all_times_array)),
                'max_time': float(np.max(all_times_array)),
                'median_time': float(np.median(all_times_array))
            }
        }

        return overall_stats

    def run_comprehensive_benchmark(self,
                                  full_set_runs: int = 3,
                                  individual_max_images: int = 10,
                                  individual_runs_per_image: int = 3,
                                  output_dir: str = None) -> Dict:
        """
        Run comprehensive benchmark including both full set and individual image timings.
        """
        print("=" * 80)
        print("🎯 COMPREHENSIVE nnUNet VALIDATION BENCHMARK")
        print("=" * 80)

        benchmark_start = time.time()

        # Store all results
        self.results['benchmark_results'] = {
            'benchmark_start_time': datetime.now().isoformat(),
        }

        try:
            # 1. Full validation set benchmark
            print("\n1️⃣ FULL VALIDATION SET BENCHMARK")
            print("-" * 50)
            full_set_results = self.benchmark_full_validation_set(
                output_dir=output_dir,
                num_runs=full_set_runs
            )
            self.results['benchmark_results']['full_validation_set'] = full_set_results

            # 2. Individual image benchmarks
            print("\n2️⃣ INDIVIDUAL IMAGE BENCHMARKS")
            print("-" * 50)
            individual_results = self.benchmark_individual_images(
                max_images=individual_max_images,
                num_runs_per_image=individual_runs_per_image
            )
            self.results['benchmark_results']['individual_images'] = individual_results

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            self.results['benchmark_results']['error'] = str(e)
            raise

        finally:
            benchmark_end = time.time()
            self.results['benchmark_results']['total_benchmark_time'] = benchmark_end - benchmark_start
            self.results['benchmark_results']['benchmark_end_time'] = datetime.now().isoformat()

        return self.results

    def generate_report(self, results: Dict = None) -> str:
        """Generate a comprehensive benchmark report."""
        if results is None:
            results = self.results

        report = []
        report.append("=" * 80)
        report.append("📊 nnUNet VALIDATION BENCHMARK REPORT")
        report.append("=" * 80)

        # Configuration
        report.append("\n🔧 CONFIGURATION")
        report.append("-" * 50)
        config = results['config']
        report.append(f"Dataset ID: {config['dataset_id']}")
        report.append(f"Configuration: {config['configuration']}")
        report.append(f"Fold: {config['fold']}")
        report.append(f"Trainer: {config['trainer']}")
        report.append(f"Plans: {config['plans']}")
        report.append(f"Checkpoint: {config['checkpoint']}")
        report.append(f"Device: {config['device']}")

        # System Information
        report.append("\n💻 SYSTEM INFORMATION")
        report.append("-" * 50)
        sys_info = results['system_info']
        report.append(f"Timestamp: {sys_info.get('timestamp', 'N/A')}")
        report.append(f"CPU Count: {sys_info.get('cpu_count', 'N/A')}")
        report.append(f"Memory: {sys_info.get('memory_total_gb', 'N/A')} GB")
        report.append(f"Python: {sys_info.get('python_version', 'N/A')}")

        if 'gpus' in sys_info:
            report.append("\n🎮 GPU Information:")
            for i, gpu in enumerate(sys_info['gpus']):
                report.append(f"  GPU {i}: {gpu['name']} ({gpu['memory_total_mb']} MB)")

        # Benchmark Results
        if 'benchmark_results' in results:
            bench_results = results['benchmark_results']

            # Full validation set results
            if 'full_validation_set' in bench_results:
                report.append("\n🚀 FULL VALIDATION SET RESULTS")
                report.append("-" * 50)
                full_results = bench_results['full_validation_set']
                report.append(f"Total Images: {full_results['total_images']}")
                report.append(f"Successful Runs: {full_results['successful_runs']}/{full_results['total_runs']}")
                report.append(f"Total Time (mean): {full_results['total_time_mean']:.4f} ± {full_results['total_time_std']:.4f} seconds")
                report.append(f"Per Image Time (mean): {full_results['per_image_time_mean']:.4f} ± {full_results['per_image_time_std']:.4f} seconds")
                report.append(f"Range: {full_results['total_time_min']:.4f}s - {full_results['total_time_max']:.4f}s")

                # Calculate throughput
                throughput = full_results['total_images'] / full_results['total_time_mean']
                report.append(f"Throughput: {throughput:.2f} images/second")

            # Individual image results
            if 'individual_images' in bench_results:
                report.append("\n🔬 INDIVIDUAL IMAGE ANALYSIS")
                report.append("-" * 50)
                individual = bench_results['individual_images']
                overall_stats = individual['overall_individual_stats']
                report.append(f"Total Individual Runs: {overall_stats['total_successful_runs']}")
                report.append(f"Mean Time: {overall_stats['mean_time']:.4f} ± {overall_stats['std_time']:.4f} seconds")
                report.append(f"Median Time: {overall_stats['median_time']:.4f} seconds")
                report.append(f"Range: {overall_stats['min_time']:.4f}s - {overall_stats['max_time']:.4f}s")

                # Top 5 fastest and slowest images
                individual_results = individual['individual_results']
                if individual_results:
                    sorted_by_time = sorted(
                        individual_results.items(),
                        key=lambda x: x[1]['mean_time']
                    )

                    report.append("\n⚡ Top 5 Fastest Images:")
                    for i, (name, stats) in enumerate(sorted_by_time[:5]):
                        report.append(f"  {i+1}. {name}: {stats['mean_time']:.4f}s")

                    report.append("\n🐌 Top 5 Slowest Images:")
                    for i, (name, stats) in enumerate(sorted_by_time[-5:]):
                        report.append(f"  {i+1}. {name}: {stats['mean_time']:.4f}s")

            # Total benchmark time
            if 'total_benchmark_time' in bench_results:
                total_time = bench_results['total_benchmark_time']
                report.append(f"\n⏱️ Total Benchmark Duration: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")

        report.append("\n" + "=" * 80)
        return "\n".join(report)

    def save_results(self, output_file: str = None, results: Dict = None):
        """Save benchmark results to JSON file."""
        if results is None:
            results = self.results

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.model_folder / f"validation_benchmark_{timestamp}.json"
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        # Also save the report as text
        report_file = output_file.with_suffix('.txt')
        with open(report_file, 'w') as f:
            f.write(self.generate_report(results))
        print(f"📄 Report saved to: {report_file}")

        return output_file


def main():
    """Main function to run the benchmark with command line arguments."""
    parser = argparse.ArgumentParser(description='nnUNet Validation Set Benchmark')

    parser.add_argument('--nnunet_base_path', type=str, required=True,
                       help='Base path to nnUNet data directories')
    parser.add_argument('--dataset_id', type=str, default='001',
                       help='Dataset ID (default: 001)')
    parser.add_argument('--configuration', type=str, default='3d_fullres',
                       help='Configuration name (default: 3d_fullres)')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number (default: 0)')
    parser.add_argument('--trainer', type=str, default='nnUNetTrainer',
                       help='Trainer name (default: nnUNetTrainer)')
    parser.add_argument('--plans', type=str, default='nnUNetResEncUNetMPlans',
                       help='Plans name (default: nnUNetResEncUNetMPlans)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pth',
                       help='Checkpoint filename (default: checkpoint_best.pth)')
    parser.add_argument('--device', type=int, default=0,
                       help='CUDA device number (default: 0)')
    parser.add_argument('--uv_command', type=str, default='uv run --extra cu124',
                       help='UV command prefix (default: "uv run --extra cu124")')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for predictions')
    parser.add_argument('--full_set_runs', type=int, default=3,
                       help='Number of full validation set runs (default: 3)')
    parser.add_argument('--individual_max_images', type=int, default=10,
                       help='Max images for individual benchmarking (default: 10)')
    parser.add_argument('--individual_runs_per_image', type=int, default=3,
                       help='Runs per individual image (default: 3)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for results (default: auto-generated)')

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = nnUNetValidationBenchmark(
        nnunet_base_path=args.nnunet_base_path,
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        fold=args.fold,
        trainer=args.trainer,
        plans=args.plans,
        checkpoint=args.checkpoint,
        device=args.device,
        uv_command=args.uv_command
    )

    try:
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark(
            full_set_runs=args.full_set_runs,
            individual_max_images=args.individual_max_images,
            individual_runs_per_image=args.individual_runs_per_image,
            output_dir=args.output_dir
        )

        # Save results and generate report
        output_file = benchmark.save_results(args.output_file, results)

        # Print report to console
        print("\n" + benchmark.generate_report(results))

        print(f"\n🎉 Benchmark completed successfully!")
        print(f"📁 Results saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())