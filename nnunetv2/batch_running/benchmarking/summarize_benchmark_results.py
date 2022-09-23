from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.file_path_utilities import get_output_folder

if __name__ == '__main__':
    trainers = ['nnUNetTrainerBenchmark_5epochs', 'nnUNetTrainerBenchmark_5epochs_noDataLoading']
    datasets = [2, 3, 4, 5]
    plans = ['nnUNetPlans']
    configs = ['2d', '3d_fullres']
    output_file = join(nnUNet_results, 'benchmark_results.csv')

    torch_version = "1.12.0a0+git664058f"  #"1.11.0a0+gitbc2c6ed"  #
    cudnn_version = 8500  # 8302  #
    num_gpus = 1

    unique_gpus = set()

    # collect results in the most janky way possible. Amazing coding skills!
    all_results = {}
    for tr in trainers:
        all_results[tr] = {}
        for p in plans:
            all_results[tr][p] = {}
            for c in configs:
                all_results[tr][p][c] = {}
                for d in datasets:
                    dataset_name = maybe_convert_to_dataset_name(d)
                    output_folder = get_output_folder(dataset_name, tr, p, c, fold=0)
                    expected_benchmark_file = join(output_folder, 'benchmark_result.json')
                    all_results[tr][p][c][d] = {}
                    if isfile(expected_benchmark_file):
                        # filter results for what we want
                        results = [i for i in load_json(expected_benchmark_file).values()
                                   if i['num_gpus'] == num_gpus and i['cudnn_version'] == cudnn_version and
                                   i['torch_version'] == torch_version]
                        for r in results:
                            all_results[tr][p][c][d][r['gpu_name']] = r
                            unique_gpus.add(r['gpu_name'])

    # haha. Fuck this. Collect GPUs in the code above.
    # unique_gpus = np.unique([i["gpu_name"] for tr in trainers for p in plans for c in configs for d in datasets for i in all_results[tr][p][c][d]])

    unique_gpus = list(unique_gpus)
    unique_gpus.sort()

    with open(output_file, 'w') as f:
        f.write('Dataset,Trainer,Plans,Config')
        for g in unique_gpus:
            f.write(f",{g}")
        f.write("\n")
        for d in datasets:
            for tr in trainers:
                for p in plans:
                    for c in configs:
                        gpu_results = []
                        for g in unique_gpus:
                            if g in all_results[tr][p][c][d].keys():
                                gpu_results.append(round(all_results[tr][p][c][d][g]["fastest_epoch"], ndigits=2))
                            else:
                                gpu_results.append("MISSING")
                        # skip if all are missing
                        if all([i == 'MISSING' for i in gpu_results]):
                            continue
                        f.write(f"{d},{tr},{p},{c}")
                        for g in gpu_results:
                            f.write(f",{g}")
                        f.write("\n")
            f.write("\n")

