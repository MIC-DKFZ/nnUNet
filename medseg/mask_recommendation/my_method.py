from medseg import utils
import os
from shutil import copyfile
import time
import subprocess
import signal
from evaluate import evaluate


def copy_masks_for_inference(load_dir, refinement_inference_tmp):
    filenames = utils.load_filenames(load_dir)
    quarter = int(len(filenames) / 4)
    filenames0 = filenames[:quarter]
    filenames1 = filenames[quarter:quarter*2]
    filenames2 = filenames[quarter*2:quarter*3]
    filenames3 = filenames[quarter*3:]
    save_dir0 = refinement_inference_tmp + "0/"
    save_dir1 = refinement_inference_tmp + "1/"
    save_dir2 = refinement_inference_tmp + "2/"
    save_dir3 = refinement_inference_tmp + "3/"

    for filename in filenames0:
        copyfile(filename, save_dir0 + os.path.basename(filename))
    for filename in filenames1:
        copyfile(filename, save_dir1 + os.path.basename(filename))
    for filename in filenames2:
        copyfile(filename, save_dir2 + os.path.basename(filename))
    for filename in filenames3:
        copyfile(filename, save_dir3 + os.path.basename(filename))


def compute_predictions(available_devices, save_path, prediction_path, gt_path, refined_prediction_save_path, refinement_inference_tmp, model, class_labels):
    copy_masks_for_inference(save_path, refinement_inference_tmp)
    start_time = time.time()
    filenames = utils.load_filenames(refined_prediction_save_path, extensions=None)
    print("load_filenames: ", time.time() - start_time)
    start_time = time.time()
    for filename in filenames:
        os.remove(filename)
    parts_to_process = [0, 1, 2, 3]
    waiting = []
    finished = []
    wait_time = 5
    start_inference_time = time.time()

    print("remove: ", time.time() - start_time)
    print("Starting inference...")
    while parts_to_process:
        if available_devices:
            device = available_devices[0]
            available_devices = available_devices[1:]
            part = parts_to_process[0]
            parts_to_process = parts_to_process[1:]
            print("Processing part {} on device {}...".format(part, device))
            command = 'nnUNet_predict -i ' + str(refinement_inference_tmp) + str(
                part) + ' -o ' + str(refined_prediction_save_path) + ' -tr nnUNetTrainerV2Guided3 -t ' + model + ' -m 3d_fullres -f 0 -d ' + str(
                device) + ' -chk model_best --disable_tta --num_threads_preprocessing 1 --num_threads_nifti_save 1'
            p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, preexec_fn=os.setsid)
            waiting.append([part, device, p, time.time()])
        else:
            for w in waiting:
                if w[2].poll() is not None:
                    print("Finished part {} on device {} after {}s.".format(w[0], w[1], time.time() - w[3]))
                    available_devices.append(w[1])
                    finished.append(w[0])
                    waiting.remove(w)
                    break
            time.sleep(wait_time)
    print("All parts are being processed.")

    def check_all_predictions_exist():
        filenames = utils.load_filenames(refined_prediction_save_path)
        nr_predictions = len(utils.load_filenames(prediction_path))
        counter = 0
        for filename in filenames:
            if ".nii.gz" in filename:
                counter += 1
        return bool(counter == nr_predictions)

    while waiting and len(finished) < 4 and not check_all_predictions_exist():
        time.sleep(wait_time)
    print("All predictions finished.")
    time.sleep(30)
    print("Cleaning up threads")
    # [os.killpg(os.getpgid(p.pid), signal.SIGTERM) for p in finished]
    [os.killpg(os.getpgid(p[2].pid), signal.SIGTERM) for p in waiting]
    os.remove(refined_prediction_save_path + "/plans.pkl")
    print("Total inference time {}s.".format(time.time() - start_inference_time))
    print("All parts finished processing.")
    results = evaluate(gt_path, refined_prediction_save_path, class_labels)
    print("Evaluation finished.")
    return results