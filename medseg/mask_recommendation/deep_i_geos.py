from medseg import utils
import os
from shutil import copyfile
import time
import subprocess
import signal
from evaluate import evaluate
from pathlib import Path
import shutil
import GeodisTK
import numpy as np


def convert(image_path, mask_path, save_path, geodisc_lambda):
    image_filenames = utils.load_filenames(image_path)
    mask_filenames = utils.load_filenames(mask_path)

    for i in range(len(image_filenames)):
        image, affine, spacing, header = utils.load_nifty(image_filenames[i])
        mask, _, _, _ = utils.load_nifty(mask_filenames[i])
        mask = mask.astype(np.uint8)
        mask[mask < 0] = 0
        geodesik_mask = GeodisTK.geodesic3d_raster_scan(image.astype(np.float32), mask, spacing.astype(np.float32), geodisc_lambda, 1)
        utils.save_nifty(save_path + os.path.basename(mask_filenames[i]), geodesik_mask, affine, spacing, header, is_mask=False)


def copy_masks_for_inference(mask_path, refinement_inference_tmp):
    mask_filenames = utils.load_filenames(mask_path)
    quarter = int(len(mask_filenames) / 4)
    filenames0 = mask_filenames[:quarter]
    filenames1 = mask_filenames[quarter:quarter*2]
    filenames2 = mask_filenames[quarter*2:quarter*3]
    filenames3 = mask_filenames[quarter*3:]
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


def compute_predictions(available_devices, save_path, image_path, prediction_path, gt_path, refined_prediction_save_path, refinement_inference_tmp, model, class_labels, geodisc_lambda):
    convert_save_path = "deep_i_geos_tmp/"
    Path(convert_save_path).mkdir(parents=True, exist_ok=True)
    convert(image_path, save_path, convert_save_path, geodisc_lambda)
    copy_masks_for_inference(convert_save_path, refinement_inference_tmp)
    shutil.rmtree(convert_save_path)
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
    return results