from i3Deep import utils
import os
from shutil import copyfile
import time
import subprocess
import signal
from evaluate import evaluate
import shutil
from i3Deep.comp_uncertainties import comp_uncertainties, comp_bhattacharyya_uncertainty, comp_entropy_uncertainty, comp_variance_uncertainty
from nnunet.inference.my_predict_simple import inference
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor as Pool


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


# def compute_predictions(available_devices, save_path, prediction_path, gt_path, refined_prediction_save_path, refinement_inference_tmp, model, class_labels):
#     copy_masks_for_inference(save_path, refinement_inference_tmp)
#     start_time = time.time()
#     filenames = utils.load_filenames(refined_prediction_save_path, extensions=None)
#     print("load_filenames: ", time.time() - start_time)
#     start_time = time.time()
#     for filename in filenames:
#         os.remove(filename)
#     parts_to_process = [0, 1, 2, 3]
#     waiting = []
#     finished = []
#     wait_time = 5
#     start_inference_time = time.time()
#
#     print("remove: ", time.time() - start_time)
#     print("Starting inference...")
#     while parts_to_process:
#         if available_devices:
#             device = available_devices[0]
#             available_devices = available_devices[1:]
#             part = parts_to_process[0]
#             parts_to_process = parts_to_process[1:]
#             print("Processing part {} on device {}...".format(part, device))
#             command = 'nnUNet_predict -i ' + str(refinement_inference_tmp) + str(
#                 part) + ' -o ' + str(refined_prediction_save_path) + ' -tr nnUNetTrainerV2Guided3 -t ' + model + ' -m 3d_fullres -f 0 -d ' + str(
#                 device) + ' -chk model_best --disable_tta --num_threads_preprocessing 1 --num_threads_nifti_save 1'
#             p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, preexec_fn=os.setsid)
#             waiting.append([part, device, p, time.time()])
#         else:
#             for w in waiting:
#                 if w[2].poll() is not None:
#                     print("Finished part {} on device {} after {}s.".format(w[0], w[1], time.time() - w[3]))
#                     available_devices.append(w[1])
#                     finished.append(w[0])
#                     waiting.remove(w)
#                     break
#             time.sleep(wait_time)
#     print("All parts are being processed.")
#
#     def check_all_predictions_exist():
#         filenames = utils.load_filenames(refined_prediction_save_path)
#         nr_predictions = len(utils.load_filenames(prediction_path))
#         counter = 0
#         for filename in filenames:
#             if ".nii.gz" in filename:
#                 counter += 1
#         return bool(counter == nr_predictions)
#
#     while waiting and len(finished) < 4 and not check_all_predictions_exist():
#         time.sleep(wait_time)
#     print("All predictions finished.")
#     time.sleep(30)
#     print("Cleaning up threads")
#     # [os.killpg(os.getpgid(p.pid), signal.SIGTERM) for p in finished]
#     [os.killpg(os.getpgid(p[2].pid), signal.SIGTERM) for p in waiting]
#     os.remove(refined_prediction_save_path + "/plans.pkl")
#     print("Total inference time {}s.".format(time.time() - start_inference_time))
#     print("All parts finished processing.")
#     results = evaluate(gt_path, refined_prediction_save_path, class_labels)
#     print("Evaluation finished.")
#     return results


# def comp_uncertainty_and_prediction(available_devices, refinement_inference_tmp, uncertainty_path, prediction_path, model, class_labels, refined_prediction_save_path):
#     start_time = time.time()
#     print("load_filenames: ", time.time() - start_time)
#     start_time = time.time()
#
#     print("remove: ", time.time() - start_time)
#     print("Starting inference...")
#     for i in range(5):
#         print("Fold: ", i)
#         uncertainty_path_tmp = uncertainty_path + "fold_" + str(i)
#         filenames = utils.load_filenames(uncertainty_path_tmp, extensions=None)
#         for filename in filenames:
#             os.remove(filename)
#         parts_to_process = [0, 1, 2, 3]
#         waiting = []
#         finished = []
#         wait_time = 5
#         start_inference_time = time.time()
#         while parts_to_process:
#             if available_devices:
#                 device = available_devices[0]
#                 available_devices = available_devices[1:]
#                 part = parts_to_process[0]
#                 parts_to_process = parts_to_process[1:]
#                 print("Uncertainty computation: Processing part {} on device {}...".format(part, device))
#                 command = 'nnUNet_predict -i ' + str(refinement_inference_tmp) + str(
#                     part) + ' -o ' + str(uncertainty_path_tmp) + ' -tr nnUNetTrainerV2Guided3 -t ' + model + ' -m 3d_fullres -f ' + str(i) + ' -d ' + str(
#                     device) + ' -chk model_best --disable_tta --num_threads_preprocessing 1 --num_threads_nifti_save 1  --output_probabilities'
#                 p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, preexec_fn=os.setsid)
#                 waiting.append([part, device, p, time.time()])
#             else:
#                 for w in waiting:
#                     if w[2].poll() is not None:
#                         print("Finished part {} on device {} after {}s.".format(w[0], w[1], time.time() - w[3]))
#                         available_devices.append(w[1])
#                         finished.append(w[0])
#                         waiting.remove(w)
#                         break
#                 time.sleep(wait_time)
#         print("All parts are being processed.")
#
#         def check_all_predictions_exist():
#             filenames = utils.load_filenames(uncertainty_path_tmp)
#             nr_predictions = len(utils.load_filenames(prediction_path)) * len(class_labels)
#             counter = 0
#             for filename in filenames:
#                 if ".nii.gz" in filename:
#                     counter += 1
#             return bool(counter == nr_predictions)
#
#         while waiting and len(finished) < 4 and not check_all_predictions_exist():
#             time.sleep(wait_time)
#         print("All predictions finished.")
#         time.sleep(30)
#         print("Cleaning up threads")
#         # [os.killpg(os.getpgid(p.pid), signal.SIGTERM) for p in finished]
#         [os.killpg(os.getpgid(p[2].pid), signal.SIGTERM) for p in waiting]
#         os.remove(uncertainty_path_tmp + "/plans.pkl")
#         print("Total inference time {}s.".format(time.time() - start_inference_time))
#         print("All parts finished processing.")
#         for filename in utils.load_filenames(uncertainty_path_tmp):
#             shutil.move(filename, uncertainty_path + "probabilities/")
#
#     comp_uncertainties(uncertainty_path + "probabilities/", uncertainty_path + "uncertainties/", comp_bhattacharyya_uncertainty, None, None, None, None, 4, None, merge_uncertainties=True, class_dices=False, prediction_path=refined_prediction_save_path)
#
#     print("Uncertainty computation finished.")

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class MyPool(mp.pool.Pool):
    Process = NoDaemonProcess


def comp_uncertainty_and_prediction(available_devices, recommended_masks_path, refinement_inference, uncertainty_path, prediction_path, model, class_labels, refined_prediction_save_path):
    for filename in utils.load_filenames(recommended_masks_path):
        shutil.move(filename, refinement_inference + os.path.basename(filename))

    print("Starting inference...")
    fold_and_device = [[0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
    # inference(fold_and_device[0], input_folder=refinement_inference, output_folder=uncertainty_path, trainer_class_name="nnUNetTrainerV2Guided3", output_probabilities=True, task_name=model)
    # pool = Pool(processes=5)
    pool = Pool(max_workers=5)
    pool.map(partial(inference, input_folder=refinement_inference, output_folder=uncertainty_path, trainer_class_name="nnUNetTrainerV2Guided3", output_probabilities=True, task_name=model), fold_and_device)
    # pool.close()
    # pool.join()
    for fold in range(5):
        uncertainty_path_tmp = uncertainty_path + "fold_" + str(fold)
        os.remove(uncertainty_path_tmp + "/plans.pkl")
        for filename in utils.load_filenames(uncertainty_path_tmp):
            shutil.move(filename, uncertainty_path + "probabilities/" + os.path.basename(filename))
    print("Inference finished.")

    # folds_to_process = [0, 1, 2, 3, 4]
    # waiting = []
    # finished = []
    # wait_time = 5
    # start_inference_time = time.time()
    # while folds_to_process:
    #     if available_devices:
    #         device = available_devices[0]
    #         available_devices = available_devices[1:]
    #         fold = folds_to_process[0]
    #         folds_to_process = folds_to_process[1:]
    #         uncertainty_path_tmp = uncertainty_path + "fold_" + str(fold)
    #         filenames = utils.load_filenames(uncertainty_path_tmp, extensions=None)
    #         for filename in filenames:
    #             os.remove(filename)
    #         print("Uncertainty computation: Processing fold {} on device {}...".format(fold, device))
    #         print("refinement_inference: ", refinement_inference)
    #         print("uncertainty_path_tmp: ", uncertainty_path_tmp)
    #         command = 'nnUNet_predict -i ' + str(refinement_inference) + str(
    #             fold) + ' -o ' + str(uncertainty_path_tmp) + ' -tr nnUNetTrainerV2Guided3 -t ' + model + ' -m 3d_fullres -f ' + str(fold) + ' -d ' + str(
    #             device) + ' -chk model_best --disable_tta --output_probabilities'  # --num_threads_preprocessing 1 --num_threads_nifti_save 1
    #         p = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, preexec_fn=os.setsid)
    #         waiting.append([fold, device, p, time.time()])
    #     else:
    #         for w in waiting:
    #             if w[2].poll() is not None:
    #                 print("Finished fold {} on device {} after {}s.".format(w[0], w[1], time.time() - w[3]))
    #                 available_devices.append(w[1])
    #                 finished.append(w[0])
    #                 waiting.remove(w)
    #                 break
    #         time.sleep(wait_time)
    # print("All parts are being processed.")
    #
    # def check_all_predictions_exist():
    #     filenames = utils.load_filenames(uncertainty_path_tmp)
    #     nr_predictions = len(utils.load_filenames(prediction_path)) * len(class_labels)
    #     counter = 0
    #     for filename in filenames:
    #         if ".nii.gz" in filename:
    #             counter += 1
    #     return bool(counter == nr_predictions)
    #
    # while waiting and len(finished) < 5 and not check_all_predictions_exist():
    #     time.sleep(wait_time)
    # print("All predictions finished.")
    # time.sleep(15)
    # print("Cleaning up threads")
    # # [os.killpg(os.getpgid(p.pid), signal.SIGTERM) for p in finished]
    # [os.killpg(os.getpgid(p[2].pid), signal.SIGTERM) for p in waiting]
    # os.remove(uncertainty_path_tmp + "/plans.pkl")
    # print("Total inference time {}s.".format(time.time() - start_inference_time))
    # print("All parts finished processing.")
    # for filename in utils.load_filenames(uncertainty_path_tmp):
    #     shutil.move(filename, uncertainty_path + "probabilities/" + os.path.basename(filename))

    comp_uncertainties(uncertainty_path + "probabilities/", uncertainty_path + "uncertainties/", comp_bhattacharyya_uncertainty, None, None, None, None, 4, None, merge_uncertainties=True, class_dices=False, prediction_path=refined_prediction_save_path)

    print("Uncertainty computation finished.")