### Install uncertainty nnUNet:
```
git clone https://github.com/Karol-G/nnUNet.git
cd nnUNet
pip install -e .
```
Modify in nnunet/path.py the variables datasets_path and os.environ["RESULTS_FOLDER"]

---

### Dataset preprocessing - Dataset 079 as example:
```
nnUNet_plan_and_preprocess -t 079 --verify_dataset_integrity
```

---

####Normal train - Example to train all folds:
```
nnUNet_train 3d_fullres nnUNetTrainerV2 Task079_frankfurt3 0 -d 0
nnUNet_train 3d_fullres nnUNetTrainerV2 Task079_frankfurt3 1 -d 1
nnUNet_train 3d_fullres nnUNetTrainerV2 Task079_frankfurt3 2 -d 2
nnUNet_train 3d_fullres nnUNetTrainerV2 Task079_frankfurt3 3 -d 3
nnUNet_train 3d_fullres nnUNetTrainerV2 Task079_frankfurt3 4 -d 4
```
```
-d: GPU device to train on
```

---

####Dropout train - Example to train all folds:
```
nnUNet_train 3d_fullres nnUNetTrainerV2 Task079_frankfurt3 0 -d 0 --dropout
nnUNet_train 3d_fullres nnUNetTrainerV2 Task079_frankfurt3 1 -d 1 --dropout
nnUNet_train 3d_fullres nnUNetTrainerV2 Task079_frankfurt3 2 -d 2 --dropout
nnUNet_train 3d_fullres nnUNetTrainerV2 Task079_frankfurt3 3 -d 3 --dropout
nnUNet_train 3d_fullres nnUNetTrainerV2 Task079_frankfurt3 4 -d 4 --dropout
```
```
-d: GPU device to train on
```

---

####Normal predict for Task079_frankfurt3 with single fold:
```
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 0 -d 0 -chk model_best --disable_tta
```
```
-i: Absolute input data path
-o: Absolute output data path
-d: The chosen fold
-chk: Either 'model_best' or 'model_final'
-d: GPU device to train on
--disable_tta: Optional, Disable test time augmentation, x8 speed up, small loss in performance
```

---

####Normal predict for Task079_frankfurt3 ensemble:
```
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task079_frankfurt3 -d 0 --disable_tta
```
```
-i: Absolute input data path
-o: Absolute output data path
-d: GPU device to train on
--disable_tta: Optional, Disable test time augmentation, x8 speed up, small loss in performance
```

---

####Uncertainty Prediction TTA for Task079_frankfurt3 with single fold:
WICHTIG: Entweder jeden `nnUNet_predict` Befehl nacheinander ausführen und warten bis der vorhergehende fertig ist oder wenn parallel dann verschiedene output directories über -o angeben, da sonst Datei-Konflikte entstehen können. Anschließend dann Ordner Inhalte zusammenführen.
```
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 0 -d 0 -chk model_best --output_probabilities -uncertainty_tta 0
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 0 -d 1 -chk model_best --output_probabilities -uncertainty_tta 1
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 0 -d 2 -chk model_best --output_probabilities -uncertainty_tta 2
...
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 0 -d 7 -chk model_best --output_probabilities -uncertainty_tta 7
```
(Need to do this from 0 to 7 for all possible 8 TTA predictions)
```
--output_probabilities: To output probabilities and not a mask
-uncertainty_tta X: Predict with tta configuration X
```

---

####Uncertainty Prediction MC dropout for Task079_frankfurt3 with single fold (NEED TO TRAIN WITH DROPOUT BEFORE):
WICHTIG: Entweder jeden `nnUNet_predict` Befehl nacheinander ausführen und warten bis der vorhergehende fertig ist oder wenn parallel dann verschiedene output directories über -o angeben, da sonst Datei-Konflikte entstehen können. Anschließend dann Ordner Inhalte zusammenführen.
```
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 0 -d 0 -chk model_best --output_probabilities -mcdo 0
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 0 -d 1 -chk model_best --output_probabilities -mcdo 1
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 0 -d 2 -chk model_best --output_probabilities -mcdo 2
...
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 0 -d 7 -chk model_best --output_probabilities -mcdo 7
```
(Need to do this from 0 to 7 for 8 MC dropout predictions)
```
--output_probabilities: To output probabilities and not a mask
-mcdo X: Predict with mcdo X
```

---

####Uncertainty Prediction Ensemble for Task079_frankfurt3:
WICHTIG: Entweder jeden `nnUNet_predict` Befehl nacheinander ausführen und warten bis der vorhergehende fertig ist oder wenn parallel dann verschiedene output directories über -o angeben, da sonst Datei-Konflikte entstehen können. Anschließend dann Ordner Inhalte zusammenführen.
```
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 0 -d 0 -chk model_best --output_probabilities
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 1 -d 1 -chk model_best --output_probabilities
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 2 -d 2 -chk model_best --output_probabilities
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 3 -d 3 -chk model_best --output_probabilities
nnUNet_predict -i /absolute/path/to/imagesTs -o /absolute/path/to/imagesTs_predicted -t Task079_frankfurt3 -m 3d_fullres -f 4 -d 4 -chk model_best --output_probabilities
```
(Need to do this from 0 to 4 for all 5 ensemble predictions)
```
--output_probabilities: To output probabilities and not a mask
```

---

####Compute uncertainties from TTA, MMC dropout or ensemble:
```
python comp_uncertainties.py -i /absolute/path/to/imagesTs_predicted -o /absolute/path/to/imagesTs_uncertainties
```