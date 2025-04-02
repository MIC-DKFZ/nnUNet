# lsf22-gpu01
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 4 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=5 nnUNetv2_train 5 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=6 nnUNetv2_train 8 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=7 nnUNetv2_train 10 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"

# lsf22-gpu03
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 17 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 27 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 55 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=6 nnUNetv2_train 220 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"

# lsf22-gpu05
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 223 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 4 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 5 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 8 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=6 nnUNetv2_train 10 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=7 nnUNetv2_train 17 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"

# lsf22-gpu06
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 27 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 55 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 220 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224.sh && CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 223 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224_balintsfix.sh && CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224_balintsfix.sh && CUDA_VISIBLE_DEVICES=5 nnUNetv2_train 4 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224_balintsfix.sh && CUDA_VISIBLE_DEVICES=6 nnUNetv2_train 5 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224_balintsfix.sh && CUDA_VISIBLE_DEVICES=7 nnUNetv2_train 8 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"

# lsf22-gpu07
screen -dm bash -c ". ~/load_env_torch224_balintsfix.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 10 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224_balintsfix.sh && CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 17 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224_balintsfix.sh && CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 27 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224_balintsfix.sh && CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 55 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224_balintsfix.sh && CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 220 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224_balintsfix.sh && CUDA_VISIBLE_DEVICES=5 nnUNetv2_train 223 3d_fullres 0 -tr nnUNetTrainer_noDummy2DDA -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224_balintsfix.sh && CUDA_VISIBLE_DEVICES=6 nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
screen -dm bash -c ". ~/load_env_torch224_balintsfix.sh && CUDA_VISIBLE_DEVICES=7 nnUNetv2_train 4 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"


# launched as jobs
bsub -R "select[hname!='e230-dgx2-2']" -R "select[hname!='e230-dgx2-1']"  -q gpu -gpu num=1:j_exclusive=yes:gmem=1G ". ~/load_env_torch224_balintsfix.sh && nnUNetv2_train 5 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
bsub -R "select[hname!='e230-dgx2-2']" -R "select[hname!='e230-dgx2-1']"  -q gpu -gpu num=1:j_exclusive=yes:gmem=1G ". ~/load_env_torch224_balintsfix.sh && nnUNetv2_train 8 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
bsub -R "select[hname!='e230-dgx2-2']" -R "select[hname!='e230-dgx2-1']"  -q gpu -gpu num=1:j_exclusive=yes:gmem=1G ". ~/load_env_torch224_balintsfix.sh && nnUNetv2_train 10 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
bsub -R "select[hname!='e230-dgx2-2']" -R "select[hname!='e230-dgx2-1']"  -q gpu -gpu num=1:j_exclusive=yes:gmem=1G ". ~/load_env_torch224_balintsfix.sh && nnUNetv2_train 17 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
bsub -R "select[hname!='e230-dgx2-2']" -R "select[hname!='e230-dgx2-1']"  -q gpu -gpu num=1:j_exclusive=yes:gmem=1G ". ~/load_env_torch224_balintsfix.sh && nnUNetv2_train 27 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
bsub -R "select[hname!='e230-dgx2-2']" -R "select[hname!='e230-dgx2-1']"  -q gpu -gpu num=1:j_exclusive=yes:gmem=1G ". ~/load_env_torch224_balintsfix.sh && nnUNetv2_train 55 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
bsub -R "select[hname!='e230-dgx2-2']" -R "select[hname!='e230-dgx2-1']"  -q gpu -gpu num=1:j_exclusive=yes:gmem=1G ". ~/load_env_torch224_balintsfix.sh && nnUNetv2_train 220 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"
bsub -R "select[hname!='e230-dgx2-2']" -R "select[hname!='e230-dgx2-1']"  -q gpu -gpu num=1:j_exclusive=yes:gmem=1G ". ~/load_env_torch224_balintsfix.sh && nnUNetv2_train 223 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetMPlans --disable_checkpointing"

