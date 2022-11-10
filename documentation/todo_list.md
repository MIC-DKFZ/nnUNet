Planned:
- parametrizable DA intensity
- random field aug 
- hrnet
- keep highres path in unet
- RMI
- resampling strats

- INFERENCE FROM RESTORED TRAINER!
- INFERENCE WITH GPU RESAMPLING
- task163 where all cases with non-adult teeth are removed or non-adult teeth are set to bg
- validation and inference need to share code, just different data generators

all on gpu should also hold data on gpu?

Done:
- label smoothing
- focal loss
- smooth parameter in dice (clamp)
- dice topk
- CE
- 2d inference mirroring axes
- no data augmentation
- no deep supervision
- baseline ord0
- dicelip + ce
- fix noda
- benchmarking
- speed get_tp_fp_fn
