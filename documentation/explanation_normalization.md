# Intensity normalization in nnU-Net 

The type of intensity normalization applied in nnU-Net can be controlled via the `channel_names` (former `modalities`)
entry in the dataset.json. Just like the old nnU-Net, per-channel z-scoring as well as dataset-wide z-scoring based on 
foreground intensities are supported. However, there have been a few additions as well.

Reminder: The `channel_names` entry typically looks like this: 

    "channel_names": {
        "0": "T2",
        "1": "ADC"
    },

It has as many entries as there are input channels for the given dataset.

To tell you a secret, nnU-Net does not really care what your channels are called. We just use this to determine what normalization
scheme will be used for the given dataset. nnU-Net requires you to specify a normalization strategy for each of your input channels! 
If you enter a channel name that is not in the following list, the default (`zscore`) will be used.

Here is a list of currently available normalization schemes:

- `CT`: Perform CT normalization. Specifically, collect intensity values from the foreground classes (all but the 
background and ignore) from all training cases, compute the mean, standard deviation as well as the 0.5 and 
99.5 percentile of the values. Then clip to the percentiles, followed by subtraction of the mean and division with the 
standard deviation. The normalization that is applied is the same for each training case (for this input channel).
The values used by nnU-Net for normalization are stored in the `foreground_intensity_properties_per_channel` entry in the 
corresponding plans file. This normalization is suitable for modalities presenting physical quantities such as CT 
images and ADC maps.
- `noNorm` : do not perform any normalization at all
- `rescale_to_0_1`: rescale the intensities to [0, 1]
- `rgb_to_0_1`: assumes uint8 inputs. Divides by 255 to rescale uint8 to [0, 1]
- `zscore`/anything else: perform z-scoring (subtract mean and standard deviation) separately for each train case

**Important:** The nnU-Net default is to perform 'CT' normalization for CT images and 'zscore' for everything else! If 
you deviate from that path, make sure to benchmark whether that actually improves results! 

# How to implement custom normalization strategies?
- Head over to nnunetv2/preprocessing/normalization
- implement a new image normalization class by deriving from ImageNormalization
- register it in nnunetv2/preprocessing/normalization/map_channel_name_to_normalization.py:channel_name_to_normalization_mapping. 
This is where you specify a channel name that should be associated with it
- use it by specifying the correct channel_name

Normalization can only be applied to one channel at a time. There is currently no way of implementing a normalization scheme 
that gets multiple channels as input to be used jointly!
