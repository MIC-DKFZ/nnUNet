# Intensity Normalization

This page explains how nnU-Net v2 chooses and applies intensity normalization.

## Where normalization is configured

Normalization is controlled through `channel_names` in `dataset.json`.

Example:

```json
{
  "channel_names": {
    "0": "T2",
    "1": "ADC"
  }
}
```

nnU-Net uses these names to map each input channel to a normalization scheme.

## Default behavior

- `CT` channels use dataset-level foreground-based normalization
- anything else defaults to per-case `zscore`

This means the channel names affect preprocessing behavior even if they are primarily human-readable labels.

## Available normalization schemes

- `CT`: clip to foreground percentiles, then normalize using dataset-level foreground statistics
- `zscore`: per-case z-scoring
- `noNorm`: no normalization
- `rescale_to_0_1`: rescale intensities to `[0, 1]`
- `rgb_to_0_1`: divide uint8 RGB inputs by `255`

## Why `CT` is special

For `CT`, nnU-Net computes foreground intensity statistics over the training set and stores them in the plans file. This is appropriate for channels with physically meaningful intensity scales such as CT and often ADC.

## Custom normalization

To add your own normalization strategy:

1. Implement a new class in `nnunetv2.preprocessing.normalization`
2. Register it in `map_channel_name_to_normalization.py`
3. Use the associated channel name in `dataset.json`

Current limitation:

- normalization is defined per channel
- there is no built-in multi-channel joint normalization scheme

## Related pages

- [Dataset and input format reference](../reference/dataset-format.md)
- [Plans and configuration reference](../reference/plans-and-configuration.md)

## Detailed legacy page

- [Intensity normalization in nnU-Net](../explanation_normalization.md)
