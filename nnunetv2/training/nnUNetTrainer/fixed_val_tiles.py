from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window


@dataclass(frozen=True)
class FixedValTile:
    """A deterministic validation crop.

    key is the dataset case identifier passed to dataset.load_case.
    starts can have these shapes:
    - (y_start, x_start) for native 2D patches on 2D cases.
    - (z_start, y_start, x_start) for native 3D patches on 3D cases.
    - (slice_idx, y_start, x_start) for 2D patches applied slice-by-slice to 3D cases.
    """
    key: str
    starts: Tuple[int, ...]


@dataclass(frozen=True)
class FixedValTileStats:
    num_cases: int
    num_candidate_tiles: int
    num_requested_tiles: int
    num_selected_tiles: int
    num_tiles_on_rank: int


class FixedValTileManager:
    def __init__(self, dataset, patch_size: Tuple[int, ...], batch_size: int, transforms,
                 tile_step_size: float, seed: int, num_tiles: int,
                 is_ddp: bool = False, rank: int = 0, world_size: int = 1):
        self.dataset = dataset
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.transforms = transforms
        self.tile_step_size = tile_step_size
        self.seed = seed
        self.num_tiles = num_tiles
        self.is_ddp = is_ddp
        self.rank = rank
        self.world_size = world_size
        self.tiles, self.stats = self._build_tiles()

    def __len__(self):
        return len(self.tiles)

    def iter_batches(self):
        for batch_start in range(0, len(self.tiles), self.batch_size):
            yield self._make_batch(self.tiles[batch_start:batch_start + self.batch_size])

    def _build_tiles(self) -> Tuple[List[FixedValTile], FixedValTileStats]:
        tiles = []
        num_cases = len(self.dataset.identifiers)

        for key in self.dataset.identifiers:
            data, _, _, _ = self.dataset.load_case(key)
            image_size = tuple(int(i) for i in data.shape[1:])

            if len(self.patch_size) < len(image_size):
                # 2D configurations keep 3D cases as (c, z, y, x) but use 2D patches (y, x), so we
                # enumerate in-plane sliding-window tiles for each slice independently.
                assert len(self.patch_size) == len(image_size) - 1
                # compute_steps_for_sliding_window requires image_size >= patch_size. For smaller validation
                # slices we enumerate one virtual patch-sized tile and let crop_and_pad_nd add the padding later.
                tiled_size = tuple(max(i, p) for i, p in zip(image_size[1:], self.patch_size))
                steps = compute_steps_for_sliding_window(tiled_size, self.patch_size, self.tile_step_size)
                for slice_idx in range(image_size[0]):
                    for starts in product(*steps):
                        tiles.append(FixedValTile(key, (slice_idx, *starts)))
            else:
                # compute_steps_for_sliding_window requires image_size >= patch_size. For smaller validation
                # images we enumerate one virtual patch-sized tile and let crop_and_pad_nd add the padding later.
                tiled_size = tuple(max(i, p) for i, p in zip(image_size, self.patch_size))
                steps = compute_steps_for_sliding_window(tiled_size, self.patch_size, self.tile_step_size)
                for starts in product(*steps):
                    tiles.append(FixedValTile(key, tuple(starts)))

        num_candidate_tiles = len(tiles)
        requested_num_tiles = int(self.num_tiles)

        if requested_num_tiles < len(tiles):
            rng = np.random.default_rng(self.seed)
            selected_indices = rng.choice(len(tiles), size=requested_num_tiles, replace=False)
            tiles = [tiles[i] for i in selected_indices]

        num_selected_tiles = len(tiles)
        if self.is_ddp:
            if len(tiles) < self.world_size:
                raise RuntimeError(
                    f"Fixed validation selected {len(tiles)} tiles for {self.world_size} DDP ranks. "
                    f"Increase fixed_val_num_tiles or validate with fewer ranks so every rank gets at least one tile."
                )
            tiles = tiles[self.rank::self.world_size]

        stats = FixedValTileStats(
            num_cases=num_cases,
            num_candidate_tiles=num_candidate_tiles,
            num_requested_tiles=requested_num_tiles,
            num_selected_tiles=num_selected_tiles,
            num_tiles_on_rank=len(tiles),
        )
        return tiles, stats

    def _crop_tile(self, data: np.ndarray, seg: np.ndarray, seg_prev: Optional[np.ndarray],
                   starts: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(starts) == len(self.patch_size) + 1:
            # 2D-over-3D case: starts contains the selected slice index followed by the in-plane
            # tile starts, so crop a single (y, x) slice from data shaped (c, z, y, x).
            slice_idx = starts[0]
            spatial_starts = starts[1:]
            # crop_and_pad_nd expects one [start, end] interval per spatial axis.
            # and our target size is self.patch_size
            bbox = [[i, i + j] for i, j in zip(spatial_starts, self.patch_size)]
            data_cropped = crop_and_pad_nd(data[:, slice_idx], bbox, 0)
            seg_cropped = crop_and_pad_nd(seg[:, slice_idx], bbox, -1)
            if seg_prev is not None:
                seg_prev_cropped = crop_and_pad_nd(seg_prev[slice_idx], bbox, -1)
                seg_cropped = np.concatenate((seg_cropped, seg_prev_cropped[None]), axis=0)
        else:
            # crop_and_pad_nd expects one [start, end] interval per spatial axis.
            # and our target size is self.patch_size
            bbox = [[i, i + j] for i, j in zip(starts, self.patch_size)]
            data_cropped = crop_and_pad_nd(data, bbox, 0)
            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_prev_cropped = crop_and_pad_nd(seg_prev, bbox, -1)
                seg_cropped = np.concatenate((seg_cropped, seg_prev_cropped[None]), axis=0)

        return torch.from_numpy(data_cropped).float(), torch.from_numpy(seg_cropped).to(torch.int16)

    def _make_batch(self, tiles: List[FixedValTile]) -> dict:
        data_samples = []
        seg_samples = []

        for tile in tiles:
            data, seg, seg_prev, _ = self.dataset.load_case(tile.key)
            data = np.asarray(data[:])
            seg = np.asarray(seg[:])
            if seg_prev is not None:
                seg_prev = np.asarray(seg_prev[:])

            data_cropped, seg_cropped = self._crop_tile(data, seg, seg_prev, tile.starts)
            transformed = self.transforms(image=data_cropped, segmentation=seg_cropped)
            data_samples.append(transformed['image'])
            seg_samples.append(transformed['segmentation'])

        data_batch = torch.stack(data_samples)

        # Deep supervision transforms return one target per supervision scale.
        if isinstance(seg_samples[0], list):
            target_batch = [
                torch.stack([sample[scale_idx] for sample in seg_samples])
                for scale_idx in range(len(seg_samples[0]))
            ]
        else:
            target_batch = torch.stack(seg_samples)

        return {'data': data_batch, 'target': target_batch, 'keys': [tile.key for tile in tiles]}
