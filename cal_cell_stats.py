import os
import random
import warnings
from typing import Optional, Callable
import numpy as np
from enum import Enum
import argparse
import lmdb
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from set_dino.constants import Column, NTC, PH_DIMS

np.random.seed(0)
random.seed(0)


def mad(arr):
    return np.median(np.abs(arr - np.median(arr))) * 1.48

class CellData:
    def __init__(self, metadata_df, dataset_path, plate, well, ntc_only=True, crop_size=100, num_channels=5):
        self.crop_radius = crop_size // 2
        self.num_channels = num_channels
        self.plate = plate
        self.well = well

        metadata_df = metadata_df[
            (metadata_df[Column.plate.value] == self.plate) & (metadata_df[Column.well.value] == self.well)
            ]
        if ntc_only:
            metadata_df = metadata_df[metadata_df[Column.gene.value] == NTC]

        # Remove cells near edges
        df_filtered = metadata_df[
            metadata_df[Column.cell_y.value].between(self.crop_radius, PH_DIMS[0] - self.crop_radius)]
        df_filtered = df_filtered[
            df_filtered[Column.cell_x.value].between(self.crop_radius, PH_DIMS[1] - self.crop_radius)]

        df_filtered.sort_values(by=[Column.plate.value, Column.well.value, Column.tile.value,
                                    Column.gene.value])
        self.df = df_filtered
        # import pdb; pdb.set_trace();
        self.plate_list = np.unique(self.df[Column.plate.value].values)
        # open the lmdb dataset
        self.txn_dict = self._load_lmdb_dataset(dataset_path)

    def _load_lmdb_dataset(self, dataset_root):
        txn_dict = {}
        for name in self.plate_list:
            if os.path.exists(os.path.join(dataset_root, name)):
                env = lmdb.Environment(os.path.join(dataset_root, name), readonly=True, readahead=False, lock=False)
                txn_dict[name] = env.begin(write=False, buffers=True)
            else:
                warnings.warn(f"LMDB dataset for plate {name} doesn't exist")
        return txn_dict

    def __len__(self):
        return self.df.shape[0]

    def get_metadata(self):
        return self.df

    def get_stats(self):
        NUM_CHANNELS = 4
        channels = [[] for i in range(NUM_CHANNELS)]
        for index in tqdm(range(len(self.df))):
            row = self.df.iloc[index]
            plate = row[Column.plate.value]
            well = row[Column.well.value]
            tile = str(row[Column.tile.value])
            gene = row[Column.gene.value]
            guide = row[Column.sgRNA.value]
            df_index = row.name
            key = f'{plate}_{well}_{tile}_{gene}_{df_index}'
            buf = self.txn_dict[plate].get(key.encode())
            if buf is None:
                continue
            arr = np.frombuffer(buf, dtype='float32')
            cell_image = arr.reshape((self.num_channels - 1, self.crop_radius * 2, self.crop_radius * 2))
            ### make it 96x96
            cell_image = cell_image[:, 2:98, 2:98]
            # import pdb
            # pdb.set_trace()
            assert np.min(cell_image) >= 0
            # image_size = (self.in_channels, self.crop_radius*2, self.crop_radius*2)
            # cell_image = seq.reshape(image_size)
            cell_image = cell_image.astype(np.float32)
            assert plate == self.plate and self.well == well
            for i in range(NUM_CHANNELS):
                channels[i].append(cell_image[i])
        channels = [np.stack(arr) for arr in channels]
        print(self.plate, self.well, channels[0].shape)
        assert channels[1][0].shape == (96, 96)
        assert channels[2].shape == channels[3].shape
        medians = [np.median(channels[i]) for i in range(NUM_CHANNELS)]
        means = [np.mean(channels[i]) for i in range(NUM_CHANNELS)]
        stds = [np.std(channels[i]) for i in range(NUM_CHANNELS)]
        mads = [mad(channels[i]) for i in range(NUM_CHANNELS)]
        return {"median": medians, "mean": means, "std": stds, "mads": mads}


def main(plate, well, ntc_only):
    dataset_root = '/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/lmdb_by_plate_interphase_illum'
    metadatadf = pd.read_pickle(
        '/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/lmdb_by_plate_interphase_illum/metadata_all.csv')
    dataset = CellData(metadatadf, dataset_root, plate, well, ntc_only=ntc_only)
    if len(dataset) > 0:
        stats = dataset.get_stats()
    else:
        stats = None
    return stats

    #np.save(os.path.join(output_root, f"{plate}_{well}.npy"), stats)

if __name__ == "__main__":
    ntc_only = False
    if ntc_only:
        output_root = '/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/funk22_metadata_for_lmdb/ntc_stats'
        output_filename = "all_ntc_stats.npy"
    else:
        output_root = '/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/funk22_metadata_for_lmdb/cell_stats'
        output_filename = "all_stats.npy"
    os.makedirs(output_root, exist_ok=True)
    print(f'{ntc_only=}')
    print(f'{output_root=}')

    #'20200202_6W-LaC024A', '20200202_6W-LaC024D', '20200202_6W-LaC024E',
    plate_list = [
                  '20200202_6W-LaC024F', '20200206_6W-LaC025A', '20200206_6W-LaC025B',
                  '20200202_6W-LaC024B', '20200202_6W-LaC024C']
    well_list = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']

    out_dict = {}
    for plate in plate_list:
        for well in well_list:
            print(f"Processing {plate} {well}")
            stats = main(plate, well, ntc_only=ntc_only)
            if stats is not None:
                out_dict[(plate, well)] = stats
            else:
                print(f'No date in {plate} {well}')
        # to keep intermediate results
        np.save(os.path.join(output_root, output_filename), out_dict)