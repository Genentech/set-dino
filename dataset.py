import random
from typing import Optional, Callable
import lmdb
import numpy as np
import pandas as pd
import torch.utils.data

import augmentations
from constants import Column


class CellularImageDatasetLMDB(torch.utils.data.Dataset):
    def __init__(self,
                 metadata_df: pd.DataFrame,
                 dataset_path: str,
                 normalizer: augmentations.Normalization,
                 min_cells: Optional[int] = 10,
                 n_cells: Optional[int] = 1,
                 crop_size: Optional[int] = 160,
                 num_channels: Optional[int] = 4,
                 global_augmentation: Optional[Callable] = None,
                 local_augmentation: Optional[Callable] = None,
                 num_local_crops=0,
                 sampling_strategy='cross_batch',
                 by_guide=True,
                 ):
        if local_augmentation is None:
            assert num_local_crops == 0

        self.min_cells = min_cells
        self.n_cells = n_cells
        self.crop_radius = crop_size // 2
        self.global_augmentation = global_augmentation
        self.local_augmentation = local_augmentation
        self.num_local_crops = num_local_crops
        self.num_channels = num_channels
        self.normalizer = normalizer

        self.by_guide = by_guide
        if by_guide:
            self.group_key = Column.sgRNA.value
        else:
            self.group_key = Column.gene.value

        # filter out guides with too few cells or too few replicates across wells
        self.df = self._data_filtering(metadata_df)

        # create grouped data frame
        self._create_grouped_data()

        # load the lmdb dataset
        env = lmdb.Environment(dataset_path, readonly=True, readahead=False, lock=False)
        txn = env.begin(write=False, buffers=True)
        self.env = env
        self.txn = txn

        self.sampling_strategy = sampling_strategy

    def _data_filtering(self, df_all: pd.DataFrame):
        # only keep cells in training set
        df_filtered = (df_all.groupby([Column.plate.value, Column.well.value, self.group_key]).
                       filter(lambda x: len(x) >= self.min_cells))

        if self.by_guide:
            df_filtered = (df_filtered.groupby([Column.gene.value, Column.sgRNA.value]).
                           filter(lambda x: len(x[[Column.gene.value, Column.sgRNA.value, Column.plate.value,
                                                   Column.well.value]].drop_duplicates()) >= 2))
        else:
            df_filtered = (df_filtered.groupby([Column.gene.value]).
                           filter(lambda x: len(x[[Column.gene.value, Column.plate.value,
                                                   Column.well.value]].drop_duplicates()) >= 2))
        return df_filtered

    def __del__(self):
        self.env.close()

    def _create_grouped_data(self):
        # extract the list of perturbation
        self.perts = list(self.df[self.group_key].drop_duplicates())

        # Scale the dataset size based on the number of cells per sample
        # It ensures that the number of mini-batches and the number of sampled cells per epoch are
        # consistent when n_cells varies.
        self.perts = self.perts * int(16 / self.n_cells)

        # On average each genetic perturbation contains four guide.
        # This ensures that the number of mini-batches remains the same when self.by_guide=False or self.by_guide=True
        if not self.by_guide:
            self.perts = self.perts * 4

        # 10 is an arbitrary scaling factor to increase the size of one epoch.
        self.perts = self.perts * 10

        # level 1 group - well level
        self.meta_group = (self.df[[self.group_key, Column.plate.value, Column.well.value]]
                           .drop_duplicates()
                           .groupby([self.group_key]))
        # level 2 group - cell level
        self.cell_group = self.df.groupby(
            [self.group_key, Column.plate.value, Column.well.value])

    def __len__(self):
        return len(self.perts)

    def __getitem__(self, index: int):
        # for each perturbation, sample a guide
        n_views = 2
        pert = self.perts[index]
        # for each perturbation, sample n_views wells (n_views global views from different wells)
        cell_group = self.meta_group.get_group(pert).drop_duplicates()

        if self.sampling_strategy == 'cross_batch':
            # sample a set of cells with the same perturbation and from different batches
            row_indices = random.sample(list(range(len(cell_group))), k=2)

        elif self.sampling_strategy == 'within_batch':
            # sample a set of cells with the same perturbation and from the same batches
            row_indices = random.sample(list(range(len(cell_group))), k=1) * 2

        elif self.sampling_strategy == 'same_cell':
            # use the same image with different data augmentations for teacher and student branches
            row_indices = random.sample(list(range(len(cell_group))), k=1)

        else:
            raise ValueError

        cell_indices = []
        well_list = []
        for view_idx in row_indices:
            cells, well = self._cell_sampler(cell_group, pert, view_idx, n_cells=self.n_cells)
            cell_indices.extend(cells)
            well_list.append(well)

        if self.sampling_strategy == 'same_cell':
            cell_indices = cell_indices*2
            well_list = well_list*2

        images = []
        for idx in cell_indices:
            images.append(self.read_single_cell_image(idx))

        images = np.stack(images, axis=0).reshape((n_views, -1) + images[0].shape)

        # Generate global views and local views
        n_local_per_view = self.num_local_crops // n_views
        global_views = []
        local_views = []
        for idx in range(n_views):
            image_set = images[idx]
            plate, well = well_list[idx]
            global_views.append(np.stack([self.normalizer(self.global_augmentation(image), (plate, well)) for image in image_set], axis=0))
            for _ in range(n_local_per_view):
                local_views.append(np.stack([self.normalizer(self.local_augmentation(image), (plate, well)) for image in image_set], axis=0))
        image_list = global_views + local_views
        image_list = [np.squeeze(temp).astype(np.float32) for temp in image_list]
        return image_list, []

    def _cell_sampler(self, cell_group, guide, idx, n_cells):
        # for each well, sample n_cells cell(s)
        plate = cell_group.iloc[idx][Column.plate.value]
        well = cell_group.iloc[idx][Column.well.value]
        cell_temp = self.cell_group.get_group((guide, plate, well))
        cell_inds = random.sample(list(cell_temp.index), k=n_cells)
        well = (plate, well)
        return cell_inds, well

    def read_single_cell_image(self, df_index):
        row = self.df.loc[df_index]
        plate = row[Column.plate.value]
        guide = row[Column.sgRNA.value]
        well = row[Column.well.value]
        tile = str(row[Column.tile.value])
        gene = row[Column.gene.value]

        key = f'{gene};{guide};{plate};{well};{tile};{df_index}'
        buf = self.txn.get(key.encode())
        arr = np.frombuffer(buf, dtype='float32')

        cell_image = arr.reshape((self.num_channels, self.crop_radius * 2, self.crop_radius * 2))
        return cell_image.astype(np.float32)


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_df, dataset_path, normalizer, crop_size=100, num_channels=4):
        self.crop_radius = crop_size // 2
        self.num_channels = num_channels

        self.df = metadata_df.sort_values(by=[Column.plate.value, Column.well.value, Column.tile.value,
                                              Column.gene.value])

        self.plate_list = np.unique(self.df[Column.plate.value].values)
        env = lmdb.Environment(dataset_path, readonly=True, readahead=False, lock=False)
        txn = env.begin(write=False, buffers=True)
        self.env = env
        self.txn = txn
        self.normalizer = normalizer

    def __len__(self):
        return self.df.shape[0]

    def __del__(self):
        self.env.close()

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        plate = row[Column.plate.value]
        well = row[Column.well.value]
        tile = str(row[Column.tile.value])
        gene = row[Column.gene.value]
        guide = row[Column.sgRNA.value]
        df_index = row.name

        key = f'{gene};{guide};{plate};{well};{tile};{df_index}'
        buf = self.txn.get(key.encode())
        arr = np.frombuffer(buf, dtype='float32')

        cell_image = arr.reshape((self.num_channels, self.crop_radius * 2, self.crop_radius * 2))
        cell_image = cell_image.astype(np.float32)

        image = self.normalizer(cell_image, (plate, well))
        uid = ';'.join([plate, well, str(tile), gene, guide, str(df_index)])
        return image, uid
