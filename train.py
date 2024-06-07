import argparse
import os
from pathlib import Path
import logging
import wandb
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from pytorch_lightning.loggers import WandbLogger
from torchvision import models as torchvision_models

import augmentations as aug
import eval_utils, utils, dataset
from model import DINO
from constants import Column, GENES_FOR_VAL


torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

_logger = logging.getLogger(__name__)


class EarlyStoppingWithWarmup(callbacks.EarlyStopping):
    """
    EarlyStopping, except don't watch the first `warmup` epochs.
    """

    def __init__(self, warmup=10, **kwargs):
        super().__init__(**kwargs)
        self.warmup = warmup

    def on_validation_end(self, trainer, pl_module):
        pass

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup:
            return
        else:
            super()._run_early_stopping_check(trainer)


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Data paths
    parser.add_argument('--data_path', required=True,
                        type=str, help='Please specify path to the saved cell images in lmdb format')
    parser.add_argument('--metadata_df_path', required=True,
                        type=str, help='Please specify path to the metadata of cells from the OPS experiment')
    parser.add_argument('--ntc_stats_file', default=None,
                        type=str, help='Please specify path to the saved ntc statistics (this is only required when using ntc normalization for cellular images).')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--restored_checkpoint', default=None, type=str, help="Path to the checkpoint for training continuation.")

    # Data processing
    parser.add_argument('--min_cells', default=20, type=int,
                        help='The minimal number of cells each guide should have in the same batch')
    parser.add_argument('--n_cells', default=1, type=int,
                        help='The number of replicates sampled in the same batch')
    parser.add_argument('--crop_size', default=100, type=int,
                        help='The size of the cropped single-cell images')
    parser.add_argument('--normalizer', type=str, default='zscore',
                        choices=['clip', 'log', 'arcsinh', 'zscore', 'ntc'],
                        help="""The method used to normalize the cellular images.""")
    parser.add_argument('--sampling_strategy', type=str, default='cross_batch',
                        choices=['cross_batch', 'within_batch', 'same_cell'],
                        help="""Strategies of sampling cells""")
    parser.add_argument('--by_guide', default=1, type=utils.bool_flag,
                                    help="Whether to group cells based on guide (True) or based on genes (False)")

    # Model architecture
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'] \
                                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
                        help="Name of architecture to train. For quick experiments with ViTs")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--in_channels', default=5, type=int, help='Number of channels of the input image')
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)

    # Model training / optimization
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--patience', default=50, type=int,
                        help='The patience for early stopping')
    parser.add_argument('--early_stop', default=False, type=utils.bool_flag,
                        help="Whether to use early stop (Default: False)")
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int,
                        help='Check the validation performance every n train epochs.')

    # Validation metrics calculation
    parser.add_argument('--n_last_blocks', default=4, type=int,
                        help="Concatenate [CLS] tokens for the `n` last blocks.")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--k', default=5, type=int,
                        help='The number of nearest neighbors')
    parser.add_argument('--temperature', default=1, type=int, help='Temperature')

    # Misc
    parser.add_argument('--output_dir', default="./", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--wandb_entity', required=True, type=str, help='Entity name in weight and bias.')
    parser.add_argument('--exp_id', default=None, type=str, help='An experiment id.')
    parser.add_argument('--seed', default=1234, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=12, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--num_gpus', default=1, type=int,
                        help='The number of gpus')

    args = parser.parse_args()
    logging.basicConfig(level='INFO')
    return args


def train_dino(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # Reproducibility
    utils.fix_random_seeds(args.seed)
    rng = np.random.default_rng()

    # Initialize wandb experiment
    dt_string = utils._timestamp()
    exp_name = args.exp_id if args.exp_id is not None else f"sweep_{dt_string}"
    project_name = 'ops-dino-sweep' if args.exp_id is None else 'ops-dino'

    local_rank = 0
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("LOCAL_RANK can't be found.")

    if local_rank > 0:
        mode = "disabled"
    else:
        # Only track rank 0 process using wandb
        mode = "online"

    wandb.init(project=project_name, entity=args.wandb_entity, mode=mode,
               config=args, name=exp_name, tags=[args.sampling_strategy, f'n{args.n_cells}'])

    # Create the output directory
    setattr(args, 'output_dir', Path(args.output_dir) / exp_name)
    setattr(args, 'batch_size_per_gpu', int(args.batch_size_per_gpu / args.n_cells))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(save_dir=args.output_dir)

    # Prepare data augmentation and normalization
    global_augmentation = aug.OPSAugmentation(rng=rng, zoom_scale=0.9)
    local_augmentation = aug.OPSAugmentation(rng=rng, local_crop=True, scale_low=0.4, scale_upper=0.6,
                                             local_crop_size=48)

    ntc_stats_file = args.ntc_stats_file
    normalizer = aug.Normalization(method=args.normalizer, ntc_stats_file=ntc_stats_file)

    #  Load the metadata for individual cells, which will be used to sample cells
    metadata_df = pd.read_parquet(args.metadata_df_path)

    # Build data loader for training
    metadata_df_train = eval_utils.sampled_df_by_dataset(metadata_df, data_set='train')
    ds_train = dataset.CellularImageDatasetLMDB(metadata_df=metadata_df_train,
                                                dataset_path=args.data_path,
                                                min_cells=args.min_cells,
                                                n_cells=args.n_cells,
                                                crop_size=args.crop_size,
                                                global_augmentation=global_augmentation,
                                                local_augmentation=local_augmentation,
                                                normalizer=normalizer,
                                                num_local_crops=args.local_crops_number,
                                                sampling_strategy=args.sampling_strategy,
                                                by_guide=args.by_guide)

    data_loader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers or 0,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    # Build data loader for validation
    metadata_df_val = eval_utils.sampled_df_by_dataset(metadata_df, data_set='val')
    if GENES_FOR_VAL is not None:
        metadata_df_val = metadata_df_val[metadata_df_val[Column.gene.value].isin(GENES_FOR_VAL)]
    ds_val = dataset.InferenceDataset(metadata_df_val,
                                      dataset_path=args.data_path,
                                      crop_size=args.crop_size,
                                      normalizer=normalizer)

    data_loader_val = torch.utils.data.DataLoader(
        dataset=ds_val,
        batch_size=args.batch_size_per_gpu * args.n_cells * 2,
        shuffle=True,
        num_workers=args.num_workers or 0,
        drop_last=False,
        persistent_workers=True
    )

    # Set up the model
    step_per_epoch = len(data_loader_train) // args.num_gpus
    dino_model = DINO(args, step_per_epoch)

    _logger.info(f'The size of the training dataset is {len(ds_train)}')
    _logger.info(f'The size of the validation dataset is {len(ds_val)}')
    _logger.info(f'The number of steps per epoch is {step_per_epoch}')

    # Init callbacks
    checkpoint_last_callback = callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        save_last=True
    )

    checkpoint_acc_callback = callbacks.ModelCheckpoint(
        monitor="val_top1_acc",
        dirpath=args.output_dir,
        mode="max",
        filename="checkpoint-{epoch:02d}-{val_top1_acc:.4f}",
        save_top_k=2
    )
    progress_bar_callback = callbacks.TQDMProgressBar(refresh_rate=75)
    callback_list = [checkpoint_acc_callback, checkpoint_last_callback, progress_bar_callback]
    if args.early_stop:
        early_stop_callback = EarlyStoppingWithWarmup(
            monitor="val_top1_acc", min_delta=0.0,
            patience=args.patience, verbose=True, mode="max",
            warmup=200
        )
        callback_list.append(early_stop_callback)

    if args.num_gpus == 1:
        trainer = pl.Trainer(default_root_dir=args.output_dir,
                             max_epochs=args.epochs,
                             logger=logger,
                             callbacks=callback_list,
                             accelerator='gpu',
                             devices=1,
                             check_val_every_n_epoch=args.check_val_every_n_epoch
                             )
    else:
        trainer = pl.Trainer(default_root_dir=args.output_dir,
                             max_epochs=args.epochs,
                             logger=logger,
                             callbacks=callback_list,
                             accelerator='gpu',
                             devices=args.num_gpus,
                             strategy='ddp_find_unused_parameters_false',
                             check_val_every_n_epoch=args.check_val_every_n_epoch,
                             use_distributed_sampler=True
                             )
    _logger.info("Starting training.")
    trainer.fit(model=dino_model,
                train_dataloaders=data_loader_train,
                val_dataloaders=data_loader_val,
                ckpt_path=args.restored_checkpoint
                )


if __name__ == '__main__':
    train_dino(args=get_args_parser())
