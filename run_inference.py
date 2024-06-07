import argparse
import os
import numpy as np
import pandas as pd
import torch.utils.data

from constants import Column
from dataset import InferenceDataset
import augmentations as aug
import utils
import vision_transformer as vits


def get_args_parser():
    parser = argparse.ArgumentParser('Model evaluation', add_help=False)

    # Data
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--dataset_path',
                        default='/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/funk22_lmdb',
                        type=str, help='Please specify path to the saved cell images')
    parser.add_argument('--metadata_df_path', default='/home/yaoh11/scratch/ops/metadata/metadata_all_interphase.csv',
                        type=str, help='Please specify path to the saved cell images')
    parser.add_argument('--ntc_stats_file', default=None,
                        type=str, help='Please specify path to the saved ntc statistics (this is only required when using ntc normalization for cellular images).')
    parser.add_argument('--crop_size', default=100, type=int,
                        help='The size of the cropped single-cell images')
    parser.add_argument('--in_channels', default=5, type=int, help='Number of channels of the input image')
    parser.add_argument('--plate', default='20200206_6W-LaC025B',
                        type=str, help='Plate name')
    parser.add_argument('--well', default='A1',
                        type=str, help='Well name')

    parser.add_argument('--normalizer', type=str, default='zscore',
                        choices=['clip', 'log', 'arcsinh', 'zscore', 'ntc'],
                        help="""The method used to aggregate embeddings from cells in the same set.""")

    # Evaluation parameters
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')


    args = parser.parse_args()
    return args


def load_pretrained_model(args):
    # Load model
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, in_chans=args.in_channels)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    else:
        raise NotImplementedError

    # Run inference
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    # # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")
    return model, embed_dim


def inference(model, dataloader, args):
    output_list = []
    key_list = []
    with torch.no_grad():
        for idx, (images, keys) in enumerate(dataloader):
            if torch.cuda.is_available():
                input = images.cuda(non_blocking=True)
            else:
                input = images

            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(input, args.n_last_blocks)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if args.avgpool_patchtokens:
                    output = torch.cat((output.unsqueeze(-1),
                                        torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(input)

            if torch.cuda.is_available():
                output_list.append(output.cpu().numpy())
            else:
                output_list.append(output.numpy())
            key_list.extend(list(keys))

    output_all = np.concatenate(output_list, axis=0)
    embed_dim = output.shape[-1]
    output_df = pd.DataFrame(output_all, columns=[f'feature_{idx}' for idx in range(embed_dim)])

    # Add metadata
    output_df['key'] = key_list
    output_df[[Column.plate.value, Column.well.value, Column.tile.value, Column.gene.value,
               Column.sgRNA.value, 'meta_df_index']] = \
        output_df.apply(lambda x: pd.Series(str(x['key']).split(';')), axis=1)
    return output_df


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    parts = args.pretrained_weights.split('/')
    run_id = parts[-2] + '_' + parts[-1].split('.')[0].split('-avg_top')[0]
    os.makedirs(os.path.join(args.output_dir, run_id, 'level_0_raw'), exist_ok=True)

    # Load metadata
    df_all = pd.read_pickle(args.metadata_df_path)
    df = df_all[df_all[Column.plate.value] == args.plate].copy()
    df_group = df.groupby([Column.plate.value, Column.well.value])
    df_sub = df_group.get_group((args.plate, args.well))

    # Create the dataset and data loader
    normalizer = aug.Normalization(method=args.normalizer, ntc_stats_file=args.ntc_stats_file)
    ds = InferenceDataset(df_sub, dataset_path=args.dataset_path, crop_size=args.crop_size, normalizer=normalizer)

    dataloader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers or 0,
        drop_last=False,
    )

    # Load the pretrained model
    model, embed_dim = load_pretrained_model(args)
    # Run inference
    output_df = inference(model, dataloader, args)
    # Save the output
    output_df.to_parquet(os.path.join(args.output_dir, run_id, 'level_0_raw', f'{args.plate}_{args.well}.parquet'))


if __name__ == '__main__':
    args = get_args_parser()
    main(args)
