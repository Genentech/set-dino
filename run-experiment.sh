DATA_PATH='/scratch/site/gred/resbioai-work/comp_vision/funk_ops/funk22_lmdb_ordered_illum'
METADATA='/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/lmdb_by_plate_interphase_illum/metadata_all.csv'
OUTPUT='/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops_dino_training'
NTC_STATS='/projects/site/gred/resbioai/comp_vision/cellpaint-ai/ops/datasets/funk22/funk22_metadata_for_lmdb/ntc_stats/all_ntc_stats.npy'
WANDB_ENTITY='rb-aiml'


python train.py \
--num_gpus 1 \
--num_workers 12 \
--wandb_entity $WANDB_ENTITY \
--data_path $DATA_PATH \
--metadata_df_path $METADATA \
--output_dir $OUTPUT \
--ntc_stats_file $NTC_STATS \
--crop_size 96 \
--norm_last_layer 1 \
--batch_size_per_gpu 512 \
--out_dim 2048 \
--patch_size 16 \
--warmup_teacher_temp 0.01 \
--warmup_teacher_temp_epochs 20 \
--teacher_temp 0.01 \
--epochs 300 \
--in_channels 4 \
--momentum_teacher 0.998 \
--check_val_every_n_epoch 5 \
--normalizer 'ntc' \
--local_crops_number 8 \
--n_cells 16 \
--sampling_strategy 'cross_batch' \
--by_guide 0 \
--exp_id "test"