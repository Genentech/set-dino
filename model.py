"""
Pytorch lightning framework for Set-DINO training

Code on the model training, including building learning rate scheduler, weight decay scheduler, and
set up the model and loss functions (Line 75-207, Line 288-352)
are copy-pasted and/or adapted from Facebook's public DINO repository, licensed under the Apache License, Version 2.0 :
https://github.com/facebookresearch/dino/blob/main/main_dino.py


"""
import math
import sys
import logging

import cupy as cp
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchvision_models

import utils
from constants import Column, NTC
from eval_utils import KNN_classifier, standardize_per_catX
import vision_transformer as vits
from vision_transformer import DINOHead


torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))
_logger = logging.getLogger(__name__)


class DINO(pl.LightningModule):
    def __init__(self, args, step_per_epoch):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.step_per_epoch = step_per_epoch
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Load models
        self._load_models()

        # Loss functions
        dino_loss = DINOLoss(
            args.out_dim,
            args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        )
        self.dino_loss = dino_loss

        # Mixed precision training
        self.fp16_scaler = None

        # Build schedulers
        self._build_schedulers()

        # Activate manual optimization
        self.automatic_optimization = False

    @property
    def logger(self):
        return logging.getLogger(self.__class__.__name__)

    def _build_schedulers(self):
        args = self.args
        self.lr_schedule = utils.cosine_scheduler(
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
            args.min_lr,
            args.epochs, self.step_per_epoch,
            warmup_epochs=args.warmup_epochs,
        )
        self.wd_schedule = utils.cosine_scheduler(
            args.weight_decay,
            args.weight_decay_end,
            args.epochs, self.step_per_epoch,
        )
        # momentum parameter is increased to 1. during training with a cosine schedule
        self.momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                                        args.epochs, self.step_per_epoch)
        self.logger.info("Loss, optimizer and schedulers ready.")

    def _load_models(self):
        args = self.args
        # ============ build student and teacher networks ============
        if args.arch in vits.__dict__.keys():
            student = vits.__dict__[args.arch](
                patch_size=args.patch_size,
                drop_path_rate=args.drop_path_rate,  # stochastic depth
                in_chans=args.in_channels
            )
            teacher = vits.__dict__[args.arch](
                patch_size=args.patch_size,
                in_chans=args.in_channels
            )
            embed_dim = student.embed_dim
        else:
            raise ValueError(f"Unknown architecture: {args.arch}")

        # ============ load the pretrained weights if provided ============
        if args.pretrained_weights:
            utils.load_pretrained_weights(student, args.pretrained_weights,
                                          'student', args.arch, args.patch_size)
            utils.load_pretrained_weights(teacher, args.pretrained_weights,
                                          'teacher', args.arch, args.patch_size)
        else:
            teacher.load_state_dict(student.state_dict(), strict=False)

        student = utils.MultiCropWrapper(student, DINOHead(embed_dim,
                                                           args.out_dim,
                                                           n_cells=args.n_cells,
                                                           use_bn=args.use_bn_in_head,
                                                           norm_last_layer=args.norm_last_layer),
                                         n_cells=args.n_cells)
        teacher = utils.MultiCropWrapper(teacher, DINOHead(embed_dim,
                                                           args.out_dim,
                                                           n_cells=args.n_cells,
                                                           use_bn=args.use_bn_in_head),
                                         n_cells=args.n_cells)

        for p in teacher.parameters():
            p.requires_grad = False
        self.logger.info(f"Student and Teacher are built: they are both {args.arch} network.")

        self.teacher = teacher
        self.student = student

    def configure_optimizers(self):
        args = self.args
        params_groups = utils.get_params_groups(self.student)
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        elif args.optimizer == "lars":
            optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
        else:
            raise ValueError(f"Unknown optimizer {args.optimizer}")
        return optimizer

    def training_step(self, batch, batch_idx):
        args = self.args
        optimizer = self.optimizers()

        it = self.step_per_epoch * self.current_epoch + batch_idx  # global training iteration

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[it]

        # Flatten the replicates into batch dimension
        # Original im has a shape of (batch, num_replicates, channel, height, width)
        images = batch[0]
        images = [im.reshape((-1,) + im.shape[-3:]) for im in images]

        # teacher and student forward passes + compute dino loss
        teacher_output = self.teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = self.student(images)
        loss, global_loss, local_loss, batch_center = self.dino_loss(student_output, teacher_output, self.current_epoch)

        # Manual optimization
        optimizer.zero_grad()
        self.manual_backward(loss)
        if args.clip_grad:
            _ = utils.clip_gradients(self.student, args.clip_grad)
        utils.cancel_gradients_last_layer(self.current_epoch, self.student, args.freeze_last_layer)
        optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = self.momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        self.training_step_outputs.append({'loss': loss, 'lr': optimizer.param_groups[0]["lr"],
                'global_loss': global_loss, 'local_loss': local_loss})
        return loss

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        for key in ['loss', 'global_loss', 'local_loss']:
            self.log('train_' + key, np.mean([x[key].item() for x in outputs]), sync_dist=True)
        for key in ['lr']:
            self.log('train_' + key, np.mean([x['lr'] for x in outputs]), sync_dist=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        images, uids = batch

        args = self.args
        if "vit" in args.arch:
            intermediate_output = self.student.backbone.get_intermediate_layers(images, args.n_last_blocks)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if args.avgpool_patchtokens:
                output = torch.cat((output.unsqueeze(-1),
                                    torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
        else:
            output = self.student.backbone(images)
        self.validation_step_outputs.append({'output': output, 'uid': uids})

    def on_validation_epoch_end(self):
        output = torch.concat([x['output'] for x in self.validation_step_outputs], dim=0)
        uid = [y for x in self.validation_step_outputs for y in x['uid']]

        output = output.cpu().numpy()
        embed_dim = output.shape[-1]
        feature_cols = [f'feature_{idx}' for idx in range(embed_dim)]
        output_df = pd.DataFrame(output, columns=feature_cols)

        # Add metadata
        output_df['key'] = uid
        metadata_cols = [Column.plate.value, Column.well.value, Column.tile.value, Column.gene.value,
                         Column.sgRNA.value, 'meta_df_index']
        output_df[metadata_cols] = \
            output_df.apply(lambda x: pd.Series(str(x['key']).split(';')), axis=1)

        # Per-plate normalization
        output_df = standardize_per_catX(output_df, feature_cols, metadata_cols)

        # Apply per-plate gene-level aggregation
        output_df['agg_key'] = output_df.apply(lambda x: '_'.join(
            [x[Column.plate.value], x[Column.well.value], x[Column.gene.value]]), axis=1)
        output_df['batch'] = output_df.apply(lambda x: '_'.join(
            [x[Column.plate.value], x[Column.well.value]]), axis=1)

        d1 = dict.fromkeys(feature_cols, 'mean')
        d2 = dict.fromkeys(metadata_cols + ['agg_key', 'batch'], 'first')
        df = output_df.groupby('agg_key', as_index=False).agg({**d1, **d2})

        # KNN
        # Calculate KNN accuracy on perturbation classification
        class_type = Column.gene.value
        df_sampled = df[df[Column.gene.value] != NTC]
        top1_acc, _, num_classes = KNN_classifier(df_sampled, feature_cols, class_type,
                                                         k=self.args.k, temperature=self.args.temperature)

        # Calculate KNN accuracy on batch classification
        class_type = 'batch'
        df_sampled = df[df[Column.gene.value] != NTC]
        top1_acc_batch, _, num_classes = KNN_classifier(df_sampled, feature_cols, class_type,
                                                         k=self.args.k, temperature=self.args.temperature)

        # SVD decomposition. SVD is used to monitor the informativeness of the embeddings.
        # When the model collpase, the ratio and percentile become very low.
        # Note these metrics are confounded by the batch effect
        z = cp.transpose(df[feature_cols].values)
        c = cp.cov(z)
        _, d, _ = cp.linalg.svd(c)
        d = d.get()
        ratio = 100 * np.sum(d > 0.0001) / len(feature_cols)
        percentile = np.percentile(np.log(d), 50)
        self.log('val_top1_acc', top1_acc, sync_dist=True)
        self.log('val_top1_acc_batch', top1_acc_batch, sync_dist=True)
        self.log('val_median_svg', percentile, sync_dist=True)
        self.log('val_ratio_svg', ratio, sync_dist=True)
        self.validation_step_outputs.clear()


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        # ! original center momentum is 0.9
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]

        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        total_loss = 0
        n_loss_terms = 0
        loss_arr = [[], []]
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
                loss_arr[iq].append(loss.mean())

        loss_arr = torch.Tensor(loss_arr).mean(dim=0)
        global_view_loss = loss_arr[0]
        local_view_loss = loss_arr[1:].mean()
        total_loss /= n_loss_terms
        batch_center = self.update_center(teacher_output)
        return total_loss, global_view_loss, local_view_loss, batch_center

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            n = dist.get_world_size()
        else:
            n = 1
        batch_center = batch_center / (len(teacher_output) * n)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        return batch_center