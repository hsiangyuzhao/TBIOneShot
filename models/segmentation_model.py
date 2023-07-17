import os

import wandb
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from monai.losses import DiceLoss, MultiScaleLoss, DiceCELoss, SSIMLoss

from base.base_modules import SobelEdge
from base.base_segmentation import MultiNetworkSegmentationModel
from base.base_wandb_model import WandBModel
from models.networks import UNet, TupleUNet
from models.registration_modules import SpatialTransformer, RegistrationCriterion, DVF2DDF
from utils.metric_calculator import calculate_Dice_score
from utils.iteration.iterator import PolynomialLRWithWarmUp, CosineAnnealingWithWarmUp, MetricMeter, rand_bbox
from utils.ddp_utils import gather_object_across_processes


class AtlasSeg(MultiNetworkSegmentationModel, WandBModel):
    """
    Proposed
    """
    def __init__(self, cfg, num_classes, amp=False):
        """
        Args:
            cfg: training configurations
            num_classes: number of classes
            amp: bool, whether to enable PyTorch native automatic mixed-precision training
        """
        MultiNetworkSegmentationModel.__init__(self, cfg, num_classes, amp)
        WandBModel.__init__(self, cfg)
        # define network
        self.net_names = ['seg', 'reg', 'aug']
        # common setting
        # self.seg = UNet(num_classes=num_classes, leaky=cfg['MODEL']['LEAKY'], norm=cfg['MODEL']['NORM']).to(self.device)
        # self.reg = UNet(num_classes=3, input_channels=2, leaky=cfg['MODEL']['LEAKY'],
        #                 norm=cfg['MODEL']['NORM']).to(self.device)
        # self.aug = TupleUNet(decoder_channels=(3, 1), input_channels=1,
        #                      leaky=cfg['MODEL']['LEAKY'], norm=cfg['MODEL']['NORM']).to(self.device)

        # tbi setting
        self.seg = UNet(num_classes=num_classes, leaky=cfg['MODEL']['LEAKY'], norm=cfg['MODEL']['NORM']).to(self.device)
        self.reg = UNet(num_classes=3, input_channels=2, leaky=cfg['MODEL']['LEAKY'],
                        norm=cfg['MODEL']['NORM']).to(self.device)
        self.aug = UNet(num_classes=4, input_channels=1, leaky=cfg['MODEL']['LEAKY'], norm=cfg['MODEL']['NORM']).to(self.device)
        # wrap to pytorch DDP
        self.seg = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(self.seg), device_ids=[self.device])
        self.reg = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(self.reg), device_ids=[self.device])
        self.aug = DDP(nn.SyncBatchNorm.convert_sync_batchnorm(self.aug), device_ids=[self.device])
        # spatial transformer for registration
        self.dvf2ddf = DVF2DDF()
        self.im_warp = SpatialTransformer(mode='bilinear')
        self.lb_warp = SpatialTransformer(mode='nearest')
        # define loss function
        self.sobel = SobelEdge(input_dim=3, channels=num_classes).to(self.device)
        self.dice_ce = DiceCELoss(to_onehot_y=True, softmax=True)
        self.registration_criterion = RegistrationCriterion(use_label_loss=False, penalty=cfg['MODEL']['PENALTY'])
        # define optimizer and scheduler, note that we have two networks
        self.seg_optimizer = torch.optim.SGD(self.seg.parameters(), lr=cfg['TRAIN']['LR'],
                                             weight_decay=cfg['TRAIN']['DECAY'], momentum=cfg['TRAIN']['MOMENTUM'])
        self.reg_optimizer = torch.optim.SGD(self.reg.parameters(), lr=cfg['TRAIN']['LR'],
                                             weight_decay=cfg['TRAIN']['DECAY'], momentum=cfg['TRAIN']['MOMENTUM'])
        self.aug_optimizer = torch.optim.SGD(self.aug.parameters(), lr=cfg['TRAIN']['LR'],
                                             weight_decay=cfg['TRAIN']['DECAY'], momentum=cfg['TRAIN']['MOMENTUM'])
        # scheduler
        self.seg_scheduler = CosineAnnealingWithWarmUp(self.seg_optimizer, total_steps=cfg['TRAIN']['EPOCHS'],
                                                    max_lr_steps=cfg['TRAIN']['BURN'],
                                                    warmup_steps=cfg['TRAIN']['BURN_IN'])
        self.reg_scheduler = CosineAnnealingWithWarmUp(self.reg_optimizer, total_steps=cfg['TRAIN']['EPOCHS'],
                                                    max_lr_steps=cfg['TRAIN']['BURN'],
                                                    warmup_steps=cfg['TRAIN']['BURN_IN'])
        self.aug_scheduler = CosineAnnealingWithWarmUp(self.aug_optimizer, total_steps=cfg['TRAIN']['EPOCHS'],
                                                    max_lr_steps=cfg['TRAIN']['BURN'],
                                                    warmup_steps=cfg['TRAIN']['BURN_IN'])
        # visualization
        self.visual_names = ['atlas_image', 'atlas_label', 'fixed_image', 'style_image',
                             'warped_image', 'warped_label', 'kl_div', 'warped_seg',
                             'dvf_sample', 'int_sample', 'sampled_dvf_d', 'sampled_int_d']
        self.visual_pairs = [
            {'name': 'name_l', 'type': 'Pred', 'image': 'image_l', 'mask': 'pred_l'},
            {'name': 'name_l', 'type': 'GT', 'image': 'image_l', 'mask': 'label_l'},
            {'name': 'name_u', 'type': 'Pred', 'image': 'image_u', 'mask': 'pred_u'},
        ]
        self.loss_names = ['reg_p', 'adv_consist', 'seg_consist', 'seg_loss']
        # wandb Table for val vis
        self.val_table = wandb.Table(columns=['ID'] + [pair['type'] for pair in self.visual_pairs])

    def sigmoid_rampup(self, current):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        rampup_length = self.cfg['TRAIN']['RAMPUP']
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def cosine_loss(self, inputs, targets, dim=1, mode='G'):
        if mode == 'G':
            return F.cosine_similarity(F.softmax(inputs, dim=dim), F.softmax(targets, dim=dim), dim=dim).mean()
        elif mode == 'D':
            return (1 - F.cosine_similarity(F.softmax(inputs, dim=dim), F.softmax(targets, dim=dim), dim=dim)).mean()
        else:
            raise NotImplementedError

    def initialize_metric_meter(self, class_list):
        """
        Define training & validation loss/metric logger
        Args:
            class_list: the classes to be segmented
        """
        self.class_list = class_list
        self.metric_meter = MetricMeter(metrics=['dice'], class_names=class_list, subject_names=['fixed_name'])
        self.train_loss = MetricMeter(metrics=self.loss_names, class_names=['train'])
        self.val_loss = MetricMeter(metrics=['loss'], class_names=['val'])

    def set_input(self, batch_data):
        """
        Set input data during training
        Args:
            batch_data_labeled: labeled data batch, expects a dict
            batch_data_unlabeled: unlabeled data batch, expects a dict
        """
        # atlas subject
        self.atlas_image = batch_data['atlas_image'].to(self.device)
        self.atlas_label = batch_data['atlas_label'].to(self.device)
        self.atlas_name = batch_data['atlas_name']
        # fixed subject for registration
        self.fixed_image = batch_data['fixed_image'].to(self.device)
        self.fixed_name = batch_data['fixed_name']
        # # style subject for intensity transform
        self.style_image = batch_data['style_image'].to(self.device)
        self.style_name = batch_data['style_name']

    def set_test_input(self, batch_data):
        """
        Set input data during testing
        Args:
            batch_data_labeled: labeled data batch, expects a dict
        """
        self.fixed_image = batch_data['fixed_image'].to(self.device)
        # only during evaluation, fixed image has label
        self.fixed_label = batch_data['fixed_label'].to(self.device)
        self.fixed_name = batch_data['fixed_name']

    def infer(self):
        """
        Perform basic forward process during inference
        Returns:
            out: dict containing network outputs
        """
        self.fixed_pred = self.seg(self.fixed_image)

    def optimize_parameters(self, epoch_idx):
        tau = self.sigmoid_rampup(epoch_idx)
        # TODO: check all the detach operations so that we won't hurt performance
        # 1. freeze the segmentation and sample, train the registration
        self.seg.module.freeze()
        self.aug.module.freeze()
        self.reg.module.unfreeze()
        with autocast(enabled=self.is_mixed):
            dvf_p = self.reg(torch.cat([self.atlas_image, self.fixed_image], dim=1))
            ddf_p = self.dvf2ddf(dvf_p)
            self.warped_image = self.im_warp(self.atlas_image, ddf_p)  # atlas to fixed (image)
            self.warped_label = self.lb_warp(self.atlas_label, ddf_p)  # atlas to fixed (label)
            self.reg_p = self.registration_criterion(self.fixed_image, self.warped_image, ddf_p)
            reg_loss = self.cfg['LOSS']['REG_RATIO'] * self.reg_p
        self.reg_optimizer.zero_grad()
        if self.is_mixed:
            # backward
            self.scaler.scale(reg_loss).backward()
            self.scaler.step(self.reg_optimizer)
            self.scaler.update()
        else:
            reg_loss.backward()
            self.reg_optimizer.step()

        # 2. generate adversarial deformed examples to 'fool' the segmentation network
        self.seg.module.freeze()
        self.aug.module.unfreeze()
        self.reg.module.freeze()
        with autocast(enabled=self.is_mixed):
            # invert the style image for style information
            dvf_style = self.reg(torch.cat([self.style_image, self.atlas_image], dim=1))
            self.inverse_style_image = self.im_warp(self.style_image, self.dvf2ddf(dvf_style)).detach()
            # generate spatial flow and intensity field
            self.spatial_flow = dvf_p.detach()
            self.spatial_displacement = self.dvf2ddf(self.spatial_flow)
            self.intensity_field = self.inverse_style_image - self.atlas_image
            # augmentation
            self.vanilla_augmented = self.im_warp(self.inverse_style_image, self.spatial_displacement)
            # adversarial augmentation, sample the flow
            augmented_g = self.aug(self.vanilla_augmented)
            dvf_sample = torch.sigmoid(augmented_g[:, 0:3, ...])
            int_sample = torch.sigmoid(augmented_g[:, 3:4, ...])
            # TODO: we can choose to either enhance or weaken the transform in each location
            sampled_dvf_g = dvf_sample * self.spatial_flow
            sampled_int_g = int_sample * self.intensity_field
            sampled_augmented_g = self.im_warp((self.atlas_image + sampled_int_g), self.dvf2ddf(sampled_dvf_g))
            # Adversarial training: maximize the difference between augmented pred and sampled pred
            vanilla_aug_pred = self.seg(self.vanilla_augmented)
            sampled_aug_pred_g = self.seg(sampled_augmented_g)
            # convert to the atlas space for loss calculation
            vanilla_aug_pred_as = self.im_warp(vanilla_aug_pred, self.dvf2ddf(-self.spatial_flow))
            sampled_aug_pred_g_as = self.im_warp(sampled_aug_pred_g, self.dvf2ddf(-sampled_dvf_g))
            # calculate losses
            # i) adversarial loss on images
            im_adv = self.cosine_loss(vanilla_aug_pred_as, sampled_aug_pred_g_as, mode='G')
            # ii) adversarial loss on edges
            edge_adv = self.cosine_loss(self.sobel(vanilla_aug_pred_as), self.sobel(sampled_aug_pred_g_as), mode='G')
            self.adv_consist = im_adv + 0.5 * edge_adv
            adversarial_loss = self.cfg['LOSS']['ADV_CON_RATIO'] * tau * self.adv_consist
        self.aug_optimizer.zero_grad()
        if self.is_mixed:
            # backward
            self.scaler.scale(adversarial_loss).backward()
            self.scaler.step(self.aug_optimizer)
            self.scaler.update()
        else:
            adversarial_loss.backward()
            self.aug_optimizer.step()

        # 3. freeze the registration and sample layer, train the segmentation
        self.seg.module.unfreeze()
        self.aug.module.freeze()
        self.reg.module.freeze()
        # self.sample.module.freeze()
        # we optimize the segmentation model in two ways
        with autocast(enabled=self.is_mixed):
            # 1) Adversarial training: minimize the segmentation difference between vanilla image and adversarial image
            # TIP: here we sample again after augmentation network update
            with torch.no_grad():
                augmented_d = self.aug(self.vanilla_augmented)
                self.dvf_sample = torch.sigmoid(augmented_d[:, 0:3, ...])
                self.int_sample = torch.sigmoid(augmented_d[:, 3:4, ...])
                self.sampled_dvf_d = self.dvf_sample * self.spatial_flow
                self.sampled_int_d = self.int_sample * self.intensity_field
            self.sampled_augmented = self.im_warp((self.atlas_image + self.sampled_int_d), self.dvf2ddf(self.sampled_dvf_d))
            sampled_warped_image = self.im_warp(self.atlas_image, self.dvf2ddf(self.sampled_dvf_d))
            sampled_warped_label = self.lb_warp(self.atlas_label, self.dvf2ddf(self.sampled_dvf_d))
            # perform segmentation, we expect to minimize difference
            sampled_aug_pred_d = self.seg(self.sampled_augmented)
            # convert to the atlas space for loss calculation
            sampled_aug_pred_d_as = self.im_warp(sampled_aug_pred_d, self.dvf2ddf(-self.sampled_dvf_d))
            # i) image adversarial loss
            im_cons = self.cosine_loss(vanilla_aug_pred_as, sampled_aug_pred_d_as, mode='D')
            # ii) edge adversarial loss
            edge_cons = self.cosine_loss(self.sobel(vanilla_aug_pred_as), self.sobel(sampled_aug_pred_d_as), mode='D')
            self.seg_consist = im_cons + 0.5 * edge_cons
            # 2) uncertainty rectified segmentation loss
            # a. loss for vanilla augmented
            warped_seg = self.seg(self.warped_image.detach())
            kl_div = F.kl_div(F.log_softmax(vanilla_aug_pred, dim=1), F.softmax(warped_seg, dim=1),
                              reduction='none').sum(dim=1)
            vanilla_loss = F.cross_entropy(vanilla_aug_pred, self.warped_label.detach().squeeze(1).long(), reduction='none')
            warped_loss = self.dice_ce(warped_seg, self.warped_label.detach())
            # b. loss for sampled augmented
            sampled_warped_seg = self.seg(sampled_warped_image.detach())
            sampled_kl_div = F.kl_div(F.log_softmax(sampled_aug_pred_d, dim=1), F.softmax(sampled_warped_seg, dim=1),
                                      reduction='none').sum(dim=1)
            sampled_vanilla_loss = F.cross_entropy(sampled_aug_pred_d, sampled_warped_label.detach().squeeze(1).long(),
                                                   reduction='none')
            sampled_warped_loss = self.dice_ce(sampled_warped_seg, sampled_warped_label.detach())
            # TIP: the final segmentation loss consists of two items
            # a. structure loss for pure warped image, accurate all the time
            structure_loss = warped_loss + sampled_warped_loss
            # b. rectified loss, not accurate at the beginning, so we need a ramp-up
            rseg_loss = (vanilla_loss * torch.exp(-kl_div) + 0.005 * kl_div).mean() + \
                        (sampled_vanilla_loss * torch.exp(-sampled_kl_div) + 0.005 * sampled_kl_div).mean()
            self.seg_loss = structure_loss + 0.5 * tau * rseg_loss
            seg_loss = self.cfg['LOSS']['SEG_RATIO'] * self.seg_loss + \
                       self.cfg['LOSS']['ADV_DIS_RATIO'] * tau * self.seg_consist
        self.seg_optimizer.zero_grad()
        if self.is_mixed:
            # backward
            self.scaler.scale(seg_loss).backward()
            self.scaler.step(self.seg_optimizer)
            self.scaler.update()
        else:
            seg_loss.backward()
            self.seg_optimizer.step()

        # Prepare extra visualizations
        self.vanilla_aug_seg = torch.argmax(vanilla_aug_pred, dim=1, keepdim=True)
        self.sampled_aug_seg_d = torch.argmax(sampled_aug_pred_d, dim=1, keepdim=True)
        self.warped_seg = torch.argmax(warped_seg, dim=1, keepdim=True)
        self.kl_div = kl_div.unsqueeze(1)

    @torch.inference_mode()
    def evaluate_one_step(self, save2disk=True, save_dir=None, affine_matrix=None, cal_metric=True):
        """
        Evaluation pass on one batch
        Args:
            save2disk: bool, save the predictions to disk
            save_dir: save path
            affine_matrix: save affine matrix
            ignore_background: bool, whether to ignore background during evaluation
        """
        self.infer()
        predictions = self.one_hot(torch.argmax(self.fixed_pred, dim=1, keepdim=True))
        if cal_metric:
            multi_loss = F.cross_entropy(self.fixed_pred, self.fixed_label.squeeze(1).long())
            self.val_loss.update({'val_loss': multi_loss.item()})
            ground_truth = self.one_hot(self.fixed_label)
            # compute metrics
            metric = {'fixed_name': self.fixed_name}
            for index, cls in enumerate(self.class_list):
                dice = calculate_Dice_score(y_pred=predictions[:, index:index + 1, ...],
                                            y=ground_truth[:, index:index + 1, ...]).cpu().numpy().tolist()
                metric['{}_dice'.format(cls)] = dice
            metric_list = gather_object_across_processes(metric)
            self.metric_meter.update(metric_list)
        if save2disk:
            batch_pred2save = torch.argmax(predictions, dim=1)  # (N, H, W, D)
            for i in range(batch_pred2save.shape[0]):
                pred2save = batch_pred2save[i, ...]
                data = pred2save.cpu().numpy().astype(np.float32)
                nib.save(nib.Nifti1Image(data, affine=affine_matrix), os.path.join(save_dir, self.fixed_name[i]))

    def save_intermediate_plots(self, epoch_idx, visualization_dir, affine_matrix):
        """
        Save intermediate results to disk
        """
        for name in self.visual_names:
            try:
                item = getattr(self, name)
            except:
                continue
            for i in range(item.shape[0]):
                item2save = item[i, ...]
                # detach the tensor and convert to channel last
                data = item2save.detach().permute(1, 2, 3, 0).squeeze().cpu().numpy().astype(np.float32)
                nib.save(nib.Nifti1Image(data, affine=affine_matrix),
                         os.path.join(visualization_dir,
                                      'Epoch_{}_Type_{}_{}'.format(epoch_idx, name, self.fixed_name[i])))
