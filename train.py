import wandb
import yaml
import tqdm
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.transforms import *
from configs.experiment import prepare_experiment, update_config_file, makedirs
from models.segmentation_model import AtlasSeg
from utils.iteration.load_data_v2 import AtlasSegDataPipeline
from utils.iteration.iterator import set_random_seed
from utils.ddp_utils import init_distributed_mode
torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='tbi')
    parser.add_argument('--eval_interval', type=int, default=1, help='interval for evaluation')
    parser.add_argument('--save_interval', type=int, default=1, help='interval for visualization saving')
    parser.add_argument('--mixed', action="store_true",
                        help='whether to use PyTorch native mixed precision training')
    parser.add_argument('-pc', '--pretrain_ckpt', type=str, help='model checkpoint used for fine tuning')
    parser.add_argument('--benchmark', action="store_true",
                        help='whether to use cudnn benchmark to speed up convolution operations')
    parser.add_argument('--ignore_background', type=bool, default=True,
                        help='ignore the background during evaluation')
    parser.add_argument('--ncpu', type=int, default=16, help='number of workers for dataloader')
    parser.add_argument('--verbose', action='store_true', help='print progress bar while training')
    parser.add_argument('--exp_name', type=str, default='running', help='experiment name to save logs')
    parser.add_argument('--debug', action="store_true", help='enable debug mode')
    parser.add_argument('--wandb', action='store_true', help='use WandB for experiment logging')
    parser.add_argument('--entity', type=str, default=None, help='WandB entity when logging')
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    return parser.parse_args()


def main():
    args = parse_args()
    ngpu = torch.cuda.device_count()
    init_distributed_mode(args)
    print('-' * 30)
    print('One-Shot Atlas Based Segmentation Training')
    print('Mixed Precision - {}; CUDNN Benchmark - {}; Num GPU - {}; Num Worker - {}'.format(
        args.mixed, args.benchmark, ngpu, args.ncpu))

    # load the cfg file
    cfg_file = 'configs/brain.cfg'
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
        print("successfully loaded config file: ", cfg)
    # update task-specific configurations
    cfg = update_config_file(cfg)
    # set important hyper-parameters
    seed = cfg['TRAIN']['SEED']  # random seed
    batch_size = cfg['TRAIN']['BATCHSIZE']  # batch size
    num_epochs = cfg['TRAIN']['EPOCHS'] if not args.debug else 5  # number of epochs
    epoch_size = cfg['TRAIN']['DATA_NUM'] if not args.debug else 15  # number of iterations in 1 epoch
    num_pairs = num_epochs * epoch_size  # total number of iterations
    # define experiment name
    exp_name = '{}_{}'.format(args.task, args.exp_name)
    cfg['EXP_NAME'] = exp_name  # experiment name
    # set random seed for reproductivity
    set_random_seed(seed=seed, benchmark=args.benchmark)

    train_transform = [
        LoadImaged(keys=['atlas_image', 'atlas_label', 'fixed_image', 'style_image'], allow_missing_keys=True),
        Orientationd(keys=['atlas_image', 'atlas_label', 'fixed_image', 'style_image'], allow_missing_keys=True, as_closest_canonical=True),
        EnsureChannelFirstd(keys=['atlas_image', 'atlas_label', 'fixed_image', 'style_image'], allow_missing_keys=True),
        NormalizeIntensityd(keys=['atlas_image', 'fixed_image', 'style_image'], allow_missing_keys=True),
        EnsureTyped(keys=['atlas_image', 'atlas_label', 'fixed_image', 'style_image'], allow_missing_keys=True),
    ]
    val_transform = [
        LoadImaged(keys=['fixed_image', 'fixed_label']),
        Orientationd(keys=['atlas_image', 'atlas_label', 'fixed_image', 'style_image'], allow_missing_keys=True, as_closest_canonical=True),
        EnsureChannelFirstd(keys=['fixed_image', 'fixed_label']),
        NormalizeIntensityd(keys=['fixed_image']),
        EnsureTyped(keys=['fixed_image', 'fixed_label']),
    ]
    tbi_augmentation = [
        ResizeWithPadOrCropd(keys=['atlas_image', 'fixed_image', 'style_image'], allow_missing_keys=True, spatial_size=(176, 224, 176)),
        RandHistogramShiftd(keys=['fixed_image'], prob=0.5, allow_missing_keys=True),
        RandHistogramShiftd(keys=['style_image'], prob=0.5, allow_missing_keys=True),
        RandGridDistortiond(keys=['fixed_image'], mode=['bilinear'], prob=0.5,
                            distort_limit=0.1, device=torch.device('cuda')),
        RandGridDistortiond(keys=['style_image'], mode=['bilinear'], prob=0.5,
                            distort_limit=0.1, device=torch.device('cuda'))
    ]
    train_transform.extend(tbi_augmentation)
    # define training & validation transforms
    train_aug = Compose(train_transform)
    val_aug = Compose(val_transform)

    save_root_path = 'experiments/AtlasSeg'
    # prepare the experiments
    atlas_root, labeled_root, unlabeled_root, num_classes, class_names, affine, = prepare_experiment(args.task)
    save_dir, metric_savedir, infer_save_dir, vis_save_dir = makedirs(exp_name, args.task, save_root_path)
    # define dataset
    data_pipeline = AtlasSegDataPipeline(atlas_root, labeled_root, unlabeled_root, num_pairs=num_pairs)
    trainset, valset = data_pipeline.get_dataset(train_aug, val_aug, cache_dataset=False)
    train_sampler = DistributedSampler(trainset, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(valset)
    # define tasks-specific information
    print('Num training pairs: {}; Num validation pairs: {}'.format(len(trainset), len(valset)))

    # define devices and loaders
    train_loader = DataLoader(trainset, batch_size=batch_size,  num_workers=args.ncpu,
                              sampler=train_sampler, persistent_workers=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, sampler=val_sampler)
    # define models
    model = AtlasSeg(cfg, num_classes=num_classes, amp=args.mixed)
    if args.pretrain_ckpt:
        model.load_networks(args.pretrain_ckpt, resume_training=False)
        print('Checkpoint {} loaded'.format(args.pretrain_ckpt))
    # classes for evaluation
    model.initialize_metric_meter(class_names)

    if args.wandb and dist.get_rank() == 0:
        wandb.init(project=cfg['PROJECT'], entity=args.entity, reinit=True, name=cfg['EXP_NAME'])
        wandb.config.update(cfg)

    best_metric = 0.0
    # starts training
    print('Start training, please wait...')
    train_sampler.set_epoch(0)
    train_loader = tqdm.tqdm(train_loader) if args.verbose else train_loader
    last_epoch = 0
    print("Epoch {}/{}, current lr {}".format(last_epoch + 1, num_epochs, model.seg_optimizer.param_groups[0]['lr']))
    for step, batch_data in enumerate(train_loader):
        epoch = int((step + 1) / epoch_size)  # get current epoch index
        model.train()
        model.set_input(batch_data)
        # forward and backward
        model.optimize_parameters(epoch + 1)
        stats = model.update_loss_meter(print=False)  # update the training loss meter
        if args.verbose:
            train_loader.set_postfix_str(stats)  # set progress bar postfix
        # if this is the last iteration of current epoch
        if epoch > last_epoch:  # finish an epoch, do the visualization, evaluation and model save
            last_epoch += 1
            # update learning rate
            model.seg_scheduler.step()
            model.reg_scheduler.step()
            model.aug_scheduler.step()
            if not args.verbose:
                print('Epoch {} finished, iterations {} / {}'.format(last_epoch, step + 1, len(train_loader)))
                model.update_loss_meter(print=True)  # print epoch data
            if dist.get_rank() == 0 and args.wandb:
                model.log_train_loss(step=last_epoch)
            # save intermediate plots to disk or wandb
            if dist.get_rank() == 0 and last_epoch % args.save_interval == 0:
                # if args.wandb:
                #     model.log_vis('train_visualization', step=last_epoch)
                # else:
                model.save_intermediate_plots(last_epoch, visualization_dir=vis_save_dir, affine_matrix=affine)

            # evaluation loop, if needed
            if last_epoch % args.eval_interval == 0 or last_epoch > num_epochs - 5:
                print('Evaluating, plz wait...')
                model.eval()
                val_loader = tqdm.tqdm(val_loader) if args.verbose else val_loader
                for step, batch_data in enumerate(val_loader):
                    model.set_test_input(batch_data)
                    model.evaluate_one_step(True if last_epoch % args.save_interval == 0 else False,
                                            infer_save_dir, affine, args.task != 'tbi')
                    # uncomment following lines to log the validation visualization during inference
                    # if args.wandb:
                    #     model.update_val_visualization()

                # print stats
                model.metric_meter.report(print_stats=True, ignore_first_class=args.ignore_background)
                # save the metric to csv file
                model.metric_meter.save(metric_savedir, '{}_Epoch_{}.csv'.format(exp_name, last_epoch))
                # print current metric
                current_metric = model.metric_meter.pop_mean_metric(True)['dice']
                print('Evaluation complete. Mean Dice score: {:.4f}'.format(current_metric))
                if args.wandb and dist.get_rank() == 0:  # log data to wandb
                    model.log_val_loss(step=last_epoch)
                    # uncomment following line to log the validation visualization during inference
                    # model.log_val_visualization(step=epoch + 1)
                    model.log_scaler('val/val_metric_mean', current_metric, step=last_epoch)

                # re-initialize the metric meter every time when performing evaluation
                model.metric_meter.initialization()
                model.val_loss.initialization()
                if last_epoch % args.eval_interval == 0 or last_epoch > num_epochs - 5:
                    model.save_networks(last_epoch, save_dir)  # save checkpoints

                if current_metric > best_metric:
                    best_metric = current_metric
                    model.save_best_networks(last_epoch, save_dir)  # save checkpoints

                torch.cuda.empty_cache()

            # finish everything, print information for the next epoch
            if last_epoch < num_epochs:
                print("Epoch {}/{}, current lr {}".format(last_epoch + 1, num_epochs,
                                                          model.seg_optimizer.param_groups[0]['lr']))


if __name__ == '__main__':
    main()
