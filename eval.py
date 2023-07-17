import yaml
import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.transforms import *
from configs.experiment import prepare_experiment, update_config_file, makedirs
from models.segmentation_model import AtlasSeg
from utils.iteration.load_data_v2 import AtlasSegDataPipeline
from utils.iteration.iterator import set_random_seed
from utils.ddp_utils import init_distributed_mode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='tbi')
    parser.add_argument('--mixed', action="store_true",
                        help='whether to use PyTorch native mixed precision training')
    parser.add_argument('-pc', '--pretrain_ckpt', type=str, help='model checkpoint used for fine tuning')
    parser.add_argument('--benchmark', action="store_true",
                        help='whether to use cudnn benchmark to speed up convolution operations')
    parser.add_argument('--verbose', action='store_true', help='print progress bar while training')
    parser.add_argument('--exp_name', type=str, default='running', help='experiment name to save logs')
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    return parser.parse_args()


def main():
    args = parse_args()
    ngpu = torch.cuda.device_count()
    init_distributed_mode(args)
    print('-' * 30)
    print('One-Shot Atlas Based Segmentation Training')
    print('Mixed Precision - {}; CUDNN Benchmark - {}; Num GPU - {}'.format(
        args.mixed, args.benchmark, ngpu))

    # load the cfg file
    cfg_file = 'configs/tbi.cfg'
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
        print("successfully loaded config file: ", cfg)
    # update task-specific configurations
    cfg = update_config_file(cfg)
    # set important hyper-parameters
    seed = cfg['TRAIN']['SEED']  # random seed
    batch_size = cfg['TRAIN']['BATCHSIZE']  # batch size
    # define experiment name
    cfg['EXP_NAME'] = args.exp_name  # experiment name

    # set random seed for reproductivity
    set_random_seed(seed=seed, benchmark=args.benchmark)

    # define training & validation transforms
    train_transform = [
        LoadImaged(keys=['atlas_image', 'atlas_label', 'fixed_image', 'style_image'], allow_missing_keys=True),
        EnsureChannelFirstd(keys=['atlas_image', 'atlas_label', 'fixed_image', 'style_image'], allow_missing_keys=True),
        NormalizeIntensityd(keys=['atlas_image', 'fixed_image', 'style_image'], allow_missing_keys=True),
        EnsureTyped(keys=['atlas_image', 'atlas_label', 'fixed_image', 'style_image'], allow_missing_keys=True),
    ]
    val_transform = [
        LoadImaged(keys=['fixed_image', 'fixed_label']),
        EnsureChannelFirstd(keys=['fixed_image', 'fixed_label']),
        NormalizeIntensityd(keys=['fixed_image']),
        EnsureTyped(keys=['fixed_image', 'fixed_label']),
    ]
    augmentation = [
        RandHistogramShiftd(keys=['fixed_image'], allow_missing_keys=True),
        RandHistogramShiftd(keys=['style_image'], allow_missing_keys=True),
        RandGridDistortiond(keys=['fixed_image'], mode=['bilinear'], distort_limit=0.1, device=torch.device('cuda')),
        RandGridDistortiond(keys=['style_image'], mode=['bilinear'], distort_limit=0.1, device=torch.device('cuda'))
    ]
    train_transform.extend(augmentation)
    # define training & validation transforms
    train_aug = Compose(train_transform)
    val_aug = Compose(val_transform)
    save_root_path = 'experiments/AtlasSeg/Evaluation'
    # prepare the experiments
    atlas_root, labeled_root, unlabeled_root, num_classes, class_names, affine = prepare_experiment(args.task)
    save_dir, metric_savedir, infer_save_dir, vis_save_dir = makedirs(args.exp_name, args.task, save_root_path)
    # define dataset
    data_pipeline = AtlasSegDataPipeline(atlas_root, labeled_root, unlabeled_root, num_pairs=cfg['TRAIN']['DATA_NUM'])
    trainset, valset = data_pipeline.get_dataset(train_aug, val_aug, cache_dataset=False)
    val_sampler = DistributedSampler(valset)
    # define tasks-specific information
    print('Num training pairs: {}; Num validation pairs: {}'.format(len(trainset), len(valset)))
    # define devices and loaders
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, sampler=val_sampler)
    # define models
    model = AtlasSeg(cfg, num_classes=num_classes, amp=args.mixed)
    model.load_networks(args.pretrain_ckpt, resume_training=False)
    print('Checkpoint {} loaded'.format(args.pretrain_ckpt))
    # classes for evaluation
    model.initialize_metric_meter(class_names)

    print('Evaluating, plz wait...')
    model.eval()
    val_loader = tqdm.tqdm(val_loader) if args.verbose else val_loader
    for step, batch_data in enumerate(val_loader):
        model.set_test_input(batch_data)
        model.evaluate_one_step(True, infer_save_dir, affine)

    model.metric_meter.report(print_stats=True)  # print stats
    # save the metric at the end of training
    model.metric_meter.save(metric_savedir, '{}.csv'.format(args.exp_name))


if __name__ == '__main__':
    main()
