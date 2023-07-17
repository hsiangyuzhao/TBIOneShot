import os
from utils.iteration.load_data_v2 import simple_affine, tbi_affine
from utils.ddp_utils import get_world_size


def prepare_experiment(task):
    if task == 'tbi':
        atlas_root = 'dataset/templates/'
        labeled_root = 'dataset/sTBI_labeled/'
        unlabeled_root = 'dataset/sTBI_unlabeled/'
        num_classes = 18
        class_names = ['bg'] + ['br{}'.format(i) for i in range(1, 18)]
        affine = tbi_affine
        return atlas_root, labeled_root, unlabeled_root, num_classes, class_names, affine
    else:
        raise NotImplementedError


def makedirs(full_exp_name, task, save_root_path):
    save_dir = os.path.join(save_root_path, 'checkpoints/{}/{}'.format(task, full_exp_name))
    os.makedirs(save_dir, exist_ok=True)
    metric_savedir = os.path.join(save_root_path, 'metrics/{}/{}'.format(task, full_exp_name))
    os.makedirs(metric_savedir, exist_ok=True)
    infer_save_dir = os.path.join(save_root_path, 'inference_display/{}/{}'.format(task, full_exp_name))
    os.makedirs(infer_save_dir, exist_ok=True)
    vis_save_dir = os.path.join(save_root_path, 'visualization/{}/{}'.format(task, full_exp_name))
    os.makedirs(vis_save_dir, exist_ok=True)
    return save_dir, metric_savedir, infer_save_dir, vis_save_dir


def update_config_file(cfg):
    cfg['TRAIN']['BURN_IN'] = cfg['TRAIN']['BURN_IN'] * get_world_size()
    cfg['TRAIN']['BURN'] = cfg['TRAIN']['BURN'] * get_world_size()
    cfg['TRAIN']['RAMPUP'] = cfg['TRAIN']['RAMPUP'] * get_world_size()
    cfg['TRAIN']['EPOCHS'] = cfg['TRAIN']['EPOCHS'] * get_world_size()
    cfg['PROJECT'] = 'AtlasSeg'
    return cfg
