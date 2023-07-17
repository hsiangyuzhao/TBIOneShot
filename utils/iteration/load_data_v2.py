import os
import copy
import random
import numpy as np
from monai.data import Dataset, CacheDataset


simple_affine = np.eye(4, 4)
tbi_affine = np.array([[-1., 0., 0., 90.], [0., 1., 0., -126.], [0., 0., 1., -72.], [0., 0., 0., 1.]])


class AtlasSegDataPipeline:
    def __init__(self, atlas_root: str, labeled_root: str, unlabeled_root: str, num_pairs: int, triplet: bool = True,
                 random_seed: int = 20000512, number=None):
        self.atlas_root = atlas_root
        self.labeled_root = labeled_root
        self.unlabeled_root = unlabeled_root
        self.num_pairs = num_pairs
        self.random_seed = random_seed
        self.prepare_train_val_pairs(triplet, number)

    def prepare_train_val_pairs(self, triplet, number=None):
        self.atlas_subjects = self.get_subjects(self.atlas_root, labeled=True, has_edge=False)
        self.labeled_subjects = self.get_subjects(self.labeled_root, labeled=True)
        self.unlabeled_subjects = self.get_subjects(self.unlabeled_root, labeled=False, number=number)
        self.training_pairs = self.get_training_pairs(self.atlas_subjects, self.unlabeled_subjects, triplet=triplet)
        self.validation_pairs = self.get_validation_subjects(self.labeled_subjects)
        print('Data pairs prepared. Number of training pairs: {}; Number of validation pairs: {}.'.format(
            len(self.training_pairs), len(self.validation_pairs)))

    def get_subjects(self, filepath, labeled=False, has_edge=False, number=None):
        subjects = []
        image_list = os.listdir(os.path.join(filepath, 'images'))
        if labeled:
            label_list = os.listdir(os.path.join(filepath, 'labels'))
            assert image_list == label_list, 'The images and labels of subjects should match'
        if has_edge:
            edge_list = os.listdir(os.path.join(filepath, 'edges'))
            assert image_list == edge_list, 'The images and labels of subjects should match'
        for index, filename in enumerate(image_list):
            subject = {
                'image': os.path.join(filepath, 'images', filename),
                'name': filename
            }
            if labeled:
                subject['label'] = os.path.join(filepath, 'labels', filename)
            if has_edge:
                subject['edge'] = os.path.join(filepath, 'edges', filename)
            subjects.append(subject)
        if number:
            subjects = random.sample(subjects, number)
        return subjects

    def get_subject_pair(self, atlas_subject, fixed_subject):
        pair_data = {}
        for key, value in atlas_subject.items():
            pair_data['atlas_{}'.format(key)] = value
        for key, value in fixed_subject.items():
            pair_data['fixed_{}'.format(key)] = value
        return pair_data

    def get_triplet_pair(self, atlas_subject, fixed_subject, style_subject):
        pair_data = {}
        for key, value in atlas_subject.items():
            pair_data['atlas_{}'.format(key)] = value
        for key, value in fixed_subject.items():
            pair_data['fixed_{}'.format(key)] = value
        for key, value in style_subject.items():
            pair_data['style_{}'.format(key)] = value
        return pair_data

    # def get_training_pairs(self, atlas_subjects, fixed_subjects, triplet=False):
    #     pairs = []
    #     if triplet:
    #         style_subjects = copy.deepcopy(fixed_subjects)
    #         random.seed(self.random_seed)
    #         random.shuffle(style_subjects)
    #         for atlas in atlas_subjects:
    #             for fixed in fixed_subjects:
    #                 for style in style_subjects:
    #                     pairs.append(self.get_triplet_pair(atlas, fixed, style))
    #     else:
    #         for atlas in atlas_subjects:
    #             for fixed in fixed_subjects:
    #                 pairs.append(self.get_subject_pair(atlas, fixed))
    #     if len(pairs) > self.num_pair:
    #         pairs = random.sample(pairs, self.num_pair)
    #     return pairs

    def get_training_pairs(self, atlas_subjects, fixed_subjects, triplet=False):
        pairs = []
        while len(pairs) < self.num_pairs:
            atlas = random.choice(atlas_subjects)
            if triplet:
                fixed, style = random.sample(fixed_subjects, 2)
                pairs.append(self.get_triplet_pair(atlas, fixed, style))
            else:
                fixed = random.choice(fixed_subjects)
                pairs.append(self.get_subject_pair(atlas, fixed))
        return pairs

    def get_validation_subjects(self, fixed_subjects):
        subjects = []
        for fixed_subject in fixed_subjects:
            pair_data = {}
            for key, value in fixed_subject.items():
                pair_data['fixed_{}'.format(key)] = value
            subjects.append(pair_data)
        return subjects

    def get_dataset(self, train_transform, val_transform, cache_dataset=False):
        dataset = CacheDataset if cache_dataset else Dataset
        trainset = dataset(data=self.training_pairs, transform=train_transform)
        valset = dataset(data=self.validation_pairs, transform=val_transform)
        return trainset, valset
