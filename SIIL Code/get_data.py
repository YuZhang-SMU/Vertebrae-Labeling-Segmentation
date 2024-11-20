import os
import random
from typing import Optional

import torch.utils.data as Data
import torchio as tio

class MySubject(tio.Subject):
     def check_consistent_attribute(
         self,
         attribute: str,
         relative_tolerance: float = 1e-5, # note the different defaults here
         absolute_tolerance: float = 1e-5,
         message: Optional[str] = None,
         ) -> None:
         return super().check_consistent_attribute(attribute, relative_tolerance, absolute_tolerance, message)

class myImageFolder_train_val(Data.Dataset):
    def __init__(self, names, Config):
        """Initializes image paths and preprocessing module."""
        self.subjects = []

        for name in names:
            subject = MySubject(
                image=tio.ScalarImage(os.path.join(Config.data_path, 'img', name)),
                label=tio.LabelMap(os.path.join(Config.data_path, 'msk', name)),
                p_map=tio.Image(os.path.join(Config.data_path, 'pap', name), type=tio.SAMPLING_MAP)
            )
            self.subjects.append(subject)

        self.transforms = tio.Compose([
            tio.OneOf({
                tio.RandomNoise(p=0.1): 0.2,
                tio.RandomAffine(scales=(0.9, 1.1), degrees=10): 0.3,
                tio.RandomElasticDeformation(): 0.3,
            }, p=0.3),
        ])

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

        self.queue = tio.Queue(
            subjects_dataset=self.training_set,
            max_length=8,
            samples_per_volume=2,
            sampler=tio.WeightedSampler((64, 64, 128), probability_map='p_map'),
			num_workers=8,
            shuffle_subjects=True,
            shuffle_patches=True,
        )

class myImageFolder_test(Data.Dataset):
    def __init__(self, names, Config):
        self.subjects = []

        for name in names:
            subject=MySubject(
                image=tio.ScalarImage(os.path.join(Config.data_path, 'img', name)),
                label=tio.LabelMap(os.path.join(Config.data_path, 'msk', name)),
            )

            self.subjects.append(subject)

def get_data_train(Config):
    with open(os.path.join(Config.data_path, 'training.txt'), 'r', encoding='utf-8') as f:
        patient_names = f.read().splitlines()
        f.close()
    Config.total_train_number = len(patient_names)
    print('train number:', Config.total_train_number)

    # 数据打包
    train_dataset = myImageFolder_train_val(patient_names, Config)
    train_loader = Data.DataLoader(train_dataset.queue,
                                   batch_size=int(Config.BATCH_SIZE),
                                   shuffle=True,
                                   num_workers=0,
                                   pin_memory=True)
    print('--------------  Load data Successful  --------------')
    return train_loader


def get_data_test(Config):
    with open(os.path.join(Config.data_path, 'testing.txt'), 'r', encoding='utf-8') as f:
        patient_names = f.read().splitlines()
        f.close()
    random.shuffle(patient_names)
    test_dataset = myImageFolder_test(patient_names, Config)
    Config.total_test_number = len(patient_names)
    print(Config.DATASET, 'test number:', Config.total_test_number)
    print('--------------  Load data Successful  --------------')
    return test_dataset