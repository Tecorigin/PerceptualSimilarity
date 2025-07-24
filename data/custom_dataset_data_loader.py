import torch.utils.data
from data.base_data_loader import BaseDataLoader
import torch
import torch.distributed as dist
import os

def CreateDataset(dataroots,dataset_mode='2afc',load_size=64,):
    dataset = None
    if dataset_mode=='2afc': # human judgements
        from data.dataset.twoafc_dataset import TwoAFCDataset
        dataset = TwoAFCDataset()
    elif dataset_mode=='jnd': # human judgements
        from data.dataset.jnd_dataset import JNDDataset
        dataset = JNDDataset()
    else:
        raise ValueError("Dataset Mode [%s] not recognized."%self.dataset_mode)

    dataset.initialize(dataroots,load_size=load_size)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, datafolders, dataroot='./dataset',dataset_mode='2afc',load_size=64,batch_size=1,serial_batches=True, nThreads=1):
        BaseDataLoader.initialize(self)
        if(not isinstance(datafolders,list)):
            datafolders = [datafolders,]
        data_root_folders = [os.path.join(dataroot,datafolder) for datafolder in datafolders]
        self.dataset = CreateDataset(data_root_folders,dataset_mode=dataset_mode,load_size=load_size)
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset) if dist.is_initialized() else None
        print('train_sampler:', train_sampler)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle= (train_sampler is None) and not serial_batches,
            num_workers=int(nThreads),
            sampler=train_sampler
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
