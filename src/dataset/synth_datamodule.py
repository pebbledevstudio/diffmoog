import os

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset.ai_synth_dataset import AiSynthDataset, NSynthDataset


class ModularSynthDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=128, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset, self.in_domain_val_dataset, self.out_of_domain_val_dataset = None, None, None
        self.num_classes = -1

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_dir = os.path.join(self.data_dir, 'train')
            self.train_dataset = AiSynthDataset(train_dir)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            val_dir = os.path.join(self.data_dir, 'val')
            self.in_domain_val_dataset = AiSynthDataset(val_dir)

            nsynth_val_dir = os.path.join(self.data_dir, 'nsynth_val')
            self.out_of_domain_val_dataset = NSynthDataset(nsynth_val_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          persistent_workers=(self.num_workers != 0), pin_memory=True)

    def val_dataloader(self):
        id_loader = DataLoader(self.in_domain_val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                               persistent_workers=(self.num_workers != 0), pin_memory=True)

        ood_loader = DataLoader(self.out_of_domain_val_dataset, batch_size=self.batch_size,
                                num_workers=self.num_workers, persistent_workers=(self.num_workers != 0),
                                pin_memory=True)

        return [id_loader, ood_loader]

    def test_dataloader(self):
        pass