from typing import Optional, Tuple
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "data/", batch_size: int = 64, train_val_test_split: Tuple[int, int, int] = (55000, 5000, 10000)):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        datasets.MNIST(self.hparams.data_dir, train=True, download=True)
        datasets.MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if not self.data_train:
            full_dataset = datasets.MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
            self.data_test = datasets.MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
            self.data_train, self.data_val = random_split(
                dataset=full_dataset,
                lengths=self.hparams.train_val_test_split[:2],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.hparams.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.hparams.batch_size, shuffle=False)