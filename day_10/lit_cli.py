import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = list(range(1, 100))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] * 1.0, self.data[idx] * 2.0


class BoringDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.ds = Dataset()
    
    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(1, 1)

    def forward(self, x):
        return self.l(x)


class BoringModel(pl.LightningModule):
    def __init__(self, lr=5e-4):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.criterion = torch.nn.MSELoss()
        self.model = LinearModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def training_step(self, data, *_):
        x, y = data
        x = x.float()
        y = y.float()
        o = self(x)
        loss = self.criterion(o, y)
        return loss

    def validation_step(self, data, *_):
        x, y = data
        x = x.float()
        y = y.float()
        o = self(x)
        loss = self.criterion(o, y)
        return loss


if __name__ == "__main__":
    cli = LightningCLI(BoringModel, BoringDataModule)
