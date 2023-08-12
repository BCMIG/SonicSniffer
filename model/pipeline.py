import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
import wandb


class SonicSniffer(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, pos_weight, fused):
        super().__init__()
        # save_hyperparameters writes args to hparams attribute, also used by load_from_checkpoint, so if we ignore "stunet", module can't be initialized
        # so we save passed modules in save_hyperparameters without writing to file
        self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters(logger=False)

        self.lr = lr
        self.weight_decay = weight_decay
        self.fused = fused

        self.model = model
        # self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        if self.current_epoch % 10 == 0:
            # log images (wandb)
            self.logger.experiment.log(
                {
                    "train_x": [wandb.Image(i) for i in x],
                    "train_y": [wandb.Image(i) for i in y],
                    "train_y_hat": [wandb.Image(i) for i in y_hat],
                }
            )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)
        if self.current_epoch % 10 == 0:
            # log images (wandb)
            self.logger.experiment.log(
                {
                    "val_x": [wandb.Image(i) for i in x],
                    "val_y": [wandb.Image(i) for i in y],
                    "val_y_hat": [wandb.Image(i) for i in y_hat],
                }
            )
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            fused=self.fused,
        )

        return {"optimizer": optimizer}
