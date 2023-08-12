import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
import wandb


class SonicSniffer(pl.LightningModule):
    def __init__(self, num_samples, model, lr, weight_decay, pos_weight, fused):
        super().__init__()
        # save_hyperparameters writes args to hparams attribute, also used by load_from_checkpoint, so if we ignore "stunet", module can't be initialized
        # so we save passed modules in save_hyperparameters without writing to file
        self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters(logger=False)

        self.num_samples = num_samples
        self.lr = lr
        self.weight_decay = weight_decay
        self.fused = fused

        self.model = model
        self.pool = nn.AdaptiveAvgPool1d(num_samples)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, x):
        # batch, classes, height, width = x.shape
        output = self.model(x).logits
        # so that axis is aligned wih time
        output = rearrange(output, "b c h w -> b c (w h)")
        # batch, classes, time
        output = self.pool(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
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
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            fused=self.fused,
        )

        return {"optimizer": optimizer}
