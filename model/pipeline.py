import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
import wandb


class SDFLoss(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, sdf_gt, sdf_hat):
        new_sdf_gt = torch.clamp(sdf_gt, -self.delta, self.delta) / self.delta
        l1 = torch.abs(new_sdf_gt - sdf_hat)
        return torch.mean(l1), new_sdf_gt


class SonicSniffer(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, pos_weight, delta, fused):
        super().__init__()
        # save_hyperparameters writes args to hparams attribute, also used by load_from_checkpoint, so if we ignore "stunet", module can't be initialized
        # so we save passed modules in save_hyperparameters without writing to file
        self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters(logger=False)

        self.lr = lr
        self.weight_decay = weight_decay
        self.fused = fused

        self.model = model
        print("pos_weight unused")
        # BCE instead of CE since it is a per-pixel binary classification problem
        # props to @zickzack
        self.seg_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.sdf_loss = SDFLoss(delta)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, labels, sdf = batch
        pred_seg, pred_sdf = self(img)
        seg_loss = self.seg_loss(pred_seg, labels)
        sdf_loss, sdf = self.sdf_loss(sdf, pred_sdf)
        loss = seg_loss + sdf_loss
        self.log_dict(
            {
                "train_seg_loss": seg_loss,
                "train_sdf_loss": sdf_loss,
                "train_loss": loss,
            }
        )
        if self.current_epoch % 100 == 0:
            # log images (wandb)
            self.logger.experiment.log(
                {
                    "train_x": [wandb.Image(i) for i in img],
                    "train_y": [wandb.Image(i) for i in labels],
                    "train_y_hat": [wandb.Image(i) for i in pred_seg],
                    "train_sdf": [wandb.Image(i) for i in sdf],
                    "train_sdf_hat": [wandb.Image(i) for i in pred_sdf],
                }
            )
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels, sdf = batch
        pred_seg, pred_sdf = self(img)
        seg_loss = self.seg_loss(pred_seg, labels)
        sdf_loss, sdf = self.sdf_loss(pred_sdf, sdf)
        loss = seg_loss + sdf_loss
        self.log_dict(
            {
                "val_seg_loss": seg_loss,
                "val_sdf_loss": sdf_loss,
                "val_loss": loss,
            },
            sync_dist=True,
        )
        if self.current_epoch % 10 == 0:
            # log images (wandb)
            self.logger.experiment.log(
                {
                    "val_x": [wandb.Image(i) for i in img],
                    "val_y": [wandb.Image(i) for i in labels],
                    "val_y_hat": [wandb.Image(i) for i in pred_seg],
                    "val_sdf": [wandb.Image(i) for i in sdf],
                    "val_sdf_hat": [wandb.Image(i) for i in pred_sdf],
                }
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            fused=self.fused,
        )

        return {"optimizer": optimizer}
