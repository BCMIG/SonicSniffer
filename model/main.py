import lightning.pytorch as pl
from pipeline import SonicSniffer
from model import get_model
from dataset import get_dataloaders
from config import get_config
from utils import FindUnusedParametersCallback
from lightning.pytorch.strategies import DDPStrategy


def main():
    logger = pl.loggers.WandbLogger(project="soundsniffer")

    cfg = get_config()
    # to ensure deterministic splits
    pl.seed_everything(cfg.seed)
    model = get_model()
    num_samples = 128
    train_loader, test_loader, val_loader = get_dataloaders(num_samples, cfg.batch_size)
    sniffer = SonicSniffer(
        num_samples, model, cfg.lr, cfg.weight_decay, cfg.pos_weight, fused=not cfg.cpu
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        precision="bf16-mixed" if cfg.mixed_precision else 32,
        logger=logger,
        log_every_n_steps=cfg.log_every_n_steps,
        strategy=DDPStrategy(find_unused_parameters=True)
        if cfg.find_unused_parameters
        else "auto",
        callbacks=FindUnusedParametersCallback()
        if cfg.find_unused_parameters
        else None,
    )
    trainer.fit(sniffer, train_loader, val_loader)


if __name__ == "__main__":
    main()