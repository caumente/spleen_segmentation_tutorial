import os

import pytorch_lightning
from monai.config import print_config

from lightning_module import Net

print_config()

# initialise the LightningModule
net = Net()

# set up loggers and checkpoints
log_dir = os.path.join("./experiments")
tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=log_dir)

# initialise Lightning's trainer.
trainer = pytorch_lightning.Trainer(
    gpus=[0],
    max_epochs=30,
    logger=tb_logger,
    enable_checkpointing=True,
    num_sanity_val_steps=1,
    log_every_n_steps=1,
)

# train
trainer.fit(net)

# results
print(f"Train completed, best_metric: {net.best_val_dice:.4f} at epoch {net.best_val_epoch}")
