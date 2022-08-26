import glob
import os

import pytorch_lightning
import torch
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
)
from monai.utils import set_determinism


data_dir = "./dataset/Task09_Spleen"


class Net(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()

        # using a 2D U-Net model
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )

        # Dice loss function
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)

        # Post-processing output to ensure they are tensors in cuda and
        # TODO: what does the AsDiscrete transform do?
        self.post_pred = Compose([EnsureType("tensor", device="cuda"), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device="cuda"), AsDiscrete(to_onehot=2)])

        # Dice will be also de metric to measure the performance
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        # initial validation values are set to 0
        self.best_val_dice = 0
        self.best_val_epoch = 0

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        """
        This function preprocess the dataset to ensure that, at the end of it, the objects self.train_ds and self.val_ds
        are defined properly.

        :return: it does not return anything, but it prepares the objects self.train_ds and self.val_ds
        """
        # set up the correct data path
        train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

        # creating dict of {image:label}
        data_dicts = [{
            "image": image_name,
            "label": label_name
        } for image_name, label_name in zip(train_images, train_labels)]

        # splitting the dataset into train & validation
        train_files, val_files = data_dicts[:-9], data_dicts[-9:]

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # define the data transforms for train set
        train_transforms = Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # randomly crop out patch samples from big image based on pos/neg ratio
                # the image centers of negative samples must be in valid image area
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                # user can also add other random transforms
                #                 RandAffined(
                #                     keys=['image', 'label'],
                #                     mode=('bilinear', 'nearest'),
                #                     prob=1.0,
                #                     spatial_size=(96, 96, 96),
                #                     rotate_range=(0, 0, np.pi/15),
                #                     scale_range=(0.1, 0.1, 0.1)),
        ])

        # define the data transforms for validation set
        val_transforms = Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
        ])

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
        self.val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)

#         self.train_ds = monai.data.Dataset(
#             data=train_files, transform=train_transforms)
#         self.val_ds = monai.data.Dataset(
#             data=val_files, transform=val_transforms)

    def train_dataloader(self) -> DataLoader:
        """
        This function creates the train dataloader from the train dataset.

        :return: train dataloader
        """

        train_loader = DataLoader(
            dataset=self.train_ds,
            batch_size=2,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        """
        This function creates the validation dataloader from the validation dataset.

        :return: validation dataloader
        """

        val_loader = DataLoader(
            dataset=self.val_ds,
            batch_size=1,
            num_workers=4
        )

        return val_loader

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        This function sets the optimizer up.

        :return: optimizer
        """

        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)

        return optimizer

    def training_step(self, batch, batch_idx) -> dict:
        """
        This function takes the batch of images-labels and feed the model. Then, the train loss error is computed and
        sent it to a tensorboard.

        :param batch: batch of images
        :param batch_idx:
        :return: dictionary that contains loss and log
        """

        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        This function takes the validation batch and feed the model by using a sliding window inference.

        """

        # generating prediction
        images, labels = batch["image"], batch["label"]
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)

        # getting validation loss error
        loss = self.loss_function(outputs, labels)

        # measuring the error
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)

        return {"val_loss": loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs: list) -> dict:
        """
        This function takes de outputs generated in the validation phase and aggregate se mean dice.

        :param outputs: list of validation loss error
        :return: log
        """

        # averaging the dice of all predicted images
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()  # reset the metric
        mean_val_loss = torch.tensor(val_loss / num_items)

        # writing metric and loss into the log
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }

        # updating the best validation metric if it is better
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        print(
            f"\n\nCurrent epoch: {self.current_epoch} --> current mean dice: {mean_val_dice:.4f}"
            f"\nBest mean dice: {self.best_val_dice:.4f} at epoch: {self.best_val_epoch}"
        )

        return {"log": tensorboard_logs}
