import torch
import pytorch_lightning as pl
import segmentation_models_pytorch_3d as smp

from x_unet import XUnet


# Model class
class CellModel(pl.LightningModule):
    def __init__(
        self,
        arch,
        encoder_name,
        in_channels,
        out_classes,
        activation="sigmoid",
        **kwargs,
    ):
        super().__init__()
        # self.model = smp.create_model(
        #     arch,
        #     encoder_name=encoder_name,
        #     in_channels=in_channels,
        #     classes=out_classes,
        #     activation=activation,
        #     **kwargs,
        # )

        self.model = XUnet(
            dim=64,
            # set this to greater than 1
            frame_kernel_size=3,
            channels=in_channels,
            out_dim=out_classes,
            dim_mults=(1, 2, 2, 4),
            # nested unet depths, from unet-squared paper
            # nested_unet_depths=(5, 4, 2, 1),
            # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
            consolidate_upsample_fmaps=True,
            weight_standardize=True,
        )

        # preprocessing parameteres for image
        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, image):
        # normalize image here
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[-2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]
        # Check that mask values in between 0 and 1, NOT 0 and 255
        assert mask.max() <= 1.0 and mask.min() >= 0

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        pred_mask = (logits_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with 'empty' images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)
