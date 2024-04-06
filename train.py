import os

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import segmentation_models_pytorch_3d as smp
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
from skimage import io
from pprint import pprint

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Dataset class
class CellDataset(Dataset):
    def __init__(self, df, split):
        self.df = df[df.split == split]
        self.df = self.df.reset_index(drop=True)
        self.images_paths = self.df.path.to_list()
        self.filenames = [os.path.basename(path) for path in self.df.path.to_list()]

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        # Read image
        img_path = self.images_paths[index]
        img = io.imread(img_path)
        img = img.astype(np.float32)
        # Normalize
        img /= img.max()
        img = torch.tensor(img)
        img = img.unsqueeze(0)

        # Read mask
        msk_path = img_path.replace('image', 'mask')
        msk = io.imread(msk_path)
        msk = msk.astype(np.float32)
        msk = torch.tensor(msk, dtype=torch.long)
        msk = F.one_hot(msk, 2)
        msk = msk.permute((3, 0, 1, 2))

        data = dict()
        data['image'] = img
        data['mask'] = msk
        return data


# Model class
class CellModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        
        # preprocessing parameteres for image
        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

    def forward(self, image):
        # normalize image here
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        image = batch['image']

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[-2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch['mask']
        # Check that mask values in between 0 and 1, NOT 0 and 255
        assert mask.max() <= 1.0 and mask.min() >= 0

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), num_classes=2, mode='multiclass')

        return {
            'loss': loss,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction='micro-imagewise')

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with 'empty' images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')

        metrics = {
            f'{stage}_per_image_iou': per_image_iou,
            f'{stage}_dataset_iou': dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'valid')

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, 'valid')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)




def main(base_dir, model_name, model_encoder, batch_size, load_path='', load=False):
    # init train, val, test sets
    data_df = pd.read_csv(os.path.join(base_dir, 'data.csv'))
    train_dataset = CellDataset(data_df, 'train')
    valid_dataset = CellDataset(data_df, 'val')
    test_dataset = CellDataset(data_df, 'test')

    # It is a good practice to check datasets don`t intersects with each other
    assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
    assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
    assert set(train_dataset.filenames).isdisjoint(
        set(valid_dataset.filenames))

    print(f'Train size: {len(train_dataset)}')
    print(f'Valid size: {len(valid_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)


    # Model
    model = CellModel(model_name, model_encoder,
                          in_channels=1, out_classes=2)

    if load:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded successfully')
        
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_dataset_iou',
        dirpath=base_dir,
        filename=model_name+'-'+model_encoder+'-epoch{epoch:03d}-{valid_dataset_iou:.4f}',
        save_top_k=2,
        mode='max',
        auto_insert_metric_name=False
        )
    
    # Training
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    # Validation and test metrics
    # run validation dataset
    valid_metrics = trainer.validate(
        model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)

    # run test dataset
    test_metrics = trainer.test(
        model, dataloaders=test_dataloader, verbose=False)
    pprint(test_metrics)



if __name__ == '__main__':
    base_dir = os.path.join('data', 'Fluo-N3DH-SIM+_splitted')
    # base_dir = os.path.join('content', 'MyDrive', 'Colab Notebooks', '3D Segmentation', 'Fluo-N3DH-SIM+_splitted')
    base_dir = "/content/drive/MyDrive/Colab Notebooks/3D segmentation/Fluo-N3DH-SIM+_splitted_filtered"
    model_name = 'Unet'
    model_encoder = 'resnet18'
    batch_size = 8
    load = False
    load_path = ''
    main(base_dir, model_name, model_encoder, batch_size, load_path, load)
