import torch
import numpy as np
import torchvision.models as models
from pytorch_lightning import LightningModule
import segmentation_models_pytorch as smp
from torchvision.models.resnet import BasicBlock as ResNetBasicBlock
# from depth_loss import BerHu
# from piqa import SSIM


class DepthToNormalsLoss(torch.nn.Module):
    def __init__(self, spacing, device='cuda'):
        super().__init__()
        kernel = np.array(
            [
                [0., -1., 0.],
                [0., 0., 0.],
                [0., 1., 0.],
            ]
        )
        self.weights0 = torch.tensor(
            kernel,
            requires_grad=False,
        ).view(1, 1, 3, 3).float().to(device) / 2
        self.weights1 = torch.tensor(
            kernel.transpose(),
            requires_grad=False,
        ).view(1, 1, 3, 3).float().to(device) / 2
        self.spacing = spacing

    def debugging(self, dpth_im):
        # dpth_im: B x 1 x W x H
        tpad = torch.nn.functional.pad(dpth_im, pad=(1, 1, 1, 1), value=0)
        dt0 = torch.nn.functional.conv2d(tpad, self.weights0) / self.spacing
        dt1 = torch.nn.functional.conv2d(tpad, self.weights1) / self.spacing  # B x 1 x W x H
        nnn = torch.hstack((-dt1, torch.ones_like(dt1), -dt0))  # it is now B x 3 x W x H
        n = torch.nn.functional.normalize(nnn, p=2, dim=1)
        return tpad, dt0, dt1, nnn, n

    def get_normals_from_depth(self, dpth_im):
        # dpth_im: B x 1 x W x H
        tpad = torch.nn.functional.pad(dpth_im, pad=(1, 1, 1, 1), value=0)
        dt0 = torch.nn.functional.conv2d(tpad, self.weights0) / self.spacing
        dt1 = torch.nn.functional.conv2d(tpad, self.weights1) / self.spacing  # B x 1 x W x H
        nnn = torch.hstack((-dt1, torch.ones_like(dt1), -dt0))  # it is now B x 3 x W x H
        n = torch.nn.functional.normalize(nnn, p=2, dim=1)
        return n

    def get_normals_image_from_depth(self, dpth_im):
        n = self.get_normals_from_depth(dpth_im=dpth_im)
        # moving to image
        n_img = n * 0.5 + 0.5
        return n_img

    def forward(self, inputs, targets):
        # inputs = depth predicitons B x W x H
        # targets = GT normals B x 3 x W x H
        normals = self.get_normals_from_depth(dpth_im=inputs)
        return torch.nn.MSELoss()(normals, targets)


class TestModel(LightningModule):
    def __init__(
            self,
            arch='UNet',
            encoder_name='mobilenet_v2',
            in_channels=3,
            out_classes=10,
            lossmode=0,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_last_features = 20
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            # encoder_depth=3,
            # encoder_weights=None,
            # decoder_use_batchnorm=False,
            # decoder_channels=(64, 32, 16),
            # decoder_attention_type=None,
            in_channels=in_channels,
            classes=self.n_last_features,
            # activation=None,
            # aux_params=None,
        )
        # self.model = Autoencoder_tiny()

        # self.depth_head = torch.nn.Sequential(
        #     torch.nn.Conv2d(10,1,3,padding=1),
        #     torch.nn.Sigmoid(),
        # )
        # self.normals_head = torch.nn.Sequential(
        #     torch.nn.Conv2d(10, 1, 3, padding=1),
        #     torch.nn.Sigmoid(),
        # )
        self.segmentation_head = self._make_head(in_features=self.n_last_features, out_features=out_classes)
        self.depth_head = self._make_head(in_features=self.n_last_features, out_features=1)
        self.normals_head = self._make_head(in_features=self.n_last_features, out_features=3)
        self.lossmode = lossmode

        # preprocessing parameteres for image
        # params = smp.encoders.get_preprocessing_params(encoder_name)
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        # self.loss_segmentation = torch.nn.CrossEntropyLoss()
        # self.loss_segmentation = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.loss_segmentation = smp.losses.FocalLoss(mode="multiclass")

    @staticmethod
    def _make_head(in_features, out_features):
        head = torch.nn.Sequential(
            ResNetBasicBlock(inplanes=in_features, planes=in_features),
            torch.nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1)
        )
        return head

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std
        features = self.model(image)
        return {
            # "depth": self.depth_head(features),
            "depth": torch.tanh(self.depth_head(features)),
            # "normals": self.normals_head(features),
            # "normals": torch.tanh(self.normals_head(features)),
            "normals": torch.tanh(self.normals_head(features)),
            # "normals": torch.nn.functional.normalize(torch.tanh(self.normals_head(features)), dim=1),
            "segmentation_logits": self.segmentation_head(features),
        }

    def shared_step(self, batch, stage):
        image = batch["image"]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        # mask = batch["mask"]
        preds = self.forward(image)
        mask = batch['classes'].long()
        # logits_mask = preds['segmentation_logits']
        # loss_segmentation = self.loss_segmentation(logits_mask, mask)
        with torch.no_grad():
            gt_foreground_mask = 1.0 - batch['mask'][:, 0].float().unsqueeze(1)
            total_foreground_pixels = torch.sum(batch['mask'][:, 1:], dim=(1, 2, 3), keepdim=True)
        preds_depth_masked = gt_foreground_mask * preds['depth']
        gt_depth_masked = gt_foreground_mask * batch['depth'].unsqueeze(1)
        depth_mse = torch.nn.MSELoss()(preds['depth'], batch['depth'].unsqueeze(1))
        depth_masked_mse = torch.nn.MSELoss().forward(
            gt_depth_masked,
            # preds['depth'],
            # batch['depth'].unsqueeze(1),
            preds_depth_masked,
        )
        with torch.no_grad():
            gt_depth_mean_const = torch.sum(gt_depth_masked, dim=(2, 3), keepdim=True) / total_foreground_pixels
            pred_depth_mean_const = torch.sum(preds_depth_masked, dim=(2,3), keepdim=True) / total_foreground_pixels
        gt_depth_dev = (gt_depth_masked - gt_depth_mean_const) * gt_foreground_mask
        depth_masked_dev = torch.nn.MSELoss().forward(
            gt_depth_dev,
            preds_depth_masked,
        )
        depth_masked_mean = torch.nn.MSELoss().forward(
            torch.mean(gt_depth_masked, dim=(2, 3), keepdim=False),
            torch.mean(preds_depth_masked, dim=(2, 3), keepdim=False),
        )
        depth_to_normals_mse_masked = torch.nn.MSELoss()(
            DepthToNormalsLoss(spacing=3.1 / 400).get_normals_from_depth(
                dpth_im=-4 * preds['depth']  # WARNING: this comes from depth pre-processing (mapping to [0,1])
            ) * gt_foreground_mask,
            batch['normals'].permute(0, 3, 1, 2) * gt_foreground_mask
        )
        sketch_lines_mask = (batch["thick_lines"] > 0.8).float().unsqueeze(1)
        depth_to_normals_mse_masked_nolines = torch.nn.MSELoss()(
            DepthToNormalsLoss(spacing=3.1 / 400).get_normals_from_depth(
                dpth_im=-4 * preds['depth']  # WARNING: this comes from depth pre-processing (mapping to [0,1])
            ) * (
                    gt_foreground_mask
            ) * (
                    1.0 - sketch_lines_mask
            ),

            batch['normals'].permute(0, 3, 1, 2) * (
                    gt_foreground_mask
            ) * (
                    1.0 - sketch_lines_mask
            )
        )
        depth_to_normals_l1_masked_nolines = torch.nn.L1Loss()(
            DepthToNormalsLoss(spacing=3.1 / 400).get_normals_from_depth(
                dpth_im=-4 * preds['depth']  # WARNING: this comes from depth pre-processing (mapping to [0,1])
            ) * (
                    gt_foreground_mask
            ) * (
                    1.0 - sketch_lines_mask
            ),

            batch['normals'].permute(0, 3, 1, 2) * (
                    gt_foreground_mask
            ) * (
                    1.0 - sketch_lines_mask
            )
        )
        normals_mse = torch.nn.MSELoss()(preds['normals'].permute(0, 2, 3, 1), batch['normals'])
        normals_masked_mse = torch.nn.MSELoss()(
            gt_foreground_mask * preds['normals'],
            # preds['normals'],
            batch['normals'].permute(0, 3, 1, 2)
        )
        with torch.no_grad():
            depth_l1 = torch.nn.L1Loss()(preds['depth'], batch['depth'].unsqueeze(1))
            normals_l1 = torch.nn.L1Loss()(preds['normals'].permute(0, 2, 3, 1), batch['normals'])
            depth_masked_l1 = torch.nn.L1Loss().forward(
                (1.0 - batch['mask'][:, 0].float().unsqueeze(1)) * preds['depth'],
                # preds['depth'],
                # batch['depth'].unsqueeze(1),
                (1.0 - batch['mask'][:, 0].float().unsqueeze(1)) * batch['depth'].unsqueeze(1),
            )
            normals_masked_l1 = torch.nn.L1Loss()(
                (1.0 - batch['mask'][:, 0].float().unsqueeze(1)) * preds['normals'],
                # preds['normals'],
                batch['normals'].permute(0, 3, 1, 2)
            )
            # segmentation_probs = torch.softmax(logits_mask, dim=1)  # probabilities
            # segmentation_probs_argmax = torch.argmax(segmentation_probs, dim=1)  # mask
            # tp, fp, fn, tn = smp.metrics.get_stats(segmentation_probs_argmax.long(), mask.long(), mode='multiclass', num_classes=10)
        # loss = depth_mse# + normals_mse
        loss = depth_mse

        if self.lossmode == 0:
            loss = depth_masked_dev + depth_to_normals_mse_masked_nolines
        if self.lossmode == 1:
            loss = depth_masked_mse
        if self.lossmode == 2:
            loss = depth_masked_dev
        if self.lossmode == 3:
            loss = depth_masked_mse + depth_to_normals_mse_masked
        if self.lossmode == 4:
            loss = depth_masked_mse + depth_to_normals_mse_masked_nolines
        if self.lossmode == 5:
            loss = depth_masked_dev + depth_to_normals_mse_masked
        if self.lossmode == 6:
            loss = depth_masked_dev + depth_to_normals_mse_masked_nolines

        return {
            "loss": loss,
            "depth_mse": depth_mse.detach().item(),
            "depth_masked_mse": depth_masked_mse.detach().item(),
            "depth_l1": depth_l1.detach().item(),
            "depth_masked_l1": depth_masked_l1.detach().item(),
            "depth_to_normals_mse_masked": depth_to_normals_mse_masked.detach().item(),
            "depth_to_normals_mse_masked_nolines": depth_to_normals_mse_masked_nolines.detach().item(),
            "depth_to_normals_l1_masked_nolines": depth_to_normals_l1_masked_nolines.detach().item(),
            "depth_masked_dev": depth_masked_dev.detach().item(),
            "depth_masked_mean": depth_masked_mean.detach().item(),
            "normals_mse": normals_mse.detach().item(),
            "normals_masked_mse": normals_masked_mse.detach().item(),
            "normals_l1": normals_l1.detach().item(),
            "normals_masked_l1": normals_masked_l1.detach().item(),
            # "segmentation_tp": tp,
            # "segmentation_fp": fp,
            # "segmentation_fn": fn,
            # "segmentation_tn": tn,

        }

    def eval_predict(self, x):
        self.eval()
        assert x.ndim == 4
        h, w = x.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        with torch.no_grad():
            preds = self.forward(x)
            logits_mask = preds['segmentation_logits']
            probability_mask = torch.softmax(logits_mask, dim=1)
            predicted_class = torch.argmax(probability_mask, dim=1)
        return {
            "depth": preds["depth"],
            "normals": preds["normals"].permute(0,2,3,1),
            "segmentation_masks": probability_mask,
            "segmentation_class": predicted_class,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        # tp = torch.cat([x["segmentation_tp"] for x in outputs])
        # fp = torch.cat([x["segmentation_fp"] for x in outputs])
        # fn = torch.cat([x["segmentation_fn"] for x in outputs])
        # tn = torch.cat([x["segmentation_tn"] for x in outputs])

        losses = np.array([x["loss"].detach().item() for x in outputs])
        mm = {
            f"{stage}_loss/{x}": np.sum([o[x] for o in outputs])
            for x in outputs[0].keys() if (x.startswith('depth')) or (x.startswith("normals"))
        }
        mm[f"{stage}_loss/loss"] = losses.sum()
        # for x in outputs[0].keys():
        #     if x.startswith("segmentation"):
        #         mm[f"{stage}_metric/{x}"] = np.mean([torch.mean(o[x]) for o in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        # per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        # mm[f"{stage}_metric/per_image_iou"] = per_image_iou
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        # dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        # mm[f"{stage}_metric/dataset_iou"] = dataset_iou

        self.log_dict(mm, prog_bar=True)

    def training_step(self, batch, batch_idx):
        self.train()
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def on_validation_epoch_start(self) -> None:
        self.n_val_dataloaders = 0

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx > self.n_val_dataloaders:
            self.n_val_dataloaders = dataloader_idx
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        if self.n_val_dataloaders == 0:
            return self.shared_epoch_end(outputs, "valid")
        else:
            self.shared_epoch_end(outputs[0], "valid")
            for i in range(1, self.n_val_dataloaders+1):
                self.shared_epoch_end(outputs[i], f"valid{i}")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
        # return torch.optim.SGD(self.parameters(), lr=1)


class SegmentationModel(LightningModule):
    def __init__(
            self,
            arch='UNet',
            encoder_name='mobilenet_v2',
            in_channels=3,
            out_classes=10,
            lossmode=0,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_last_features = 20
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            # encoder_depth=3,
            # encoder_weights=None,
            # decoder_use_batchnorm=False,
            # decoder_channels=(64, 32, 16),
            # decoder_attention_type=None,
            in_channels=in_channels,
            classes=self.n_last_features,
            # activation=None,
            # aux_params=None,
        )
        self.mode = 'multiclass'

        self.segmentation_head = self._make_head(in_features=self.n_last_features, out_features=out_classes)
        self.lossmode = lossmode

        # preprocessing parameteres for image
        # params = smp.encoders.get_preprocessing_params(encoder_name)
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.loss_fn_dice = smp.losses.DiceLoss(mode=self.mode, from_logits=True)
        self.loss_fn_focal = smp.losses.FocalLoss(mode=self.mode)
        self.loss_fn_crossentropy = torch.nn.CrossEntropyLoss()

    @staticmethod
    def _make_head(in_features, out_features):
        head = torch.nn.Sequential(
            ResNetBasicBlock(inplanes=in_features, planes=in_features),
            torch.nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1)
        )
        return head

    def forward(self, image):
        # normalize image here
        # image = (image - self.mean) / self.std
        features = self.model(image)
        return {
            "depth": torch.zeros((image.shape[0], 1, image.shape[-2], image.shape[-1])),
            "normals": torch.zeros_like(image),
            "segmentation_logits": self.segmentation_head(features),
        }

    def shared_step(self, batch, stage):
        image = batch["image"]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        # mask = batch["mask"]
        preds = self.forward(image)
        mask = batch['classes'].long()
        logits_mask = preds['segmentation_logits']
        loss_segmenation_dice = self.loss_fn_dice(logits_mask, mask)
        loss_segmenation_focal = self.loss_fn_focal(logits_mask, mask)
        loss_segmentation_ce = self.loss_fn_crossentropy(logits_mask, mask)

        with torch.no_grad():
            segmentation_probs = torch.softmax(logits_mask, dim=1)  # probabilities
            segmentation_probs_argmax = torch.argmax(segmentation_probs, dim=1)  # mask
            tp, fp, fn, tn = smp.metrics.get_stats(segmentation_probs_argmax.long(), mask.long(), mode='multiclass', num_classes=10)

        loss = loss_segmenation_dice

        if self.lossmode == 0:
            loss = loss_segmenation_dice
        if self.lossmode == 1:
            loss = loss_segmenation_focal
        if self.lossmode == 2:
            loss = loss_segmentation_ce

        return {
            "loss": loss,
            "segm_dice": loss_segmenation_dice.detach().item(),
            "segm_focal": loss_segmenation_focal.detach().item(),
            "segm_ce": loss_segmentation_ce.detach().item(),
            "segmentation_tp": tp,
            "segmentation_fp": fp,
            "segmentation_fn": fn,
            "segmentation_tn": tn,
        }

    def eval_predict(self, x):
        self.eval()
        assert x.ndim == 4
        h, w = x.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        with torch.no_grad():
            preds = self.forward(x)
            logits_mask = preds['segmentation_logits']
            probability_mask = torch.softmax(logits_mask, dim=1)
            predicted_class = torch.argmax(probability_mask, dim=1)
        return {
            "depth": preds["depth"],
            "normals": preds["normals"].permute(0,2,3,1),
            "segmentation_masks": probability_mask,
            "segmentation_class": predicted_class,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([predbatch["segmentation_tp"] for predbatch in outputs])
        fp = torch.cat([predbatch["segmentation_fp"] for predbatch in outputs])
        fn = torch.cat([predbatch["segmentation_fn"] for predbatch in outputs])
        tn = torch.cat([predbatch["segmentation_tn"] for predbatch in outputs])

        totalpixels = outputs[0]["segmentation_tp"][0, 0] \
                      + outputs[0]["segmentation_fp"][0, 0] \
                      + outputs[0]["segmentation_fn"][0, 0] \
                      + outputs[0]["segmentation_tn"][0, 0]

        losses = np.array([predbatch["loss"].detach().item() for predbatch in outputs])
        mm = {
            f"{stage}_loss/{x}": np.sum([o[x] for o in outputs])
            for x in outputs[0].keys() if (x.startswith('segm_'))
        }
        mm[f"{stage}_loss/loss"] = losses.sum()
        for metric in outputs[0].keys():
            if metric.startswith("segmentation_"):
                # outputs[0]["segmentation_fp"] has shape (batch_size, n_classes) with dtype long
                # meaning number of fp pixels for this class
                mm[f"{stage}_metric/{metric}"] = np.mean(
                    torch.cat(
                        [predbatch[metric].flatten().detach().cpu() / totalpixels for predbatch in outputs]).numpy()
                )

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        mm[f"{stage}_metric/per_image_iou"] = per_image_iou
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        mm[f"{stage}_metric/dataset_iou"] = dataset_iou

        self.log_dict(mm, prog_bar=True)

    def training_step(self, batch, batch_idx):
        self.train()
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def on_validation_epoch_start(self) -> None:
        self.n_val_dataloaders = 0

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx > self.n_val_dataloaders:
            self.n_val_dataloaders = dataloader_idx
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        if self.n_val_dataloaders == 0:
            return self.shared_epoch_end(outputs, "valid")
        else:
            self.shared_epoch_end(outputs[0], "valid")
            for i in range(1, self.n_val_dataloaders+1):
                self.shared_epoch_end(outputs[i], f"valid{i}")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
        # return torch.optim.SGD(self.parameters(), lr=1)


if __name__ == "__main__":
    a = torch.rand((7,3,512,512)).cuda()
    model = TestModel(in_channels=3, out_classes=10).cuda()
    b = model.forward(a)
    print("input - ", a.shape)
    for k in b.keys():
        print(k, " - ", b[k].shape)

    tt = b["normals"][0, :, 0, 0]
    print(tt)
    print(torch.sum(tt**2))

    DepthToNormalsLoss(spacing=3.1 / 400).get_normals_from_depth(
        dpth_im=-4 * b['depth']
    )

