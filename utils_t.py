import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from torchvision.transforms import v2 as T

import torchvision
import kornia.augmentation as K

import lightning as pl
from transformers import (
    AutoProcessor,
    
    AutoModelForZeroShotImageClassification,
    CLIPModel,
    CLIPProcessor,
    CLIPProcessor,
)
from PIL import Image
import copy
import os
import subprocess
from torch.utils.data import Subset
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint



clip_model = "openai/clip-vit-base-patch32"
processor: CLIPProcessor = AutoProcessor.from_pretrained(
    clip_model, use_scale_data=False, do_normalize=False, do_rescale=True
)  # try patch 14
clip: CLIPModel = AutoModelForZeroShotImageClassification.from_pretrained(clip_model)
vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
encoder = copy.deepcopy(vgg19.features[:21])
for param in encoder.parameters():
    param.requires_grad = False
for param in clip.parameters():
    param.requires_grad = False
clip.eval()
encoder.eval()
clip.requires_grad_(False)
encoder.requires_grad_(False)



class DirectionalCLIPLoss(nn.Module):
    def __init__(
        self,
        clip_model: CLIPModel,
        processor: CLIPProcessor,
        patch_transform,
        threshold: float = 0.7,
    ):
        """
        patch_transform: transform to be applied to images to get patches for patch loss it should take a batch of images and return a batch of n patches for each image i.e shape (B,3,H,W) -> (B,n,3,224,224)
        it is applied to get patches from generated images
        """
        super(DirectionalCLIPLoss, self).__init__()
        self.clip_model = clip_model
        self.processor = processor
        self.threshold = threshold
        self.style_features = None
        self.text_features = None
        self.diff_text = None

        self.patch_transform = patch_transform

    def forward(self, real_images, generated_images):
        """
        computes directional CLIP loss between real_images and generated_images given style text
        real_images: tensor of shape (B,3,H,W)
        generated_images: tensor of shape (B,3,H,W)
        style: str describing the style (e.g. "a sketch", "a painting", etc.)
        threshold: float, losses below this value are ignored (set to 0)
        returns: loss_direction, loss_patch
        NOTE: images should be preprocessed to match CLIP input requirements
        """

        diff_text = self.diff_text.repeat(real_images.shape[0], 1)
        diff_text = diff_text.to(real_images.device)

        real_images_features = self.clip_model.get_image_features(
            F.interpolate(real_images, size=224)
        )
        generated_images_features = self.clip_model.get_image_features(
            F.interpolate(generated_images, size=224)
        )
        real_images_features = real_images_features / (
            real_images_features.clone().norm(dim=-1, keepdim=True) + 1e-8
        )
        generated_images_features = generated_images_features / (
            generated_images_features.clone().norm(dim=-1, keepdim=True) + 1e-8
        )

        diff_img = generated_images_features - real_images_features
        diff_img = diff_img / (diff_img.clone().norm(dim=-1, keepdim=True) + 1e-8)

        loss_direction = (
            1 - torch.cosine_similarity(diff_img, diff_text, dim=-1)
        ).mean()
        patches = self.patch_transform(generated_images)
        patches = patches.to(generated_images.device)
        diff_text = self.diff_text.to(generated_images.device)
        diff_text = diff_text.repeat(patches.shape[1], 1)

        # Vectorized patch processing
        n_patches_features = self.clip_model.get_image_features(
            patches.view(-1, *patches.shape[2:])
        )
        n_patches_features = n_patches_features.view(
            patches.shape[0], patches.shape[1], -1
        )
        n_patches_features = n_patches_features / (
            n_patches_features.clone().norm(dim=-1, keepdim=True) + 1e-8
        )

        real_images_features_expanded = real_images_features.unsqueeze(1)
        diff_img_patch = n_patches_features - real_images_features_expanded
        diff_img_patch = diff_img_patch / (
            diff_img_patch.clone().norm(dim=-1, keepdim=True) + 1e-8
        )

        loss_patch = 1 - torch.cosine_similarity(
            diff_img_patch, diff_text.unsqueeze(0), dim=-1
        )
        mask = loss_patch > self.threshold
        loss_patch = loss_patch * mask.float()
        loss_patch_total = loss_patch.mean()

        loss_patch_total = loss_patch_total / patches.shape[0]
        return loss_direction, loss_patch_total

    def cache(self, style_text):
        """
        Precomputes and caches the text embeddings for a given style text.
        style_text: str describing the style (e.g. "a sketch", "a painting", etc.)
        """
        with torch.no_grad():
            emb = self.processor(
                text=["photo", style_text], return_tensors="pt", padding=True
            )
            emb = {
                k: v.to(next(self.clip_model.parameters()).device)
                for k, v in emb.items()
            }
            text_features = self.clip_model.get_text_features(**emb)
            text_features = text_features.detach()
            self.style_features = text_features[1, :].unsqueeze(0)
            self.text_features = text_features[0, :].unsqueeze(0)
            self.style_features /= self.style_features.norm(dim=-1, keepdim=True) + 1e-8
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True) + 1e-8
            self.diff_text = self.style_features - self.text_features
            self.diff_text = self.diff_text / (
                self.diff_text.norm(dim=-1, keepdim=True) + 1e-8
            )

    def avrege_text_embedding(self, style_texts: list):
        """
        Precomputes and caches the average text embeddings for a given list of style texts.
        style_texts: list of str describing the styles (e.g. ["a sketch", "a painting", etc.])
        """
        with torch.no_grad():
            emb = self.processor(
                text=["photo"] + style_texts, return_tensors="pt", padding=True
            )
            emb = {
                k: v.to(next(self.clip_model.parameters()).device)
                for k, v in emb.items()
            }
            text_features = self.clip_model.get_text_features(**emb)
            text_features = text_features.detach()
            self.style_features = text_features[1:, :].mean(dim=0, keepdim=True)
            self.text_features = text_features[0, :].unsqueeze(0)
            self.style_features /= self.style_features.norm(dim=-1, keepdim=True) + 1e-8
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True) + 1e-8
            self.diff_text = self.style_features - self.text_features
            self.diff_text = self.diff_text / (
                self.diff_text.norm(dim=-1, keepdim=True) + 1e-8
            )
        return self.style_features


class ContentLoss(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.enc1 = encoder[:7]
        self.enc2 = encoder[7:11]
        self.enc3 = encoder[11:15]
        self.enc4 = encoder[15:]

    def forward(self, y_gen, real_images):
        loss = torch.tensor(0.0, requires_grad=True).to(y_gen.device)
        for enc in [self.enc1, self.enc2, self.enc3, self.enc4]:
            y_gen = enc(y_gen)
            real_images = enc(real_images)
            loss += ((y_gen - real_images) ** 2).mean()
        return loss

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return tv_h + tv_w


class CLIPStylerLoss(nn.Module):
    def __init__(
        self,
        directional_clip_loss,
        content_loss,
        tv_loss,
        lambda_clip=5e2, 
        lambda_patch=9e3,
        lambda_content=150.0,
        lambda_tv=2e-3,
        use_log=False,
    ):
        super(CLIPStylerLoss, self).__init__()
        self.directional_clip_loss = directional_clip_loss
        self.content_loss = content_loss
        self.tv_loss = tv_loss
        self.lambda_clip = lambda_clip
        self.lambda_patch = lambda_patch
        self.lambda_content = lambda_content
        self.lambda_tv = lambda_tv
        self.use_log = use_log

    def forward(self, real_images, generated_images):
        """
        Computes the combined loss for style transfer.
        images should not be normalized yet it is done in this function.
        real_images: tensor of shape (B,3,H,W) original image before style transfer
        generated_images: tensor of shape (B,3,H,W) image after style transfer
        real_patches: tensor of patches with shape (B,3,H,W) batch of real images patches
        generated_patches: tensor of patches with shape (B,3,H,W) batch of generated images patches
        style_text: str describing the style (e.g. "a sketch", "a painting", etc.)
        returns: total_loss, clip_loss, clip_patch_loss, content_loss, tv_loss
        NOTE: cause clip and vgg needs diffrence preprocessing processing is done manually here just use a transform that does rescale but not normalize to process the image
        the rest is left to this function
        """

        real_images_normalized = normalize(real_images)
        generated_images_normalized = normalize(generated_images)
        clip_loss, clip_patch_loss = self.directional_clip_loss(
            real_images_normalized, generated_images_normalized
        )
        clip_loss = clip_loss * self.lambda_clip
        clip_patch_loss = clip_patch_loss * self.lambda_patch
        content_loss = (
            self.content_loss(generated_images_normalized, real_images_normalized)
            * self.lambda_content
        )

        tv_loss = self.tv_loss(generated_images) * self.lambda_tv

        if self.use_log:
            clip_loss = torch.log(clip_loss + 1.0)
            clip_patch_loss = torch.log(clip_patch_loss + 1.0)
            content_loss = torch.log(content_loss + 1.0)
            tv_loss = torch.log(tv_loss + 1.0)

        total_loss = clip_loss + clip_patch_loss + content_loss + tv_loss
        return total_loss, clip_loss, clip_patch_loss, content_loss, tv_loss


class UpBlock(nn.Module):
    def __init__(self, in_channel=128, out_channel=64):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = nn.Sequential(
            nn.Conv2d(out_channel + in_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )

        self.skip = nn.Conv2d(out_channel + in_channel, out_channel, 1, 1, 0)

    def forward(self, input, FB_in):
        out_temp = self.upsample(input)
        out_temp = torch.cat([out_temp, FB_in], dim=1)
        out = self.conv(out_temp) + self.skip(out_temp)

        return out


class DownBlock(nn.Module):
    def __init__(self, in_channel=3, out_channel=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        self.skip = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        # self.downsample = nn.MaxPool2d(2,2)
        self.downsample = nn.Conv2d(out_channel, out_channel, 4, 2, 1)

    def forward(self, input):
        out_temp = self.conv(input) + self.skip(input)
        out = self.downsample(out_temp)
        return out, out_temp


class EncodingBlock(nn.Module):
    def __init__(self, in_channel=256, out_channel=512):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            # nn.BatchNorm2d(out_channel),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )
        self.skip = nn.Conv2d(in_channel, out_channel, 1, 1, 0)

    def forward(self, input):
        out = self.conv(input) + self.skip(input)
        return out



class UNet(nn.Module):
    def __init__(self, depth: int = 3, ngf=16, input_channel=3, output_channel=3):
        super(UNet, self).__init__()
        assert depth >= 1, "depth must be >= 1"
        self.depth = depth
        self.conv_init = nn.Conv2d(input_channel, ngf, 1, 1, 0)
        self.init = EncodingBlock(ngf, ngf)

        ch = [ngf * (2 ** max(0, i - 1)) for i in range(depth + 1)]

        self.down_blocks = nn.ModuleList([DownBlock(ch[i], ch[i + 1]) for i in range(depth)])

        self.encoding = EncodingBlock(ch[depth], ch[depth] * 2)

        prev_in = ch[depth] * 2
        up_blocks = []
        for k in range(depth, 0, -1):
            up_blocks.append(UpBlock(prev_in, ch[k]))
            prev_in = ch[k]
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = EncodingBlock(ch[1] + ch[0], ch[0])
        self.conv_fin = nn.Conv2d(ch[0], output_channel, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, use_sigmoid=True):
        x0 = self.conv_init(x)
        x_init = self.init(x0)

        d = x_init
        skips = []
        for down in self.down_blocks:
            d, d_f = down(d)
            skips.append(d_f)

        h = self.encoding(d)

        out = h
        for up, skip in zip(self.up_blocks, reversed(skips)):
            out = up(out, skip)

        h = self.out(torch.cat([out, x_init], dim=1))
        h = self.conv_fin(h)
        if use_sigmoid:
            h = self.sigmoid(h)
        return h


class ImageDataset(Dataset):
    def __init__(self, image_paths: list, image_transform: transforms.Compose):
        """Dataset that returns a single random patch from the given image using the specified transform.
        n_patches: number of patches to sample from each image
        length: number of patches sampled per epoch acts as dataset length and a sign for DataLoader to know when to stop
        transform: transform to apply on an tensor image image_transform and patch_transform is independent of each other i.e what happens in the other transform does not affect the other
        """
        self.image_paths = image_paths
        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_res = self.image_transform(img)

        return img_res

def get_patch_transform(patch_size, n_patches):
    transform = nn.Sequential(
        K.RandomCrop((patch_size, patch_size)),
        K.RandomPerspective(p=1, distortion_scale=0.5),
        K.Resize((224, 224)),
    )

    def fun(imgs):
        b, c, h, w = imgs.shape
        imgs = imgs.repeat_interleave(n_patches, dim=0)
        patches = transform(imgs)
        return patches.view(b, n_patches, c, 224, 224)

    return fun


def normalize(imgs):
    """Normalizes a batch of images with ImageNet mean and std.
    imgs: tensor of shape (B,3,H,W)
    returns: tensor of shape (B,3,H,W) normalized images
    """
    mean = torch.tensor([0.481, 0.456, 0.406]).view(1, 3, 1, 1).to(imgs.device)
    std = torch.tensor([0.268, 0.261, 0.275]).view(1, 3, 1, 1).to(imgs.device)
    imgs = (imgs - mean) / std
    return imgs

def montier():
    tensor_process = subprocess.Popen(
    ["tensorboard", "--logdir", "logs", "--port", "6006"],
)
    ngrok_process = subprocess.Popen(
        ["ngrok", "http", "6006"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
def denormalize(imgs):
    """Denormalizes a batch of images normalized with ImageNet mean and std.
    imgs: tensor of shape (B,3,H,W)
    returns: tensor of shape (B,3,H,W) denormalized images
    """
    mean = torch.tensor([0.481, 0.456, 0.406]).view(1, 3, 1, 1).to(imgs.device)
    std = torch.tensor([0.268, 0.261, 0.275]).view(1, 3, 1, 1).to(imgs.device)
    imgs = imgs * std + mean
    return imgs


class CLIPStyler(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_fn: CLIPStylerLoss, lr=1e-4, T_0=20):
        super(CLIPStyler, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.init_lr = lr
        self.T_0 = T_0
        self.last_eval_results = None  # stores prev epoch eval results
        self.eval_results = (
            dict()
        )  # stores current epoch eval results and aggregates them at each iteration then avreges at epoch end

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        batch: a tuble of (original_image, real_patches) use attributes of self.style_text for prompt control
        """
        self.model.train()
        real_images = batch
        generated_images = self.model(real_images)
        total_loss, clip_loss, clip_patch_loss, content_loss, tv_loss = self.loss_fn(
            real_images,
            generated_images,
        )
        
        self.logger.experiment.add_scalars(
            "loss",
            {
                "total": total_loss,
                "clip": clip_loss,
                "clip_patch": clip_patch_loss,
                "content": content_loss,
                "tv": tv_loss,
                # "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            self.global_step,
        )

        # self.logger.experiment.add_scalars("train/total_loss", total_loss, self.global_step)
        return total_loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        (
            t_loss_total,
            clip_loss_total,
            clip_patch_loss_total,
            content_loss_total,
            tv_loss_total,
        ) = (
            self.eval_results.get("total_loss", 0.0),
            self.eval_results.get("clip_loss", 0.0),
            self.eval_results.get("clip_patch_loss", 0.0),
            self.eval_results.get("content_loss", 0.0),
            self.eval_results.get("tv_loss", 0.0),
        )

        with torch.no_grad():
            real_images = batch
            generated_images = self.model(real_images)
            if batch_idx == 0:
                self.logger.experiment.add_images(
                    "real_images", real_images, self.global_step
                )
                self.logger.experiment.add_images(
                    "generated_images", generated_images, self.global_step
                )
            t_loss, clip_loss, clip_patch_loss, content_loss, tv_loss = self.loss_fn(
                real_images,
                generated_images,
            )
            t_loss_total += t_loss
            clip_loss_total += clip_loss
            clip_patch_loss_total += clip_patch_loss
            content_loss_total += content_loss
            tv_loss_total += tv_loss
            self.log("val/total_loss", t_loss, prog_bar=True)
            self.log("val/clip_loss", clip_loss, prog_bar=True)
            self.log("val/clip_patch_loss", clip_patch_loss, prog_bar=True)
            self.log("val/content_loss", content_loss, prog_bar=True)
            self.log("val/tv_loss", tv_loss, prog_bar=True)
        eval_results = {
            "total_loss": t_loss_total,
            "clip_loss": clip_loss_total,
            "clip_patch_loss": clip_patch_loss_total,
            "content_loss": content_loss_total,
            "tv_loss": tv_loss_total,
        }
        self.eval_results = eval_results

    def on_validation_epoch_end(self):
        # average the eval results over the number of batches
        num_batches = self.trainer.num_val_batches[0]
        avg_eval_results = {k: v / num_batches for k, v in self.eval_results.items()}
        self.last_eval_results = avg_eval_results
        self.eval_results = dict()  # reset for next epoch

        if self.current_epoch > 0 and self.last_eval_results is not None:
            prev_eval_results = self.last_eval_results
            t_loss_change = (
                avg_eval_results["total_loss"] - prev_eval_results["total_loss"]
            )
            clip_loss_change = (
                avg_eval_results["clip_loss"] - prev_eval_results["clip_loss"]
            )
            clip_patch_loss_change = (
                avg_eval_results["clip_patch_loss"]
                - prev_eval_results["clip_patch_loss"]
            )
            content_loss_change = (
                avg_eval_results["content_loss"] - prev_eval_results["content_loss"]
            )
            tv_loss_change = avg_eval_results["tv_loss"] - prev_eval_results["tv_loss"]

            self.logger.experiment.add_scalars(
                "val/loss_change",
                {
                    "total_loss_change": t_loss_change,
                    "clip_loss_change": clip_loss_change,
                    "clip_patch_loss_change": clip_patch_loss_change,
                    "content_loss_change": content_loss_change,
                    "tv_loss_change": tv_loss_change,
                },
                self.current_epoch,
            )
        self.last_eval_results = avg_eval_results
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.init_lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.T_0, T_mult=1, eta_min=1e-6
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


# paths = os.listdir("/kaggle/input/div2k-high-resolution-images/DIV2K_train_HR/DIV2K_train_HR/")
# paths = [os.path.join("/kaggle/input/div2k-high-resolution-images/DIV2K_train_HR/DIV2K_train_HR",p) for p in paths]

def train(
    paths,
    prompts,
    patch_size = 64,
    val_paths = None,
    scale_data=1,
    n_epochs = None,
    batch_size = 1,
    n_patches=16,
    img_size=512,
    lambda_clip=500,
    lambda_patch=5000,
    lambda_content=150,
    lambda_tv=2e3,
    lr = 5e-4,
    save_name= None,
    accumulate_grad_batches = 1,
    compile_model = False,
    use_logger = False,
    loader_process = 0,
    n_devices = 1,
    checkpoint_path = None,
    depth = None,
    restart_epochs = 10,
):
    """
    Args:
        paths:
            a list of paths to the input images or dir of root.

        prompts:
            the style you want the model to output

        patch_size:
            Size of each patch. Instead of training on the full image at once,
            the model learns on smaller local regions, which stabilizes optimization
            and improves generalization. A common choice is ~1/8 of the image size
        n_patches:
            Number of patches sampled per training step. Typical values range from
            16 to 64. Keep this small enough to avoid excessive overlap between
            randomly sampled patches

        scale_data:
            scale factor for small dataset.
        n_epochs:
            Total number of epochs
        lambda_clip:
            Weight of the CLIP loss. Increase this if the model is afraid of styling
            the image (weak stylization). Decrease it if the model
            overfits to style prompts and ignores the source image structure.

        lambda_patch:
            Weight of the patch-level loss. Usually set 10  times lambda_clip.
            Tuning logic is similar to lambda_clip, but this term mainly controls
            spatial consistency. If stylization becomes overly strong in some regions
            while others remain unchanged, reduce this value.

        lambda_content:
            Weight of the content preservation loss. Increase it if the model
            regions or full image structure. Avoid setting it
            too high — excessive content weight makes the model ignore style prompt and
            output same input image

        lambda_tv:
            Weight of the total variation regularization. Penalizes
            color changes (noise). Increase it if the output is noisy
            or speckled. Decrease it if the output collapses to flat color regions
    """
    
    if type(paths) == list and scale_data is not None and scale_data >1: 
        paths = paths * scale_data
    elif type(paths) == list:
        paths = paths
    else:
        root = paths
        paths = os.listdir(paths)
        paths = [os.path.join(root, path) for path in paths]
    data = ImageDataset(
        paths,
        image_transform=transforms.Compose(
            [
                transforms.Resize((img_size,img_size), antialias=False),
                transforms.ToTensor(),
            ],
        ),
    )
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=loader_process
    )
    for batch in loader:
        print('input size',batch.shape)
        break
    val_dataset = None
    val_loader = None
    if val_paths:
        if type(val_paths) == list:
            val_paths = val_paths
        else:
            root = val_paths
            val_paths = os.listdir(val_paths)
            val_paths = [os.path.join(root, path) for path in val_paths]
        val_dataset = ImageDataset(
            val_paths,
            image_transform=transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ],
            ),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=loader_process
        )

    d_loss = DirectionalCLIPLoss(
        clip, processor, get_patch_transform(patch_size=patch_size, n_patches=n_patches)
    )
    c_loss = ContentLoss(encoder)
    tv_loss = TotalVariationLoss()
    loss_fn = CLIPStylerLoss(
        directional_clip_loss=d_loss,
        content_loss=c_loss,
        tv_loss=tv_loss,
        lambda_clip=lambda_clip,
        lambda_patch=lambda_patch,
        lambda_content=lambda_content,
        lambda_tv=lambda_tv,
        use_log=False,
    )

    d_loss.avrege_text_embedding(prompts)
    if depth is None and checkpoint_path is not None:
        raise ValueError('depth must be specified when loading from checkpoint')
    model = UNet(depth=depth) if depth is not None else UNet()
    if compile_model:
        model = torch.compile(model, dynamic=False, fullgraph=True)
        d_loss = torch.compile(d_loss, dynamic=False)
    if checkpoint_path:
        try:
            model = CLIPStyler.load_from_checkpoint(checkpoint_path,model = model, loss_fn=loss_fn, lr=lr,T_0 = restart_epochs,strict = True)
        except Exception as e:
            model = CLIPStyler.load_from_checkpoint(checkpoint_path,model = model, loss_fn=loss_fn, lr=lr,T_0 = restart_epochs,strict = False)
            print(f'Warning: strict loading failed due to {e}, loaded with strict = False')
    else:
        model = CLIPStyler(model = model, loss_fn=loss_fn, lr=lr,T_0 = restart_epochs)
    callbacks = [ModelCheckpoint(monitor=None,save_last=True,filename=save_name,save_on_exception=True,dirpath="./checkpoints",every_n_epochs=5)] if save_name else None
    logger = None or pl.pytorch.loggers.TensorBoardLogger("logs", name="clip_styler")
    print(f"training model {save_name}")
    print(f"Using batch size {batch_size}")
    print(f'Train Dataset {len(data)} images(scale: {scale_data}), {len(loader)} steps per epoch')
    print(f'n_patches: {n_patches}, patch_size: {patch_size}')
    # print(f'Validation Dataset {len(val_dataset)} images, {len(val_loader)} steps per epoch')
    print(f"Using {n_devices} GPU(s)")
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator="gpu",
        devices=n_devices,
        log_every_n_steps=10,
        precision=16,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=logger,
        # profiler="simple",
    )
    trainer.fit(model, loader,val_loader)

    return model



