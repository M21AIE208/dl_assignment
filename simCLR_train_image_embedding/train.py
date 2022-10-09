import xml.etree.ElementTree as ET
from glob import glob
import os
import pandas as pd
import torchvision
from PIL import Image
from torchsummary import summary
from torchvision import transforms
import torch
from sklearn import svm
from PIL import Image
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
tqdm.pandas()
import torchvision.transforms.functional as tvf
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import random
from torch.multiprocessing import cpu_count
from torch.optim import RMSprop
from argparse import Namespace


class ResizedRotation():
    def __init__(self, angle, output_size=(96, 96)):
        self.angle = angle
        self.output_size = output_size

    def angle_to_rad(self, ang): return np.pi * ang / 180.0

    def __call__(self, image):
        w, h = image.size
        new_h = int(
            np.abs(w * np.sin(self.angle_to_rad(90 - self.angle))) + np.abs(h * np.sin(self.angle_to_rad(self.angle))))
        new_w = int(
            np.abs(h * np.sin(self.angle_to_rad(90 - self.angle))) + np.abs(w * np.sin(self.angle_to_rad(self.angle))))
        img = tvf.resize(image, (new_w, new_h))
        img = tvf.rotate(img, self.angle)
        img = tvf.center_crop(img, self.output_size)
        return img


class WrapWithRandomParams():
    def __init__(self, constructor, ranges):
        self.constructor = constructor
        self.ranges = ranges

    def __call__(self, image):
        randoms = [float(np.random.uniform(low, high)) for _, (low, high) in zip(range(len(self.ranges)), self.ranges)]
        return self.constructor(*randoms)(image)

cifar10_unlabeled = CIFAR10("../dataset", download=True)

def random_rotate(image):
    if random.random() > 0.5:
        return tvf.rotate(image, angle=random.choice((0, 90, 180, 270)))
    return image


class PretrainingDatasetWrapper(Dataset):
    def __init__(self, ds: Dataset, target_size=(96, 96), debug=False):
        super().__init__()
        self.ds = ds
        self.debug = debug
        self.target_size = target_size
        if debug:
            print("DATASET IN DEBUG MODE")

        # I will be using network pre-trained on ImageNet first, which uses this normalization.
        # Remove this, if you're training from scratch or apply different transformations accordingly
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        random_resized_rotation = WrapWithRandomParams(lambda angle: ResizedRotation(angle, target_size),
                                                       [(0.0, 360.0)])
        self.randomize = transforms.Compose([
            transforms.RandomResizedCrop(target_size, scale=(1 / 3, 1.0), ratio=(0.3, 2.0)),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Lambda(random_rotate)
            ]),
            transforms.RandomApply([
                random_resized_rotation
            ], p=0.33),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, idx, preprocess=True):
        this_image_raw, _ = self.ds[idx]

        if self.debug:
            random.seed(idx)
            t1 = self.randomize(this_image_raw)
            random.seed(idx + 1)
            t2 = self.randomize(this_image_raw)
        else:
            t1 = self.randomize(this_image_raw)
            t2 = self.randomize(this_image_raw)

        if preprocess:
            t1 = self.preprocess(t1)
            t2 = self.preprocess(t2)
        else:
            t1 = transforms.ToTensor()(t1)
            t2 = transforms.ToTensor()(t2)

        return (t1, t2), torch.tensor(0)

    def __getitem__(self, idx):
        return self.__getitem_internal__(idx, True)

    def raw(self, idx):
        return self.__getitem_internal__(idx, False)


class ImageEmbedding(nn.Module):
    class Identity(nn.Module):
        def __init__(self): super().__init__()

        def forward(self, x):
            return x

    def __init__(self, embedding_size=1024):
        super().__init__()

        base_model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        internal_embedding_size = base_model.fc.in_features
        base_model.fc = ImageEmbedding.Identity()

        self.embedding = base_model

        self.projection = nn.Sequential(
            nn.Linear(in_features=internal_embedding_size, out_features=embedding_size),
            nn.ReLU(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size)
        )

    def calculate_embedding(self, image):
        return self.embedding(image)

    def forward(self, X):
        image = X
        embedding = self.calculate_embedding(image)
        projection = self.projection(embedding)
        return embedding, projection


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss



class ImageEmbeddingModule(pl.LightningModule):
    def __init__(self, hparams):
        # hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        super().__init__()
        self.save_hyperparameters(hparams)
        # self.hparams = hparams
        self.model = ImageEmbedding()
        self.loss = ContrastiveLoss(self.hparams.batch_size)

    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.epochs

    def train_dataloader(self):
        return DataLoader(PretrainingDatasetWrapper(cifar10_unlabeled,
                                                    debug=getattr(self.hparams, "debug", False)),
                          batch_size=self.hparams.batch_size,
                          num_workers=cpu_count(),
                          sampler=SubsetRandomSampler(list(range(hparams.train_size))),
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(PretrainingDatasetWrapper(cifar10_unlabeled,
                                                    debug=getattr(self.hparams, "debug", False)),
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=cpu_count(),
                          sampler=SequentialSampler(
                              list(range(hparams.train_size + 1, hparams.train_size + hparams.validation_size))),
                          drop_last=True)

    def forward(self, X):
        return self.model(X)

    def step(self, batch, step_name="train"):
        (X, Y), y = batch
        embX, projectionX = self.forward(X)
        embY, projectionY = self.forward(Y)
        loss = self.loss(projectionX, projectionY)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        return {("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                "progress_bar": {loss_key: loss}}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def validation_end(self, outputs):
        if len(outputs) == 0:
            return {"val_loss": torch.tensor(0)}
        else:
            loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            return {"val_loss": loss, "log": {"val_loss": loss}}

    def configure_optimizers(self):
        optimizer = RMSprop(self.model.parameters(), lr=self.hparams.lr)
        return [optimizer], []


hparams = Namespace(
    # lr=0.000630957344480193,
    lr=0.0005,
    epochs=15,
    batch_size=160,
    train_size=20000,
    validation_size=1000
)
module = ImageEmbeddingModule(hparams)
trainer = pl.Trainer(gpus=1,max_epochs=hparams.epochs)
trainer.fit(module)
checkpoint_file = "../models/resnet-18-cifar-embeddings.ckpt"
trainer.save_checkpoint(checkpoint_file)