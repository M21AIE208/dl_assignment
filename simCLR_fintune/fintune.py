from torchvision.datasets import CIFAR100
from torch.utils.data import random_split
import numpy as np
import torchvision.transforms as T
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
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


cifa100_data = CIFAR100(root="./dataset", download=True,
                        transform=T.Compose([
                                T.Resize(size=224),
                                T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]))


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

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


balanced_batch_sampler = BalancedBatchSampler(cifa100_data, n_classes=100, n_samples=10)
dataloader = torch.utils.data.DataLoader(cifa100_data, batch_sampler=balanced_batch_sampler)
my_testiter = iter(dataloader)
images, target = my_testiter.next()
tensor_x = torch.Tensor(images)  # transform to torch tensor
tensor_y = torch.Tensor(target)
cifar100_sub_sampled_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset


class SimCLRClassifier(nn.Module):
    def __init__(self, n_classes, freeze_base, embeddings_model_path, hidden_size=512):
        super().__init__()

        base_model = ImageEmbeddingModule.load_from_checkpoint(embeddings_model_path).model

        self.embeddings = base_model.embedding

        if freeze_base:
            print("Freezing embeddings")
            for param in self.embeddings.parameters():
                param.requires_grad = False

        # Only linear projection on top of the embeddings should be enough
        self.classifier = nn.Linear(in_features=base_model.projection[0].in_features,
                                    out_features=n_classes if n_classes > 2 else 1)

    def forward(self, X, *args):
        emb = self.embeddings(X)
        return self.classifier(emb)


class SimCLRClassifierModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = SimCLRClassifier(hparams.n_classes, hparams.freeze_base,
                                      hparams.embeddings_path,
                                      self.hparams.hidden_size)
        self.loss = nn.CrossEntropyLoss()

    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.epochs

    def preprocessing(self):
        return transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_dataloader(self, split):
        return DataLoader(cifar100_sub_sampled_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=split == "train",
                          num_workers=cpu_count(),
                          drop_last=False)

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("test")

    def forward(self, X):
        return self.model(X)

    def step(self, batch, step_name="train"):
        X, y = batch
        y_out = self.forward(X)
        loss = self.loss(y_out, y)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        return {("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                "progress_bar": {loss_key: loss}}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(Batch, "test")

    def validation_end(self, outputs):
        if len(outputs) == 0:
            return {"val_loss": torch.tensor(0)}
        else:
            loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            return {"val_loss": loss, "log": {"val_loss": loss}}

    def configure_optimizers(self):
        optimizer = RMSprop(self.model.parameters(), lr=self.hparams.lr)
        schedulers = [
            CosineAnnealingLR(optimizer, self.hparams.epochs)
        ] if self.hparams.epochs > 1 else []
        return [optimizer], schedulers

hparams_cls = Namespace(
    lr=0.000251,
    epochs=10,
    batch_size=64,
    n_classes=100,
    freeze_base=True,
    embeddings_path="../models/resnet-18-cifar-embeddings.ckpt",
    hidden_size=512
)
trainer2 = pl.Trainer(gpus=1, max_epochs=hparams_cls.epochs)
module2 = SimCLRClassifierModule(hparams_cls)
trainer2.fit(module2)
checkpoint_file = "../models/cifar100-classifer_1.ckpt"
trainer2.save_checkpoint(checkpoint_file)