from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from fastai.vision.all import ImageDataLoaders, Resize, aug_transforms
from pathlib import Path

class DataLoaderBase(ABC):
    """Abstract class to define the structure of a DataLoader."""

    def __init__(self, config, collate_fn=None, num_workers=0):
        self.config = config
        self.collate_fn = collate_fn
        self.batch_size = config["rcnn_training"]["batch_size"]
        self.num_workers = num_workers

    @abstractmethod
    def get_train_loader(self):
        """Returns the training DataLoader."""
        pass

    @abstractmethod
    def get_val_loader(self):
        """Returns the validation DataLoader."""
        pass


class CocoDataLoader(DataLoaderBase):
    """Implementation of DataLoader for COCO datasets."""

    def __init__(self, config, collate_fn=None, num_workers=0):
        super().__init__(config, collate_fn, num_workers)

    def get_train_loader(self):
        train_data_path = self.config["paths"]["train_data"]
        train_annotations_path = self.config["paths"]["train_annotations"]

        coco_train = CocoDetection(
            root=train_data_path,
            annFile=train_annotations_path,
            transform=ToTensor()
        )

        train_loader = DataLoader(
            coco_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

        return train_loader

    def get_val_loader(self):
        val_data_path = self.config["paths"]["val_data"]
        val_annotations_path = self.config["paths"]["val_annotations"]

        coco_val = CocoDetection(
            root=val_data_path,
            annFile=val_annotations_path,
            transform=ToTensor()
        )

        val_loader = DataLoader(
            coco_val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

        return val_loader


class FastAIDataLoader(DataLoaderBase):
    """Implementation of DataLoader for FastAI."""

    def __init__(self, config, collate_fn=None, num_workers=0):
        super().__init__(config, collate_fn, num_workers)
        self.data = None  #--- Initialize `self.data` to None.

    def get_train_loader(self):
        if self.data is None:  #--- Only initialize once.
            path = self.config["paths"]["fastai_data"]

            self.data = ImageDataLoaders.from_folder(
                path,
                valid_pct=0.2,
                item_tfms=Resize(460),
                batch_tfms=aug_transforms(size=224),
                num_workers=self.num_workers
            )

        return self.data.train  #--- Return the training set.

    def get_val_loader(self):
        if self.data is None:  #--- Check that `self.data` is initialized.
            raise ValueError("You must first call get_train_loader to initialize the DataLoaders.")
        return self.data.valid  #--- Return the validation set.
