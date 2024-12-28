from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor

class DataLoaderBase(ABC):
    """Clase abstracta para definir la estructura de un DataLoader."""

    def __init__(self, config, collate_fn):
        self.config = config
        self.collate_fn = collate_fn
        self.batch_size = config["training"]["batch_size"]

    @abstractmethod
    def get_train_loader(self):
        """Devuelve el DataLoader de entrenamiento."""
        pass

    @abstractmethod
    def get_val_loader(self):
        """Devuelve el DataLoader de validación."""
        pass


class CocoDataLoader(DataLoaderBase):
    """Implementación de DataLoader para datasets COCO."""

    def __init__(self, config, collate_fn):
        super().__init__(config, collate_fn)

    def get_train_loader(self):
        train_data_path = self.config["paths"]["train_data"]
        train_annotations_path = self.config["paths"]["train_annotations"]

        # Dataset de entrenamiento
        coco_train = CocoDetection(
            root=train_data_path,
            annFile=train_annotations_path,
            transform=ToTensor()
        )

        # DataLoader de entrenamiento
        train_loader = DataLoader(
            coco_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        return train_loader

    def get_val_loader(self):
        val_data_path = self.config["paths"]["val_data"]
        val_annotations_path = self.config["paths"]["val_annotations"]

        # Dataset de validación
        coco_val = CocoDetection(
            root=val_data_path,
            annFile=val_annotations_path,
            transform=ToTensor()
        )

        # DataLoader de validación
        val_loader = DataLoader(
            coco_val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        return val_loader
