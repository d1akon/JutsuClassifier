#-------------------------
#        IMPORTS   
#-------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.pipelines.scripts.collate_fn import collate_fn
from src.utils.visualization import validate_resized_batches
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
import yaml

#-------------------------
#        MAIN CODE   
#-------------------------

#----- Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

train_data_path = config["paths"]["train_data"]
train_annotations_path = config["paths"]["train_annotations"]
id_to_name = config["labels"]["id_to_name"]

#----- Load the COCO dataset for training
coco_train = CocoDetection(
    root=train_data_path,
    annFile=train_annotations_path,
    transform=ToTensor()  #---Convert images to tensors
)

#----- Create a DataLoader with a custom collate function 
train_loader = DataLoader(
    coco_train,
    batch_size=4,  #---Number of images per batch
    shuffle=True,
    collate_fn=collate_fn  
)

#----- Validate and visualize the first few batches of resized images
validate_resized_batches(train_loader, id_to_name)
