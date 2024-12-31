#-------------------------
#        IMPORTS   
#-------------------------
import sys
import os
import torch
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.pipelines.scripts.dataloader import CocoDataLoader, FastAIDataLoader
from src.model.faster_rcnn.faster_rcnn import get_model as get_faster_rcnn_model, save_model
from src.pipelines.scripts.collate_fn import collate_fn
from src.model.faster_rcnn.train import train_faster_rcnn_model
from src.model.fastai.train import train_fastai_model

#-------------------------
#        MAIN CODE   
#-------------------------

if __name__ == "__main__":
    #----- Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    #----- Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #----- Select the model and DataLoader based on configuration
    model_type = config["model"]["type"]

    if model_type == "faster_rcnn":
        #----- Set up DataLoader for Faster R-CNN model
        data_loader = CocoDataLoader(config, collate_fn=collate_fn)
        train_loader = data_loader.get_train_loader()

        #----- Create and train
        model = get_faster_rcnn_model(config["rcnn_training"]["num_classes"]).to(device)
        train_faster_rcnn_model(model, train_loader, config, device)

        #----- Save
        save_model(model, config["paths"]["frcnn_model_save_path"])

    elif model_type == "fastai":
        #----- Set up DataLoader for FastAI model
        data_loader = FastAIDataLoader(config, num_workers=0)  # Important: num_workers=0 on Windows
        data = data_loader.get_train_loader()

        #----- Create, train and save
        learner = train_fastai_model(data, config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
