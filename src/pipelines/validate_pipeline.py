#-------------------------
#        IMPORTS   
#-------------------------
import sys
import os
import torch
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fastai.vision.all import load_learner
from src.pipelines.scripts.dataloader import CocoDataLoader, FastAIDataLoader
from src.model.faster_rcnn.faster_rcnn import get_model as get_faster_rcnn_model
from src.model.faster_rcnn.validate import validate_model as validate_faster_rcnn_model
from src.model.fastai.validate import validate_fastai_model

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
    id_to_name = config["labels"]["id_to_name"]

    if model_type == "faster_rcnn":
        #--- Create instance of DataLoader for Faster R-CNN
        from src.pipelines.scripts.collate_fn import collate_fn
        data_loader = CocoDataLoader(config, collate_fn)
        val_loader = data_loader.get_val_loader()

        #--- Load Faster R-CNN model
        model = get_faster_rcnn_model(config["rcnn_training"]["num_classes"])
        model.load_state_dict(torch.load(config["paths"]["frcnn_model_load_path"]))
        model.to(device)

        #--- Validate Faster R-CNN model
        validate_faster_rcnn_model(model, val_loader, device, id_to_name, max_samples=500)

    elif model_type == "fastai":
        #--- Create instance of DataLoader for FastAI
        data_loader = FastAIDataLoader(config, num_workers=0)
        data = data_loader.get_train_loader()
        val_loader = data_loader.get_val_loader()

        #--- Load FastAI model
        learner = load_learner(config["paths"]["fastai_model_load_path"], cpu=False)

        #--- If the model doesn't contain DataLoaders, assign them manually
        learner.dls = data

        #--- Validate FastAI model
        validate_fastai_model(learner, val_loader)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
