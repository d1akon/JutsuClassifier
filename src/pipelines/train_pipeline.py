import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import yaml
from src.model.faster_rcnn.faster_rcnn import get_model, save_model
from src.pipelines.scripts.dataloader import CocoDataLoader
from src.pipelines.scripts.collate_fn import collate_fn
from src.model.faster_rcnn.train import train_model
from utils.visualization import visualize_predictions


#----- Cargar configuración
with open("Z:/VSC/projects/projects_github/JutsuClassifier/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

#----- Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----- Cargar datos y modelo
#----- Crear instancia del DataLoader
data_loader = CocoDataLoader(config, collate_fn)

#----- Obtener el val_loader
train_loader = data_loader.get_train_loader()

model = get_model(config["training"]["num_classes"]).to(device)

#----- Entrenar modelo
train_model(model, train_loader, config, device)

#----- Guardar modelo
save_model(model, config["paths"]["model_save_path"])
