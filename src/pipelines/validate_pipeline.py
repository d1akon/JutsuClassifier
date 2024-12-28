import sys
import os
# Agregar el directorio raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from tqdm import tqdm
from utils.visualization import visualize_predictions
from src.pipelines.scripts.dataloader import CocoDataLoader
from src.pipelines.scripts.collate_fn import collate_fn
from sklearn.metrics import classification_report
from src.model.faster_rcnn.faster_rcnn import get_model
import yaml

def validate_model(model, val_loader, device, id_to_name, max_samples=None):
    model.eval()  #----- Cambiar a modo evaluación
    all_true_labels = []
    all_pred_labels = []
    total_batches = 0
    processed_samples = 0  #----- Contador de ejemplos procesados

    with torch.no_grad():  #----- No calcular gradientes
        for images, targets in tqdm(val_loader, desc="Validando modelo"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                true_labels = target['labels'].tolist()
                pred_scores = pred['scores'].tolist()
                pred_labels = pred['labels'].tolist()

                #----- Filtrar predicciones con confianza baja
                confidence_threshold = 0.5
                filtered_pred_labels = [
                    label for label, score in zip(pred_labels, pred_scores) if score >= confidence_threshold
                ]

                all_true_labels.extend(true_labels)
                all_pred_labels.extend(filtered_pred_labels)

                #----- Actualizar el contador de ejemplos procesados
                processed_samples += len(true_labels)
                if max_samples and processed_samples >= max_samples:
                    break

            total_batches += 1

            #----- Detener el bucle si se alcanza el límite de muestras
            if max_samples and processed_samples >= max_samples:
                break

    print(f"Validación completada en {total_batches} lotes.")

    #----- Obtener las clases presentes en los datos
    unique_labels = sorted(set(all_true_labels + all_pred_labels))
    print("Clasificación:")
    print(
        classification_report(
            all_true_labels,
            all_pred_labels,
            labels=unique_labels,
            target_names=[id_to_name[i] for i in unique_labels],
            zero_division=1  # Evitar errores para etiquetas sin predicciones
        )
    )


if __name__ == "__main__":

    id_to_name = {
        1: "bird", 2: "boar", 3: "dog", 4: "dragon", 5: "hare",
        6: "horse", 7: "monkey", 8: "ox", 9: "ram", 10: "rat",
        11: "snake", 12: "tiger"
    }

    #----- Cargar configuración
    with open("Z:/VSC/projects/projects_github/JutsuClassifier/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    #----- Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #----- Crear instancia del DataLoader
    data_loader = CocoDataLoader(config, collate_fn)

    #----- Obtener el val_loader
    val_loader = data_loader.get_val_loader()

    #----- Cargar el modelo
    model = get_model(config["training"]["num_classes"])
    model.load_state_dict(torch.load(config["paths"]["model_load_path"]))
    model.to(device)

    #----- Validar el modelo
    validate_model(model, val_loader, device, id_to_name, max_samples=500)
