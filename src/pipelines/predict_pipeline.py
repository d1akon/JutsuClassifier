import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from src.model.faster_rcnn.faster_rcnn import get_model
import yaml

id_to_name = {
    1: "bird", 2: "boar", 3: "dog", 4: "dragon", 5: "hare",
    6: "horse", 7: "monkey", 8: "ox", 9: "ram", 10: "rat",
    11: "snake", 12: "tiger"
}

def predict_and_draw(frame, model, device, confidence_threshold=0.5):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = ToTensor()(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    if len(predictions[0]['scores']) > 0:
        max_idx = predictions[0]['scores'].argmax()
        max_score = predictions[0]['scores'][max_idx]
        if max_score >= confidence_threshold:
            max_box = predictions[0]['boxes'][max_idx]
            max_label = predictions[0]['labels'][max_idx]
            class_name = id_to_name.get(max_label.item(), "Desconocido")

            x_min, y_min, x_max, y_max = map(int, max_box.tolist())
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f"{class_name}, Conf: {max_score:.2f}"
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

if __name__ == "__main__":
    #----- Cargar configuración
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(config["training"]["num_classes"])
    model.load_state_dict(torch.load(config["paths"]["model_load_path"]))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame de la webcam.")
            break

        frame_with_predictions = predict_and_draw(frame, model, device)
        cv2.imshow("Predicción en Tiempo Real", frame_with_predictions)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
