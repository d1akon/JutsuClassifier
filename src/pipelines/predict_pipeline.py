#-------------------------
#        IMPORTS   
#-------------------------
import sys
import os
import cv2
import torch
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from torchvision.transforms import ToTensor
from PIL import Image
from fastai.vision.all import load_learner
from src.model.faster_rcnn.faster_rcnn import get_model

#-------------------------
#        MAIN CODE   
#-------------------------

def predict_faster_rcnn(frame, model, device, confidence_threshold=0.5):
    """Prediction with Faster R-CNN."""
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
            class_name = id_to_name.get(max_label.item(), "Unknown")

            x_min, y_min, x_max, y_max = map(int, max_box.tolist())
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f"{class_name}, Conf: {max_score:.2f}"
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

def predict_fastai(frame, learner):
    """Prediction with the FastAI model."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    #----- Convert image to tensors and get predictions
    with learner.no_bar():
        pred, pred_idx, probs = learner.predict(pil_img)

    #----- Draw prediction on the image
    text = f"{pred}, Conf: {probs[pred_idx]:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame

if __name__ == "__main__":
    #----- Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    #----- Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id_to_name = config["labels"]["id_to_name"]

    #----- Select which model to use based on configuration
    model_type = config["model"]["type"]

    if model_type == "faster_rcnn":
        #--- Load Faster R-CNN model
        model = get_model(config["rcnn_training"]["num_classes"])
        model.load_state_dict(torch.load(config["paths"]["frcnn_model_load_path"]))
        model.to(device)
        model.eval()

        #--- Camera capture for real-time predictions
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            frame_with_predictions = predict_faster_rcnn(frame, model, device)
            cv2.imshow("Real-Time Prediction (Faster R-CNN)", frame_with_predictions)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif model_type == "fastai":
        #--- Load FastAI model
        learner = load_learner(config["paths"]["fastai_model_load_path"], cpu=False)

        #--- Camera capture for real-time predictions
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            frame_with_predictions = predict_fastai(frame, learner)
            cv2.imshow("Real-Time Prediction (FastAI)", frame_with_predictions)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        raise ValueError(f"Unknown model type: {model_type}")
