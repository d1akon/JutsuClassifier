import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, num_classes):
    model = get_model(num_classes)
    state_dict = torch.load(path)  #--- Load model's state
    model.load_state_dict(state_dict)  #--- Load weights into the model
    return model

