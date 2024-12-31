from fastai.vision.all import vision_learner, resnet50, accuracy, EarlyStoppingCallback
from src.pipelines.scripts.collate_fn import collate_fn
from src.model.faster_rcnn.train import train_faster_rcnn_model


def train_fastai_model(data, config):
    """Train the image classification model with FastAI."""
    learner = vision_learner(
        data,
        resnet50,
        metrics=accuracy
    )

    #----- Find learning rate
    learner.lr_find()
    lr = config["fastai_training"]["learning_rate"]

    #----- Train model
    learner.fit_one_cycle(config["fastai_training"]["epochs"], lr, cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=3)])
    learner.unfreeze()
    learner.fit_one_cycle(5, slice(1e-5, lr / 10), cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=3)])

    #----- Save model
    learner.export(config["paths"]["fastai_model_save_path"])
    return learner
