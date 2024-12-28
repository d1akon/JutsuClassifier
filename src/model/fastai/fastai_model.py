from fastai.vision.all import *
from pathlib import Path
from sklearn.metrics import classification_report

class FastAIModel:
    def __init__(self, data_path, model_path, seed=42):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.seed = seed
        self.learner = None

    def load_data(self):
        np.random.seed(self.seed)
        self.data = ImageDataLoaders.from_folder(
            self.data_path,
            valid_pct=0.2,
            item_tfms=Resize(460),
            batch_tfms=aug_transforms(
                size=224,
                max_rotate=10,
                max_zoom=1.1,
                max_lighting=0.2,
                max_warp=0.2,
                p_affine=0.75,
                p_lighting=0.5
            )
        )

    def create_learner(self):
        self.learner = cnn_learner(self.data, resnet50, metrics=accuracy)

    def train(self, epochs=10, lr=3e-3, fine_tune_epochs=5):
        self.learner.fit_one_cycle(epochs, lr, cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=3)])
        self.learner.unfreeze()
        self.learner.fit_one_cycle(fine_tune_epochs, slice(1e-5, 1e-3), cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=3)])

    def save_model(self):
        self.learner.export(self.model_path)

    def load_model(self):
        self.learner = load_learner(self.model_path)

    def evaluate(self):
        preds, targs = self.learner.get_preds()
        labels = [self.data.vocab[i] for i in targs]
        pred_labels = [self.data.vocab[i] for i in preds.argmax(dim=1)]
        return classification_report(labels, pred_labels, target_names=self.data.vocab)

    def predict(self, image_path):
        img = PILImage.create(image_path)
        pred_class, pred_idx, probs = self.learner.predict(img)
        return pred_class, probs[pred_idx]
