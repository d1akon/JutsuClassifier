from sklearn.metrics import classification_report

def validate_fastai_model(learner, val_loader):
    """Validate the FastAI model."""
    #----- Get predictions and true labels
    preds, targs = learner.get_preds(dl=val_loader)

    #----- Convert predictions to the most probable class index
    pred_labels = preds.argmax(dim=1)
    true_labels = targs

    #----- Get class names from the DataLoaders
    class_names = learner.dls.vocab

    #----- Display classification metrics
    print(classification_report(true_labels, pred_labels, target_names=class_names))
