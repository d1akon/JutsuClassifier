import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

def validate_model(model, val_loader, device, id_to_name, max_samples=None):
    model.eval()  #--- Set to evaluation mode
    all_true_labels = []
    all_pred_labels = []
    total_batches = 0
    processed_samples = 0  #--- Processed sample counter

    with torch.no_grad():  #--- Do not calculate gradients
        for images, targets in tqdm(val_loader, desc="Validating model"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            for pred, target in zip(predictions, targets):
                true_labels = target['labels'].tolist()
                pred_scores = pred['scores'].tolist()
                pred_labels = pred['labels'].tolist()

                #--- Filter out predictions with low confidence
                confidence_threshold = 0.5
                filtered_pred_labels = [
                    label for label, score in zip(pred_labels, pred_scores) if score >= confidence_threshold
                ]

                all_true_labels.extend(true_labels)
                all_pred_labels.extend(filtered_pred_labels)

                #--- Update processed samples counter
                processed_samples += len(true_labels)
                if max_samples and processed_samples >= max_samples:
                    break

            total_batches += 1

            #----- Stop the loop if the sample limit is reached
            if max_samples and processed_samples >= max_samples:
                break

    print(f"Validation completed in {total_batches} batches.")

    #----- Get unique classes present in the data
    unique_labels = sorted(set(all_true_labels + all_pred_labels))
    print("Classification:")
    print(
        classification_report(
            all_true_labels,
            all_pred_labels,
            labels=unique_labels,
            target_names=[id_to_name[i] for i in unique_labels],
            zero_division=1  #--- Prevent errors for labels without predictions
        )
    )
