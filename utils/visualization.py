import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_predictions(image, predictions, id_to_name, confidence_threshold=0.5):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image.permute(1, 2, 0).numpy())

    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if score < confidence_threshold:
            continue
        x_min, y_min, x_max, y_max = box.tolist()
        ax.add_patch(patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor="r", facecolor="none"
        ))
        ax.text(x_min, y_min - 10, f"{id_to_name[label.item()]}: {score:.2f}")
    plt.show()


