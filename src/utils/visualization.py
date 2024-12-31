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


def visualize_resized_example(image, target, label_map=None):
    """
    Visualiza una imagen redimensionada junto con sus cajas de anotación.
    
    Args:
        image (Tensor): Imagen en formato Tensor [C, H, W].
        target (dict): Diccionario con las anotaciones de la imagen. Debe contener 'boxes' y 'labels'.
        label_map (dict): Mapeo opcional de IDs de clase a nombres.
    """
    # Convertir la imagen de Tensor a formato HWC y normalizar para mostrar correctamente
    img = image.permute(1, 2, 0).cpu().numpy()

    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)

    # Agregar cajas de anotación
    for box, label in zip(target['boxes'], target['labels']):
        x_min, y_min, x_max, y_max = box.tolist()
        width, height = x_max - x_min, y_max - y_min

        # Crear un rectángulo para la caja
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        # Agregar etiqueta si existe un label_map
        if label_map:
            label_name = label_map.get(label.item(), f"ID: {label.item()}")
            ax.text(
                x_min, y_min - 5, label_name,
                color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.5)
            )
        else:
            ax.text(
                x_min, y_min - 5, f"ID: {label.item()}",
                color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.5)
            )

    plt.axis('off')
    plt.show()


def validate_resized_batches(data_loader, label_map=None):
    """
    Visualiza un lote de imágenes con sus respectivas cajas de anotación.
    
    Args:
        data_loader (DataLoader): DataLoader que contiene imágenes y anotaciones.
        label_map (dict): Mapeo opcional de IDs de clase a nombres.
    """
    for batch_idx, (images, targets) in enumerate(data_loader):
        print(f"Visualizando lote {batch_idx + 1}")

        # Iterar sobre las imágenes y sus respectivas anotaciones
        for img, target in zip(images, targets):
            visualize_resized_example(img, target, label_map)

        # Visualizar solo el primer lote para evitar demasiadas imágenes
        if batch_idx == 0:
            break
