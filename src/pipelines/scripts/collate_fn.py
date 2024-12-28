from torchvision.transforms.functional import resize
import torch

def collate_fn(batch, resize_shape=(224, 224)):
    images, targets = [], []
    for img, target in batch:
        _, h, w = img.shape
        img_resized = resize(img, resize_shape)
        scale_x, scale_y = resize_shape[1] / w, resize_shape[0] / h

        valid_boxes, valid_labels = [], []
        for obj in target:
            bbox = obj["bbox"]
            x_min, y_min, width, height = bbox
            valid_boxes.append([
                x_min * scale_x, y_min * scale_y,
                (x_min + width) * scale_x, (y_min + height) * scale_y
            ])
            valid_labels.append(obj["category_id"])

        images.append(img_resized)
        targets.append({"boxes": torch.tensor(valid_boxes), "labels": torch.tensor(valid_labels)})
    return images, targets
