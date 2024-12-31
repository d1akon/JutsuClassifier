from torchvision.transforms.functional import resize
import torch

def collate_fn(batch):
    new_size = (224, 224)  #--- images new size
    images, targets = [], []

    for img, target in batch:
        _, original_height, original_width = img.shape  #--- Format [C, H, W]

        #----- Resize image
        img_resized = resize(img, new_size)

        #----- Calculate scaling factors
        scale_x = new_size[1] / original_width
        scale_y = new_size[0] / original_height

        valid_boxes = []
        valid_labels = []
        for obj in target:
            bbox = obj["bbox"]
            x_min, y_min, width, height = bbox

            #--- Scale the bounding box coordinates
            x_min = x_min * scale_x
            y_min = y_min * scale_y
            width = width * scale_x
            height = height * scale_y

            if width > 0 and height > 0:
                valid_boxes.append([x_min, y_min, x_min + width, y_min + height])
                valid_labels.append(obj["category_id"])

        if valid_boxes:
            images.append(img_resized)
            targets.append({
                "boxes": torch.tensor(valid_boxes, dtype=torch.float32),
                "labels": torch.tensor(valid_labels, dtype=torch.int64),
            })

    return images, targets
