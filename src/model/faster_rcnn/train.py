from tqdm import tqdm
import torch

def train_faster_rcnn_model(model, train_loader, config, device):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["rcnn_training"]["learning_rate"],
        momentum=config["rcnn_training"]["momentum"],
        weight_decay=config["rcnn_training"]["weight_decay"]
    )

    num_epochs = config["rcnn_training"]["num_epochs"]
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):

            try:
                #----- Process and move targets to the device
                processed_targets = []
                for t in targets:
                    processed_target = {}
                    for k, v in t.items():
                        if k == 'bbox':
                            #--- Convert list of tensors into a single tensor if needed
                            processed_target[k] = torch.tensor(v).to(device) if isinstance(v, list) else v.to(device)
                        else:
                            processed_target[k] = v.to(device)
                    processed_targets.append(processed_target)

                #----- Move images to the device
                images = [img.to(device) for img in images]

                #----- Calculate loss
                loss_dict = model(images, processed_targets)
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()

                #----- Backpropagation and optimization
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            except Exception as e:
                #----- Debugging in case of error
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Targets structure: {targets}")
                print(f"Images shape: {[img.shape for img in images]}")
                raise e  #--- Reraise the error after printing the information

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
