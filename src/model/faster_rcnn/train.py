from tqdm import tqdm
import torch

def train_model(model, train_loader, config, device):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"]
    )

    num_epochs = config["training"]["num_epochs"]
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
