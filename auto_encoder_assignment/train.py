import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from auto_encoder import Autoencoder, Autoencoder2, Autoencoder3, Autoencoder4, Autoencoder5, Autoencoder6  # Your autoencoder model here
import os
from tqdm import tqdm
import sys
import argparse
import yaml
import pandas as pd
import piqa
from torchvision.transforms import functional as F

from piqa import SSIM

torch.manual_seed(42)


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor()
])

# Custom Dataset class
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, horizontal_flip=False):
        self.root_dir = root_dir
        self.transform = transform
        self.horizontal_flip = horizontal_flip
        self.image_paths = sorted([os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.png')])
        self.image_dict = {}
        self.load_all_images()
        
    def __len__(self):
        return len(self.image_paths)

    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self.image_dict[img_path]
        # first 1000 images are saved for validation
        if idx > 1000 and self.horizontal_flip and torch.rand(1).item() < 0.5:
            image = F.hflip(image)
        
        return image, 0  # Return dummy label, as we don't have classes
    
   
    def load_all_images(self):
        for img_path in tqdm(self.image_paths):
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            self.image_dict[img_path] = image
           
    


def main(config: dict, logdir: str):    
    # Hyperparameters
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']

    
    
    dataset_root = '../../dataset'
    horizontal_flip = config.get('horizontal_flip', False)
    full_dataset = ImageDataset(root_dir=dataset_root, transform=transform, horizontal_flip=horizontal_flip)

    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Number of epochs: {num_epochs}")

    # Device configuration (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device ", device)
    print("device name", torch.cuda.get_device_name(device=device))


    # Split indices for training and validation
    val_indices = list(range(1000))  # First 1000 images for validation
    train_indices = list(range(1000, len(full_dataset)))  # Remaining images for training

    # Create training and validation subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    # Create DataLoaders for training and validation
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


    # Initialize model
    model_type = config['model_type']
    if model_type == 'base':
        model = Autoencoder().to(device)
    elif model_type == 'lrelu':
        model = Autoencoder2().to(device)
    elif model_type == 'batchnorm':
        model = Autoencoder3().to(device)
    elif model_type == 'batchnorm_dropout':
        model = Autoencoder4().to(device)
    elif model_type == 'residual':
        model = Autoencoder5().to(device)
    elif model_type == "residual2":
        model = Autoencoder6().to(device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    print(model)
    loss_type = config['loss_type']
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'ssim':
        criterion = SSIMLoss().cuda()
    elif loss_type == 'l1':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store the loss values
    train_losses = []
    val_losses = []
    print("Cuda available: ", torch.cuda.is_available())
    print("Began Training")
    best_val_loss = float('inf')
    best_val_epoch = 0
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        iteration = 0
        for data in tqdm(train_loader):
            # if iteration > 10:
            #     break
            inputs, _ = data
            inputs = inputs.to(device)
            

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            train_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1
        # Average training loss for the epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation after each epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in tqdm(val_loader):
                inputs, _ = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, inputs).item()
        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_model_save_path = os.path.join(logdir, 'best_autoencoder_model.pth')
            torch.save(model.state_dict(), best_model_save_path)
            print(f"Best model saved to {best_model_save_path}")
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print epoch losses
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Load the best model
    model.load_state_dict(torch.load(best_model_save_path))
    model.to(device)
    model.eval()
    print(f"Best model loaded from {best_model_save_path}")
    print(f"Best model at epoch {best_val_epoch + 1} with validation loss: {best_val_loss:.4f}")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    img_path = os.path.join(logdir, 'learning_curve.png')
    plt.savefig(img_path)
    plt.show()

    # Saving outputs of 50 images from the validation set before and after reconstruction
    
    output_dir = os.path.join(logdir, 'reconstruction_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the loss lists to a DataFrame and save as CSV
    loss_df = pd.DataFrame({
        "Epoch": range(1, num_epochs + 1),
        "Training Loss": train_losses,
        "Validation Loss": val_losses
    })
    csv_path = os.path.join(logdir, "loss_values.csv")
    loss_df.to_csv(csv_path, index=False)

    print("Loss values saved to loss_values.csv")

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Convert tensors to images and save
            for j in range(inputs.size(0)):
                original_image = transforms.ToPILImage()(inputs[j].cpu())
                reconstructed_image = transforms.ToPILImage()(outputs[j].cpu())

                # Concatenate original and reconstructed images side by side
                side_by_side = Image.new('RGB', (original_image.width * 2, original_image.height))
                side_by_side.paste(original_image, (0, 0))
                side_by_side.paste(reconstructed_image, (original_image.width, 0))

                # Save the concatenated image
                side_by_side.save(os.path.join(output_dir, f'comparison_{i*batch_size + j}.png'))
    
    
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--yaml_config', default=None)
    args = parser.parse_args()
    
    logdir = args.logdir
    yaml_path = args.yaml_config
    if logdir is None:
        raise RuntimeError("no logdir provided")
    if yaml_path is None:
        yaml_path = os.path.join(logdir, 'run_config.yaml')
        if not os.path.exists(yaml_path):
            raise RuntimeError("no yaml file provided")
    print("yaml path:", yaml_path)
    with open(yaml_path, 'r') as fp:
        yaml_config = yaml.load(fp, Loader=yaml.FullLoader)
    main(yaml_config['train_params'], logdir)
    