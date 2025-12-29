import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # 1. Device Configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 2. Hyperparameters
    BATCH_SIZE = 64
    NOISE_FACTOR = 0.5
    LEARNING_RATE = 1e-3
    EPOCHS = 5

    # 3. Data Preparation
    print("Loading data...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Download to local ./data folder
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Model Definition
    class Denoiser(nn.Module):
        def __init__(self):
            super(Denoiser, self).__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.1),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1)
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.1),
                nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    model = Denoiser().to(device)

    # 5. Training Loop
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        train_loss = 0.0
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            
            # Add noise
            noise = torch.randn_like(img) * NOISE_FACTOR
            noisy_img = img + noise
            noisy_img = torch.clamp(noisy_img, 0., 1.)
            
            optimizer.zero_grad()
            outputs = model(noisy_img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss/len(train_loader):.4f}")

    # 6. Evaluation & Visualization
    print("Training complete. Generating results...")
    model.eval()
    
    # Get one batch
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    images = images.to(device)
    
    noise = torch.randn_like(images) * NOISE_FACTOR
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    
    with torch.no_grad():
        outputs = model(noisy_images)
        
    # Plotting
    images = images.cpu()
    noisy_images = noisy_images.cpu()
    outputs = outputs.cpu()
    
    # Plot first 10 images
    num_images = 10
    fig, axes = plt.subplots(3, num_images, figsize=(20, 6))
    
    for i in range(num_images):
        # Original
        axes[0, i].imshow(images[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original")
        
        # Noisy
        axes[1, i].imshow(noisy_images[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Noisy")
        
        # Denoised
        axes[2, i].imshow(outputs[i].reshape(28, 28), cmap='gray')
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title("Denoised")
    
    plt.tight_layout()
    plt.savefig('results.png')
    print("Results saved to results.png")

if __name__ == "__main__":
    main()
