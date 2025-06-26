"""
Latent Representation Analysis
Extract 6 features from first layer and 3 features from second layer from test set data
Based on the notebook implementation with proper unnormalization
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from task2 import DeconvNet

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the trained model
model = DeconvNet().to(device)
model.load_state_dict(torch.load("deConvModel.pt", map_location=device))
model.eval()

# Define proper transforms (same as training)
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define unnormalization transform
unnormalize = transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])

# Load test dataset with proper normalization
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# Get a test image
test_image, test_label = next(iter(testloader))
test_image = test_image.to(device)

print(f"Analyzing test image with label: {testset.classes[test_label.item()]}")

# Create visualization figure
fig = plt.figure(figsize=(18, 12), constrained_layout=True)
(originalImg, firstLayer, secondLayer) = fig.subfigures(3, 1)
originalAxes = originalImg.subplots(1, 1)
axesFirst = firstLayer.subplots(1, 6)
axesSecond = secondLayer.subplots(1, 3)

fig.suptitle('Latent Representations from Test Set', fontsize='18')
originalImg.suptitle('Original Test Image', color='black', fontsize='14')
firstLayer.suptitle('1st Convolutional Layer (6 Features)', color='blue', fontsize='14')
secondLayer.suptitle('2nd Convolutional Layer (3 Selected Features)', color='red', fontsize='14')

with torch.no_grad():
    # Plot Original Image
    original_display = unnormalize(test_image.squeeze(0).cpu())
    original_display = np.clip(original_display, 0, 1)
    original_display = np.array(original_display).transpose(1, 2, 0)
    originalAxes.imshow(original_display)
    originalAxes.set_title(f'Class: {testset.classes[test_label.item()]}')
    originalAxes.axis('off')

    # Forward pass through first conv layer and pool
    conv1_out = F.relu(model.conv1(test_image))
    pooled1_out, indices1 = model.pool(conv1_out)
    
    # Plot first convolutional layer latent representations (all 6 features)
    for i in range():
        # Create a copy and zero out all channels except the one we want
        first_layer_representation = pooled1_out.clone()
        for j in range(6):
            if j != i:
                first_layer_representation[:, j, :, :] = 0
        
        # Reconstruct through deconv2 only (from first layer)
        reconstructed = model.deconv2(F.relu(model.unpool(first_layer_representation, indices1)))
        
        # Unnormalize and prepare for display
        reconstructed = unnormalize(reconstructed.squeeze(0).cpu())
        reconstructed = np.clip(reconstructed, 0, 1)
        reconstructed = np.array(reconstructed).transpose(1, 2, 0)
        
        axesFirst[i].imshow(reconstructed)
        axesFirst[i].set_title(f'Feature {i+1}')
        axesFirst[i].axis('off')

    # Forward pass through second conv layer and pool
    conv2_out = F.relu(model.conv2(pooled1_out))
    pooled2_out, indices2 = model.pool(conv2_out)
    
    # Plot second convolutional layer latent representations (3 selected features)
    selected_features = [0, 7, 12]  # Choose 3 features from 16 total
    for i, feature_idx in enumerate(selected_features):
        # Create a copy and zero out all channels except the one we want
        second_layer_representation = pooled2_out.clone()
        for j in range(16):
            if j != feature_idx:
                second_layer_representation[:, j, :, :] = 0
        
        # Reconstruct through both deconv layers
        # First deconv1 (16->6 channels)
        upsampled = model.unpool(second_layer_representation, indices2, output_size=conv2_out.size())
        deconv1_out = model.deconv1(F.relu(upsampled))
        
        # Then deconv2 (6->3 channels)
        upsampled2 = model.unpool(deconv1_out, indices1, output_size=conv1_out.size())
        reconstructed = model.deconv2(F.relu(upsampled2))
        
        # Unnormalize and prepare for display
        reconstructed = unnormalize(reconstructed.squeeze(0).cpu())
        reconstructed = np.clip(reconstructed, 0, 1)
        reconstructed = np.array(reconstructed).transpose(1, 2, 0)
        
        axesSecond[i].imshow(reconstructed)
        axesSecond[i].set_title(f'Feature {feature_idx+1}')
        axesSecond[i].axis('off')

plt.savefig('latent_representations_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete! Visualization saved as 'latent_representations_analysis.png'")
print(f"Analyzed test image from class: {testset.classes[test_label.item()]}")
print("First layer: 6 features extracted")
print("Second layer: 3 features extracted (indices 1, 8, 13)")

# ============================================================================
# TRAINING SET ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("ANALYZING TRAINING SET IMAGE")
print("="*60)

# Load training dataset with proper normalization
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

# Get a training image
train_image, train_label = next(iter(trainloader))
train_image = train_image.to(device)

print(f"Analyzing training image with label: {trainset.classes[train_label.item()]}")

# Create visualization figure for training image
fig2 = plt.figure(figsize=(18, 12), constrained_layout=True)
(originalImg2, firstLayer2, secondLayer2) = fig2.subfigures(3, 1)
originalAxes2 = originalImg2.subplots(1, 1)
axesFirst2 = firstLayer2.subplots(1, 6)
axesSecond2 = secondLayer2.subplots(1, 3)

fig2.suptitle('Latent Representations from Training Set', fontsize='18')
originalImg2.suptitle('Original Training Image', color='black', fontsize='14')
firstLayer2.suptitle('1st Convolutional Layer (6 Features)', color='blue', fontsize='14')
secondLayer2.suptitle('2nd Convolutional Layer (3 Selected Features)', color='red', fontsize='14')

with torch.no_grad():
    # Plot Original Training Image
    original_display2 = unnormalize(train_image.squeeze(0).cpu())
    original_display2 = np.clip(original_display2, 0, 1)
    original_display2 = np.array(original_display2).transpose(1, 2, 0)
    originalAxes2.imshow(original_display2)
    originalAxes2.set_title(f'Class: {trainset.classes[train_label.item()]}')
    originalAxes2.axis('off')

    # Forward pass through first conv layer and pool
    conv1_out2 = F.relu(model.conv1(train_image))
    pooled1_out2, indices1_2 = model.pool(conv1_out2)
    
    # Plot first convolutional layer latent representations (all 6 features)
    for i in range(6):
        # Create a copy and zero out all channels except the one we want
        first_layer_representation2 = pooled1_out2.clone()
        for j in range(6):
            if j != i:
                first_layer_representation2[:, j, :, :] = 0
        
        # Reconstruct through deconv2 only (from first layer)
        reconstructed2 = model.deconv2(F.relu(model.unpool(first_layer_representation2, indices1_2)))
        
        # Unnormalize and prepare for display
        reconstructed2 = unnormalize(reconstructed2.squeeze(0).cpu())
        reconstructed2 = np.clip(reconstructed2, 0, 1)
        reconstructed2 = np.array(reconstructed2).transpose(1, 2, 0)
        
        axesFirst2[i].imshow(reconstructed2)
        axesFirst2[i].set_title(f'Feature {i+1}')
        axesFirst2[i].axis('off')

    # Forward pass through second conv layer and pool
    conv2_out2 = F.relu(model.conv2(pooled1_out2))
    pooled2_out2, indices2_2 = model.pool(conv2_out2)
    
    # Plot second convolutional layer latent representations (3 selected features)
    selected_features2 = [0, 7, 12]  # Same features as test image for comparison
    for i, feature_idx in enumerate(selected_features2):
        # Create a copy and zero out all channels except the one we want
        second_layer_representation2 = pooled2_out2.clone()
        for j in range(16):
            if j != feature_idx:
                second_layer_representation2[:, j, :, :] = 0
        
        # Reconstruct through both deconv layers
        # First deconv1 (16->6 channels)
        upsampled2 = model.unpool(second_layer_representation2, indices2_2, output_size=conv2_out2.size())
        deconv1_out2 = model.deconv1(F.relu(upsampled2))
        
        # Then deconv2 (6->3 channels)
        upsampled2_2 = model.unpool(deconv1_out2, indices1_2, output_size=conv1_out2.size())
        reconstructed2 = model.deconv2(F.relu(upsampled2_2))
        
        # Unnormalize and prepare for display
        reconstructed2 = unnormalize(reconstructed2.squeeze(0).cpu())
        reconstructed2 = np.clip(reconstructed2, 0, 1)
        reconstructed2 = np.array(reconstructed2).transpose(1, 2, 0)
        
        axesSecond2[i].imshow(reconstructed2)
        axesSecond2[i].set_title(f'Feature {feature_idx+1}')
        axesSecond2[i].axis('off')

plt.savefig('latent_representations_training.png', dpi=300, bbox_inches='tight')
plt.show()

print("Training set analysis complete! Visualization saved as 'latent_representations_training.png'")
print(f"Analyzed training image from class: {trainset.classes[train_label.item()]}")
print("First layer: 6 features extracted")
print("Second layer: 3 features extracted (indices 1, 8, 13)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("✅ Test set analysis: latent_representations_analysis.png")
print("✅ Training set analysis: latent_representations_training.png")
print("Both analyses use the same feature indices for comparison") 