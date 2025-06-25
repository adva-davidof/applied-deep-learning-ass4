import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import DeconvNet

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 30

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

class DeconvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        # Deconv layers
        self.deconv1 = nn.ConvTranspose2d(16, 6, 5)
        self.deconv2 = nn.ConvTranspose2d(6, 3, 5)
        self.unpool = nn.MaxUnpool2d(2, 2)

        # Fully-Connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Conv part
        x = F.relu(self.conv1(x))
        size1 = x.size()
        x, indices1 = self.pool(x)

        x = F.relu(self.conv2(x))
        size2 = x.size()
        x, indices2 = self.pool(x)

        # Fully-Connected part
        y = torch.flatten(x, 1) # flatten all dimensions except batch
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)

        # Deconv part
        deconved_x = self.deconv1(F.relu(self.unpool(x, indices2, output_size=size2)))
        deconved_x = self.deconv2(F.relu(self.unpool(deconved_x, indices1, output_size=size1)))

        return y, deconved_x

deconvNet = DeconvNet()

criterion_ce = torch.nn.CrossEntropyLoss()
criterion_rec = torch.nn.MSELoss()

Lambda = 10 # "Lambda" "parameter for the hybrid loss (the weight of the Lrec loss)
optimizer = torch.optim.Adam(deconvNet.parameters(), lr=LEARNING_RATE)
losses_for_plot = [[], []] # Value of loss and number of steps
steps = 0

deconvNet.train()
for epoch in range(EPOCHS):
    running_loss = 0.0

    for i, (images, labels) in enumerate(trainloader):
        steps += 1

        predictions, latents = deconvNet(images)
        loss_ce = criterion_ce(predictions, labels)
        loss_rec = criterion_rec(latents, images)

        optimizer.zero_grad()
        (loss_ce + Lambda*loss_rec).backward()
        optimizer.step()

        running_loss += (loss_ce + Lambda*loss_rec).item()
        if (i+1) % 100 == 0:
            losses_for_plot[0].append(running_loss/100)
            losses_for_plot[1].append(steps)

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

plt.plot(losses_for_plot[1], losses_for_plot[0], '-*')
plt.title('Task 2 - Conv "hybrid" Loss after t Steps')
plt.xlabel('Steps (t)')
plt.ylabel('Loss = Lce + l*Lrec')
plt.show()

deconvNet.eval()
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs, reconstructedImages = deconvNet(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

unnormalize = transforms.Normalize(mean=[-1.0, -1.0, -1.0],std=[2.0, 2.0, 2.0])

fig = plt.figure(constrained_layout=True)
(origFig, deconvFig) = fig.subfigures(2, 1)
axesOrigin = origFig.subplots(1, 3)
axesDeconv = deconvFig.subplots(1, 3)

fig.suptitle('Original and Reconstructed Image Comparison', fontsize='18')
origFig.suptitle('Original Images', color='b')
deconvFig.suptitle('Reconstructed Images', color='r')

for i in range(3):
    imgInx = i+6

    originalImg = unnormalize(images[imgInx])
    originalImg = np.clip(originalImg, 0, 1)
    deconvImg = unnormalize(reconstructedImages[imgInx])
    deconvImg = np.clip(deconvImg, 0, 1)
    originalImg = np.array(originalImg).transpose(1, 2, 0)
    deconvImg = np.array(deconvImg).transpose(1, 2, 0)

    axesOrigin[i].imshow(originalImg)
    axesDeconv[i].imshow(deconvImg)

plt.show()

torch.save(deconvNet.state_dict(), './deConvModel.pt')