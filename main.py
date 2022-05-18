from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from deepstormnet import DeepSTORM
import Datastore
from torch import optim, nn
import torch


plt.ion()

tforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
dataset = Datastore.Datastore(transforms=tforms)

net = DeepSTORM()

batch_N = 1
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_N, shuffle=True, num_workers=0)

epochs = 1
lr = 0.001
N_train=100

optimizer = optim.SGD(net.parameters(),
                      lr=lr,
                      momentum=0.9)
criterion = nn.BCEWithLogitsLoss()

fig = plt.figure(figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
fig.tight_layout()

for epoch in range(epochs):
    epoch_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs = data['image']
        inputs = inputs.permute(0, 1, 2, 3)
        masks = data['mask']
        print(masks.shape)
        optimizer.zero_grad()
        predicted_masks = net(inputs)
        loss = criterion(predicted_masks.view(-1), masks.contiguous().view(-1))
        epoch_loss += loss.item()

        plt.subplot(1, 3, 1)
        plt.title("Input")
        im = plt.imshow(inputs[0].permute(1, 2, 0).detach().numpy().squeeze())
        plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
        plt.subplot(1, 3, 2)
        plt.title("Mask")
        im = plt.imshow(masks[0].detach().numpy().squeeze())
        plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        im = plt.imshow(predicted_masks[0].detach().numpy().squeeze())
        plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
        plt.show()
        plt.draw()
        plt.pause(0.0001)
        plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)

        print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_N / N_train, loss.item()))