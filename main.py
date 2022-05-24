from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from deepstormnet import DeepSTORM
import Datastore
from torch import optim, nn
import torch
from PIL import Image
from torchvision import transforms


def DeepSTORMLoss(output, target,crit1,crit2):
    #print(output)
    #loss = torch.mean(torch.mean(torch.pow(output-target,2)))
    #loss = torch.mean(torch.mean(torch.pow(output - target,2)) + torch.linalg.matrix_norm(output, ord=1))
    #loss = mse(output,target) + l1(output,torch.zeros(target.shape))
    #loss = mse(output,target)
    #loss = crit1(output,target) + crit2(output,torch.zeros(target.shape))
    #loss = torch.mean(output,0)
    #torch.mean(torch.mean(torch.pow(output - target, 2), 2), 2)
    loss = torch.mean(torch.mean(torch.mean(torch.pow(output - target, 2), 2), 2) + torch.mean(torch.mean(torch.abs(output-target),2),2))
    return loss


plt.ion()

tforms = transforms.Compose([transforms.ToTensor()])
dataset = Datastore.Datastore(transforms=tforms)

net = DeepSTORM()


N_train = 9500
batch_N = 16
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_N, shuffle=True, num_workers=0)

epochs = 100
lr = 0.00001


optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = DeepSTORMLoss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5,min_lr=0.00000001,verbose=True)
msecriterion = nn.MSELoss()
l1criterion = nn.L1Loss()

criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss()

blurrer = transforms.GaussianBlur(kernel_size=(7,7),sigma=(1,1))

fig = plt.figure(figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
fig.tight_layout()

try:
    checkpoint = torch.load('model.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print('Model loaded')
except FileNotFoundError:
    print(f"No model file found")

for epoch in range(epochs):
    epoch_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs = data['image']
        inputs = inputs.permute(0, 1, 2, 3)
        masks = data['mask']

        optimizer.zero_grad()
        predicted_masks = net(inputs)

        masks = blurrer(masks)*100.0
        predicted_masks = blurrer(predicted_masks)

        loss = criterion(predicted_masks, masks,criterion1,criterion2)
        #loss = msecriterion(predicted_masks, masks) + l1criterion(predicted_masks, torch.zeros(masks.shape))
        epoch_loss += loss.item()

        plt.subplot(1, 4, 1)
        plt.title("Input")
        im = plt.imshow(inputs[0].permute(1, 2, 0).detach().numpy().squeeze())
        plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
        plt.subplot(1, 4, 2)
        plt.title("Mask")
        im = plt.imshow(masks[0].detach().numpy().squeeze())
        plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
        plt.subplot(1, 4, 3)
        plt.title("Prediction")
        im = plt.imshow(predicted_masks[0].detach().numpy().squeeze())
        plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
        plt.subplot(1, 4, 4)
        plt.title("Prediction")
        im = plt.imshow(predicted_masks[0].detach().numpy().squeeze(),vmin=0)
        plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
        plt.show()
        plt.draw()
        plt.pause(0.0001)
        plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)

        print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_N / N_train, loss.item()))

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            modelsavepath = 'model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, modelsavepath)

            print(f'Model saved at {modelsavepath}')
    scheduler.step(epoch_loss)