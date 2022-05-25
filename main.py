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


def DeepSTORMLoss(output, target):
    #Custom loss function, mse and L1
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
lr = 0.0001
startepoch = 0


optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = DeepSTORMLoss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5,min_lr=0.00000001,verbose=True)

blurrer = transforms.GaussianBlur(kernel_size=(7,7),sigma=(1,1))

fig = plt.figure(figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
fig.tight_layout()

try:
    checkpoint = torch.load('model.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    startepoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print('Model loaded')
except FileNotFoundError:
    print(f"No model file found")
a = 1
for epoch in range(startepoch, epochs):
    print(f'Epoch number: {epoch}')
    epoch_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs = data['image']
        inputs = inputs.permute(0, 1, 2, 3)
        masks = data['mask']

        optimizer.zero_grad()
        predicted_masks = net(inputs)

        #masks = blurrer(masks)*100.0
        masks = masks*100
        predicted_masks = blurrer(predicted_masks)

        loss = criterion(predicted_masks, masks)
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
            modelsavepath = 'model_'+str(epoch)+'_'+str(a)+'.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, modelsavepath)
            a += 1

            print(f'Model saved at {modelsavepath}')
    scheduler.step(epoch_loss)