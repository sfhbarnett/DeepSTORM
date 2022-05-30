import numpy as np
import matplotlib.pyplot as plt
import tifffile

from deepstormnet import DeepSTORM
import Datastore
from torch import optim
import torch
from PIL import Image
from torchvision import transforms
import os
import re

def natsort(tosort):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(tosort, key=alphanum_key)


def DeepSTORMLoss(output, target):
    #Custom loss function, mse and L1
    loss = torch.mean(torch.mean(torch.mean(torch.pow(output - target, 2), 2), 2) + torch.mean(torch.mean(torch.abs(output-target),2),2))
    return loss


def train():

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
    # Try and load in previous checkpoint

    try:
        checkpoint = torch.load('model_21_150.pt')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        startepoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('Model loaded')
    except FileNotFoundError:
        print(f"No model file found")

    a = 151

    for epoch in range(startepoch, epochs):
        print(f'Epoch number: {epoch}')
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data['image']
            inputs = inputs.permute(0, 1, 2, 3)
            masks = data['mask']

            optimizer.zero_grad()
            predicted_masks = net(inputs)

            masks = masks*100
            predicted_masks = blurrer(predicted_masks)

            loss = criterion(predicted_masks, masks)
            epoch_loss += loss.item()

            # Plot to monitor progress
            plt.subplot(1, 4, 1)
            plt.title("Input")
            im = plt.imshow(inputs[0].permute(1, 2, 0).detach().numpy().squeeze())
            plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
            plt.subplot(1, 4, 2)
            plt.title("Ground Truth")
            im = plt.imshow(masks[0].detach().numpy().squeeze())
            plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
            plt.subplot(1, 4, 3)
            plt.title("Prediction")
            im = plt.imshow(predicted_masks[0].detach().numpy().squeeze())
            plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
            plt.subplot(1, 4, 4)
            plt.title("Prediction overlay")
            scale1 = predicted_masks[0].detach().numpy().squeeze()
            scale1 = (scale1 - np.min(scale1)) / (np.max(scale1) - np.min(scale1))*255
            filler = np.zeros(scale1.shape).astype('uint8')
            scale1rgb = np.dstack((scale1.astype('uint8'),filler,filler))
            scale2 = masks[0].detach().numpy().squeeze()
            scale2 = (scale2 - np.min(scale2)) / (np.max(scale2) - np.min(scale2)) * 255
            scale2rgb = np.dstack((filler,scale2.astype('uint8'),filler))
            blend = Image.blend(Image.fromarray(scale1rgb).convert('RGBA'),Image.fromarray(scale2rgb).convert('RGBA'),0.5)
            im = plt.imshow(blend)
            plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
            plt.show()
            plt.draw()
            plt.pause(0.0001)
            plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_N / N_train, loss.item()))

            loss.backward()
            optimizer.step()

            # Checkpoint every 100 images
            if i % 100 == 0:
                modelsavepath = 'model_'+str(epoch)+'_'+str(a)+'.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'scheduler_state_dict': scheduler.state_dict(),
                }, modelsavepath)
                a += 1

                print(f'Model saved at {modelsavepath}')
        # Schedule learning rate
        scheduler.step(epoch_loss)

def predict():
    filelist = os.listdir('/Users/sbarnett/PycharmProjects/DeepSTORM/DeepSTORM/models')
    imagepath = '/Users/sbarnett/PycharmProjects/DeepSTORM/DeepSTORM/DeepSTORM dataset_v1/BIN4_glia_actin_2D.tif'
    ts = Datastore.TiffStack(imagepath)
    net = DeepSTORM()
    net.eval()
    filelist = natsort(filelist)
    print(filelist[-1])
    try:
        checkpoint = torch.load(os.path.join('models/', filelist[-1]))
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Model loaded')
    except FileNotFoundError:
        print(f"No model file found")

    tforms = transforms.Compose([transforms.ToTensor()])

    output = np.zeros((ts.width*ts.upsample,ts.height*ts.upsample),dtype='float32')

    for index in range(100):
        print(index)
        image = ts.getimageupsampled(index)
        timg1 = tforms(image)
        timg1 -= torch.min(timg1)
        timg1 /= torch.max(timg1)
        timg1 = (timg1 - ts.mean) / ts.std
        timg1 = timg1.unsqueeze(0)
        prediction = net(timg1)
        output += prediction[0].detach().numpy().squeeze()

    tifffile.imwrite('output.tif',output)



    # net.eval()
    # for filename in filelist[1:]:
    #     print(filename)
    #     img = np.array(Image.open(os.path.join(mainpath, filename)))
    #     tforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    #     img = tforms(img)
    #     img = img.unsqueeze(0)
    #
    #     with torch.no_grad():
    #         prediction = net(img)
    #         # plt.imshow(prediction[0].detach().numpy().squeeze())
    #         # plt.show()
    #         predicted = prediction[0].detach().numpy().squeeze()
    #         predicted[predicted <= 0] = 0
    #         predicted = Image.fromarray(predicted)
    #         predicted.save(os.path.join(mainpath, filename[:-4]+'_predicted.tif'))
    #


if __name__ == '__main__':
    predict()