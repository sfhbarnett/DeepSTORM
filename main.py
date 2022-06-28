import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch
from PIL import Image
from torchvision import transforms
import re

from DeepSTORM.Datastore import Datastore
from DeepSTORM.TiffStack import TiffStack
from DeepSTORM.deepstormnet import DeepSTORM

plt.rcParams['figure.figsize'] = [18, 10]


def natsort(tosort):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(tosort, key=alphanum_key)


def DeepSTORMLoss(output, target, dev):
    # Custom loss function, mse and L1
    # loss = torch.mean(torch.mean(torch.mean(torch.pow(output - target, 2), 2), 2) + torch.mean(torch.mean(torch.abs(output-target),2),2))
    criterion1 = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()
    loss = criterion1(output, target) + criterion2(target, torch.zeros(target.shape).to(dev))
    return loss


def train():
    # plt.ion()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(dev)

    tforms = transforms.Compose([transforms.ToTensor()])
    dataset = Datastore(images, spikes, transforms=tforms)
    # train_set, val_set = torch.utils.data.random_split(dataset, [len(images)*0.7, len(images)*0.3])

    net = DeepSTORM()
    net = net.to(dev)

    N_train = 9500
    batch_N = 16
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_N, shuffle=True, num_workers=4, pin_memory=True)
    # valloader = torch.utils.data.DataLoader(valset, batch_size=batch_N, shuffle=True, num_workers=4,pin_memory=True)
    epochs = 100
    lr = 0.001
    startepoch = 0

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = DeepSTORMLoss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                           min_lr=0.00000001, verbose=True)

    blurrer = transforms.GaussianBlur(kernel_size=(7, 7), sigma=(1, 1))

    # fig = plt.figure(figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
    # fig.tight_layout()
    # Try and load in previous checkpoint
    print('hello')
    try:
        checkpoint = torch.load('/content/gdrive/MyDrive/Deepstormdata/model_246.pt')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        startepoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print('Model loaded')
    except FileNotFoundError:
        print(f"No model file found")

    a = 0

    for epoch in range(startepoch, epochs):
        net.train()
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data['image']
            inputs = inputs.permute(0, 1, 2, 3)
            masks = data['mask']
            inputs = inputs.to(dev)
            masks = masks.to(dev)

            optimizer.zero_grad()
            predicted_masks = net(inputs)

            masks = masks * 100
            predicted_masks = blurrer(predicted_masks)

            loss = criterion(predicted_masks, masks, dev)
            epoch_loss += loss.item()
            a += 1
            # Plot to monitor progress
            if a > 25:
                plt.subplot(1, 4, 1)
                plt.title("Input")
                im = plt.imshow(inputs[0].cpu().permute(1, 2, 0).detach().numpy().squeeze())
                plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.1)
                plt.subplot(1, 4, 2)
                plt.title("Ground Truth")
                im = plt.imshow(masks[0].cpu().detach().numpy().squeeze())
                plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.1)
                plt.subplot(1, 4, 3)
                plt.title("Prediction")
                im = plt.imshow(predicted_masks[0].cpu().detach().numpy().squeeze())
                plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.1)
                plt.subplot(1, 4, 4)
                plt.title("Prediction overlay")
                scale1 = predicted_masks[0].cpu().detach().numpy().squeeze()
                scale1 = (scale1 - np.min(scale1)) / (np.max(scale1) - np.min(scale1)) * 255
                filler = np.zeros(scale1.shape).astype('uint8')
                scale1rgb = np.dstack((scale1.astype('uint8'), filler, filler))
                scale2 = masks[0].cpu().detach().numpy().squeeze()
                scale2 = (scale2 - np.min(scale2)) / (np.max(scale2) - np.min(scale2)) * 255
                scale2rgb = np.dstack((filler, scale2.astype('uint8'), filler))
                blend = Image.blend(Image.fromarray(scale1rgb).convert('RGBA'),
                                    Image.fromarray(scale2rgb).convert('RGBA'), 0.5)
                im = plt.imshow(blend)
                plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.1)
                fig = plt.gcf()
                fig.set_size_inches(18, 6)
                plt.show()
                plt.draw()
                plt.pause(0.0001)
                a = 0

            # print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_N / N_train, loss.item()))
            print(
                f'Epoch: {epoch} --- LR: {optimizer.param_groups[0]["lr"]} --- progress: {round((i + 1) * batch_N / N_train, 4)} --- loss: {round(epoch_loss / ((i + 1) * batch_N / N_train), 4)}')

            loss.backward()
            optimizer.step()

        modelsavepath = '/content/gdrive/MyDrive/Deepstormdata/model_246.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scheduler_state_dict': scheduler.state_dict(),
        }, modelsavepath)
        print("model saved")

        # Schedule learning rate
        scheduler.step(epoch_loss)


def predict():
    torch.cuda.empty_cache()
    filelist = '/content/gdrive/MyDrive/Deepstormdata/model_246.pt'
    imagepath = '/content/gdrive/MyDrive/Deepstormdata/test/BIN4_glia_actin_2D.tif'
    ts = TiffStack(imagepath)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batchsize = 3
    blurrer = transforms.GaussianBlur(kernel_size=(7, 7), sigma=(1, 1))
    net = DeepSTORM()
    net.eval()
    net = net.to(dev)

    # filelist = natsort(filelist)
    try:
        checkpoint = torch.load(filelist)
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Model loaded')
    except FileNotFoundError:
        print(f"No model file found")

    tforms = transforms.Compose([transforms.ToTensor()])
    batch_N = 3
    output = np.zeros((2048, 2048))
    batch = torch.zeros(batch_N, 1, ts.width * ts.upsample, ts.height * ts.upsample)
    batch = batch.to(dev)
    fig = plt.figure(figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
    # fig.tight_layout()
    path = '/content/gdrive/MyDrive/Deepstormdata/locs2.csv'
    with open(path, 'w') as f:
        with torch.no_grad():
            for index in range(0, ts.nfiles, batch_N):
                torch.cuda.empty_cache()
                print(index)
                for b in range(batch_N):
                    image = ts.getimageupsampled(index)
                    timg1 = tforms(image)
                    timg1 -= torch.min(timg1)
                    timg1 /= torch.max(timg1)
                    timg1 = (timg1 - ts.mean) / ts.std
                    batch[b, :, :, :] = timg1

                prediction = blurrer(net(batch))

                thresh = 7
                condition = prediction > thresh
                # sum up all sub-diffraction frames
                # output += torch.sum(prediction,dim=0).cpu().numpy().squeeze()
                # if index % 10 == 0:
                #     tifffile.imwrite('/content/gdrive/MyDrive/Deepstormdata/output.tif',output)
                # move to cpu to stop memory problems
                locs = condition.nonzero().tolist()
                locshold = np.asarray([[x[0] + index, x[2], x[3]] for x in locs])
                # Plot image and points for inspection
                # plt.imshow(prediction[0].cpu().numpy().squeeze())
                # plt.plot(locshold[:,2],locshold[:,1],'ro',markerfacecolor='none')
                # plt.pause(0.01)
                # time.sleep(10)
                f.write("\n".join([str(x).strip('[').strip(']') for x in locshold.tolist()]))


if __name__ == '__main__':
    predict()