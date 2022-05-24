import torch.nn as nn

class DeepSTORM(nn.Module):
    def __init__(self):
        super(DeepSTORM, self).__init__()
        self.down1 = Down(1, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.midsection = nn.Conv2d(128, 512, kernel_size=(3, 3), padding=1, padding_mode='replicate', bias=False)
        self.up1 = Up(512, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), padding_mode='replicate', bias=False)

    def forward(self, x):
        down = self.down1(x)
        down = self.down2(down)
        down = self.down3(down)
        mid = self.midsection(down)
        up = self.up1(mid)
        up = self.up2(up)
        up = self.up3(up)
        up = self.up4(up)
        return up



class Down(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Down, self).__init__()
        self.level = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.level(x)
        return x

class Up(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Up, self).__init__()
        #self.up = nn.ConvTranspose2d(inchannels, inchannels, stride=2, kernel_size=2)
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
