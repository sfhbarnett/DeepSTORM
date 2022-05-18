from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
csvpath = 'DeepSTORM dataset_v1/BIN4 - Training dataset/SimulatedDataset.csv'

points = []

with open(csvpath, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.split(',')
        line[1] = int(float(line[1]))
        points.append(line)

width = 64
height = 64
pixel_size = 160
upsample = 8
mask = np.zeros([width*upsample,height*upsample])

for p in points:
    print(p)
    if p[1] == 3:
        x = round(float(p[2])//pixel_size)*upsample
        y = round(float(p[3])//pixel_size)*upsample
        mask[x, y] += 100

plt.imshow(mask)
plt.show()
