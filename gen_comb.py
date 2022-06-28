
import numpy as np
import matplotlib.pyplot as plt


XB = np.linspace(0,7,8)
YB = np.linspace(0,7,8)
X,Y = np.meshgrid(XB,YB)
Z = np.exp(-((X-3.1)**2+(Y-3.4)**2))
plt.imshow(Z,interpolation='none')#
plt.show()