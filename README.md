# DeepSTORM
Implementation of DeepSTORM neural network for super-resolution imaging in PyTorch. Also ported to python is the matlab code in the original repositiory for generating the training data. Future developments could convert this to on-the-fly generation to avoid having to store large numbers of training images. Images below are partway through training and don't represent the final predictions.

Images left to right:
1) the input image, a diffraction limited image of single molecules
2) Ground truth, this is the location of individual molecules
3) Prediction of the molecular positions by the network
4) overlay of 2 (green) and 3 (red)

<img width="1174" alt="image" src="https://user-images.githubusercontent.com/45679976/170332343-01db1d7e-3b4e-4295-809e-5eace0928100.png">

# Results

On the left is the diffraction limited max projection of STORM dataset, on the right is the partially processed DeepSTORM image (660/10000 frames). 

<img width="425" alt="image" src="https://user-images.githubusercontent.com/45679976/172792642-477fa20a-71d0-4b9f-9343-28ff567771f4.png"> <img width="425" alt="image" src="https://user-images.githubusercontent.com/45679976/172792581-05d6d60f-38d5-4a22-8846-8c9513439d19.png">


Original Deep-STORM repository written in Keras/Tensorflow.
https://github.com/EliasNehme/Deep-STORM

Deep-STORM paper:
https://opg.optica.org/optica/fulltext.cfm?uri=optica-5-4-458&id=385495
