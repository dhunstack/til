# Calculating parameters in AlexNet

There's some debate over input size of AlexNet, if it's 224 or 227.
This creates a significant difference in the number of parameters in `fc1` layer, a difference of almost `11 million` parameters.
Based on the original paper, PyTorch forum [link](https://discuss.pytorch.org/t/alexnet-input-size-224-or-227/41272/2?u=dhunstack) and Wikipedia [entry](https://en.wikipedia.org/wiki/AlexNet), I will work with 224 input size.

```
conv1: (11*11)*3*96 + 96 = 34944

conv2: (5*5)*96*256 + 256 = 614656

conv3: (3*3)*256*384 + 384 = 885120

conv4: (3*3)*384*384 + 384 = 1327488

conv5: (3*3)*384*256 + 256 = 884992

fc1: (5*5)*256*4096 + 4096 = 26218496

fc2: 4096*4096 + 4096 = 16781312

fc3: 4096*1000 + 1000 = 4097000
```

This results in a total number of 50844008 parameters.

The exact memory requirements during training would also need to account for gradients, momentum, intermediate layer outputs etc. Details in slide 16 of [this](https://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D2L1-memory.pdf) presentation.

Below figure calculates number of parameters and estimates memory requirements at `FP32` precision.
Manual Calculation, only weights no bias parameters.
![Alexnet Parameters](./alexnet-parameters.png)