# Pooling

This post continues the series of revisiting convolutions.
Pooling is a common operation used in ConvNets for aggregating information.
It serves the dual purpose of mitigating the sensitivity of convolutional layers to location and of spatially downsampling representations.
Pooling implementation is very similar to convolutions.

```python
def pool2d(X, pool_size, mode='max'):
    kh, kw = pool_size.shape
    xh, xw = X.shape
    Y = torch.zeros((xh-kh+1, xw-kw+1))

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=='avg':
                Y[i,j] = X[i:i+kh, j:j+kw].mean()
            elif mode=='max':
                Y[i,j] = X[i:i+kh, j:j+kw].max()
    return Y
```

Pooling implementation in PyTorch also supports strides and padding like convolutions.

```python
X = torch.arange(16, dtype=torch.float32).reshape((1,1,4,4)) 
# Inputs to both Conv2d and MaxPool2d need to be tensors of shape (B, C, H, W)
# Pooling acts independently across channels, so don't need to mention channels inside
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X) # Output shape (1,1,2,2)
```
