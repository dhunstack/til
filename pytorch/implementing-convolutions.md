# Implementing convolution

Convolution operation for two functions f and g is written as 
$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t-\tau) d\tau
$$
 
In discrete domain, this becomes
$$
(f*g)[n] = \sum_{m=-\infty}^{\infty} f(n) \cdot g(m - n)
$$

In deep learning, what we commonly call convolutions are actually cross-correlations. In convolutions, the function g (filter/kernel) needs to be flipped and shifted.
Since DL involves learning these filters, the filters that get learnt are identical, except its flipped and shifted.

DL Convolutions (cross correlations) can be written as 
$$
[H]_{i,j,d} = \sum_{a=-\Delta}^{\Delta} \sum_{b=-\Delta}^{\Delta} \sum_{c} [V]_{a,b,c, d} \cdot [X]_{i+a, j+b, c}
$$

These can be implemented in Python as

```python
def corr2d(X, K):
    kh, kw = K.shape
    ih, iw = X.shape

    Y = torch.zeros(ih-kh+1, iw-kw+1)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            H[i,j] = (X[i:i+kh, j:j+kw] * K).sum()
        
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        out = corr2d(x, self.weight) + self.bias
        return out
```

The above example code is for single channel 2D input.

Below is actual usage example from PyTorch,
attempting to learn the kernel K.
```python
X = torch.ones(6,8)
X[:, 2:6] = 0
K = torch.Tensor([[1, -1]])
Y = corr2d(X, K)

conv2d = nn.LazyConv2d(1, kernel_size=(1,2), bias=False)

X = X.reshape((1,1,6,8))
Y = Y.reshape((1,1,6,7))
lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y-Y_hat)**2
    conv2d.zero_grad()
    l.sum().backward()

    conv2d.weight.data -= lr*conv2d.weight.grad

    if (i%2==0):
        print(f"epoch {i+1}, loss {l.sum():.3f}")
```
