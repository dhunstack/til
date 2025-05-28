# LeNet

This post continues the series of revisiting convolutions.

LeNet (LeNet-5) consists of two parts:

- a convolutional encoder consisting of two convolutional layers; and
- a dense block consisting of three fully connected layers.

Python implementation follows 

```python
class LeNet(nn.Module):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(    # (B, 1,28,28)
            nn.LazyConv2d(6,kernel_size=5, padding=2), nn.Sigmoid(), # (B, 6,28,28)
            nn.AvgPool2d(kernel_size=2, stride=2),  # (B, 6,14,14)
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),  # (B, 16,10,10)
            nn.AvgPool2d(kernel_size=2, stride=2),  # (B, 16,5,5),
            nn.Flatten(),   # (B, 400)
            nn.LazyLinear(120), nn.Sigmoid(),  # (B, 120)
            nn.LazyLinear(84), nn.Sigmoid(),    # (B, 84)
            nn.LazyLinear(num_classes)    # (B, 10)
        )
```

To visualize the class layers, this snippet is useful

```python
@d2l.add_to_class(LeNet)
def layer_summary(self, X_shape):
    X = torch.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
model = LeNet()
model.layer_summary((1, 1, 28, 28))
```
