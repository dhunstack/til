# Padding and Strides

This post continues the series of revisiting convolutions.
Today we'll talk about padding and strides, why they're useful and how we can use it in PyTorch.
Convolutions without padding leads to continuous reduction of image size.
For a 240 x 240 image, 10 convolutions with kernel of size 5 will lead to image of 200 x 200, resulting in 30% size reduction.
Padding can be used to counter this size reduction.

Strides on the other hand are actually used for downsampling (or upsampling when s<1!), so a stride of 2 can result in reduction of image size by 1/2.

The general formula for output size of image of dim $(n_h, n_w)$ on applying filter size $(k_h, k_w)$ is $(n_h-k_h+1, n_w-k_w+1)$.

When padding $(p_h, p_w)$ is applied, we can think of it as effectively making the input image larger, so the formula can simply become $(n_h-k_h+p_h+1,n_w+p_w-k_w+1)$

Usually, we use $p_h=k_h-1$ to make the output size same as input size. This is also known as `same` convolution. In this case, if $k_h$ is odd, we can pad $p_h/2$ on top and bottom each. If $k_h$ is even, we can pad $\lfloor p_h/2 \rfloor+1$ on top and $\lfloor p_h/2 \rfloor$ on bottom.

When we also use stride $(s_h, s_w)$, the output size becomes $( \lfloor (n_h-k_h+p_h+s_h)/s_h \rfloor, \lfloor(n_w-k_w+p_w+s_w)/s_w \rfloor )$.

In case we use $p_h=k_h-1$, this becomes $\lfloor (n_h+s_h-1)/s_h \rfloor$. If $n_h$ is exactly divisible by $s_h$, this would imply the size of output is exactly $\lfloor n_h/s_h \rfloor$.

TLDR, general output size - $( \lfloor (n_h-k_h+p_h+s_h)/s_h \rfloor, \lfloor(n_w-k_w+p_w+s_w)/s_w \rfloor )$.

The documentation for `Conv2D` in PyTorch can be found [here](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).
The general syntax is `torch.nn.Conv2d(input_channels, output_channels, kernel_size, padding=0, stride=1, dilation=1, bias=True)`

We haven't discussed about dilation in this post but output size with dilation can be found in the linked `Conv2d` documentation.

Usage is as follows -

```python
def comp_conv2d(conv2d, X):
    # conv2d takes input of the shape (B, C, H, W)
    # so we reshape X
    X = X.reshape((1,1)+X.shape)
    Y = conv2d(X)
    # strip batch and channel dimensions
    return Y.reshape(Y.shape[2:])

conv2d = torch.nn.LazyConv2d(1, kernel_size = 3, padding = 1)
X = torch.rand((8,8))
print(comp_conv2d(conv2d, X).shape)
# torch.Size([8,8])
```

A more complicated example 

```python
conv2d = torch.nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)
# h - floor((8-3+2*0+3)/3) = 2
# w - floor((8-5+2*1+4)/4) = 2
# torch.Size([2,2])
```
