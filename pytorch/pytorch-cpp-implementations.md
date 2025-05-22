# Locate Cpp implementations in PyTorch

This week I learnt how to locate PyTorch's internal core functions in C++. IDEs are not automatically able to locate the C++ implementations by following references.

I encountered this issue while translating Demucs' default Pytorch stft call to custom stft implementation. I wanted to figure out the role of `normalized` parameter in `stft()`. While the documentation had a one line description, I decided to double check my understanding with the actual implementation. Since `stft()` had been implemented inside C++, I needed to look up the ways to reach the implementation since my IDE was unable to find the reference.

I found one good method to locate the C++ references using this method I found on PyTorch forums [here](https://dev-discuss.pytorch.org/t/how-to-find-the-c-cuda-implementation-of-specific-operators-in-pytorch-source-code/1551/3)

## Steps I followed

### Step 1: Explored Python API

torch.stft() is defined in Python, but it calls into C++ through the dispatcher:
Python wrapper is in:

- torch/functional.py `stft()` wrapper [link](https://github.com/pytorch/pytorch/blob/aec7cc60d7d290dfe143c7a67fdb5eea4f1ecb43/torch/functional.py#L557)
- It uses: _VF.stft(...), that stands for torch._C._VariableFunctions, which routes to C++.

### Step 2: Search in native_functions.yaml

Searched for `stft` function in the yaml file, [link](https://github.com/pytorch/pytorch/blob/aec7cc60d7d290dfe143c7a67fdb5eea4f1ecb43/aten/src/ATen/native/native_functions.yaml#L5777)

### Step 3: Find the CPP implementation

The CPU implementation is in `SpectralOps` [file](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SpectralOps.cpp)

The STFT function is [here](https://github.com/pytorch/pytorch/blob/aec7cc60d7d290dfe143c7a67fdb5eea4f1ecb43/aten/src/ATen/native/SpectralOps.cpp#L826)

## Understanding normalized behaviour

This is the relevant section of code I found.

```
  const fft_norm_mode norm = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none;
```

`normalized` is being used to set the `norm` variable, which can either take the value `fft_norm_mode::by_root_n` if `True` and `fft_norm_mode::none` if `False`. 

It corresponds to unitary normalization, which scales the FFT by 1/sqrt(n) where n is the FFT size. This is standard in many DSP and mathematical libraries to preserve signal energy in the frequency domain, based on Perserval's theorem.

Since Demucs uses `normalized=True` in its implementation, we will implement this normalization in our custom STFT as well.
That will be explored in another blog post.