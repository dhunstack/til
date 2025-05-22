# Reimplementing Pytorch STFT

Since I've been working on the project to export Demucs to ONNX, I need to find a way to reimplement `torch.stft` and `torch.istft` functions, without using complex tensors. 

ONNX currently doesn't support complex tensors [<sup> 1 </sup>][1]. PyTorch has deprecated `return_complex=False` parameter for `stft` so we can't use the inbuilt function.

I found an alternate implementation [<sup> 2 </sup>][2], but it had a few differences in the parameter values compared to Demucs. Additionally, it didn't support the `normalize=True` parameter used in Demucs' torch stft call, as described in my [previous post][3].

## Steps I followed

### Step 1 - Add parameter in function API

Added support for `normalize` parameter in the constructor of `STFT_Process` class.

```
NORMALIZED = True

class STFT_Process(torch.nn.Module):
    def __init__(self, model_type, n_fft=NFFT, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE, normalized=NORMALIZED):
```

### Step 2 - Normalize window functions if required

```
# Calculate window function
cos_kernel = (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
sin_kernel = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)

# Normalize window if needed
if normalized:
    cos_kernel_new = cos_kernel / torch.sqrt(torch.tensor([n_fft], dtype=torch.float32))
    sin_kernel_new = sin_kernel / torch.sqrt(torch.tensor([n_fft], dtype=torch.float32))
```

### Step 3 - Update tests to verify the implementation

```
def test_onnx_stft_B(input_signal):
    torch_stft_output = torch.view_as_real(torch.stft(
        input_signal.squeeze(0),
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        window=WINDOW,
        win_length=NFFT,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode=PAD_MODE,
    ))
    pytorch_stft_real = torch_stft_output[..., 0].squeeze().numpy()
    pytorch_stft_imag = torch_stft_output[..., 1].squeeze().numpy()

    stft_model = STFT_Process(
        model_type=STFT_TYPE,
        n_fft=NFFT,
        hop_len=HOP_LENGTH,
        max_frames=MAX_SIGNAL_LENGTH,
        window_type=WINDOW_TYPE,
        normalized=True
    ).eval()

    with torch.no_grad():
        stft_output = stft_model(input_signal)
        # stft_output = torch.view_as_real(stft_output)
        onnx_stft_real = stft_output[..., 0].squeeze().numpy()
        onnx_stft_imag = stft_output[..., 1].squeeze().numpy()

    mean_diff_real = np.abs(pytorch_stft_real - onnx_stft_real).mean()
    mean_diff_imag = np.abs(pytorch_stft_imag - onnx_stft_imag).mean()
    mean_diff = (mean_diff_real + mean_diff_imag) * 0.5
    print("\nSTFT Result: Mean Difference =", mean_diff)
```

## Result 

The normalization implementation is verified to be correct and also resulting in reduction of mean error compared to the original unnormalized implementation.

```
Testing the Custom.STFT normalized versus Pytorch.STFT normalized ...

STFT Result: Mean Difference = 1.1999522e-05

Testing the Custom.STFT unnormalized versus Pytorch.STFT unnormalized ...

STFT Result: Mean Difference = 0.00038578542

Testing the Custom.STFT normalized versus Pytorch.STFT unnormalized ...

STFT Result: Mean Difference = 10.576519
```

## References

[1]: https://github.com/pytorch/pytorch/issues/126972 "ONNX complex tensor issue"
[2]: https://github.com/DakeQQ/STFT-ISTFT-ONNX "STFT ONNX implementation"
[3]: ./pytorch-cpp-implementations.md "Previous post"