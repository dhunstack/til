# Testing STFT implementation

Today I worked on writing tests for new STFT implementation inside Demucs.
I had earlier adapted the custom STFT function to the format needed by Demucs, detailed in my [previous blog](../pytorch/reimplementing-pytorch-stft.md), and also written tests within that file. But I hadn't integrated it into Demucs before.

## Steps I followed

### Step 1: Writing sanity check tests for STFT output shapes

I wrote simple tests to check the shapes of STFT outputs for both Pytorch and custom STFT.
The sample code for PyTorch is below -

```python
def test_spectrogram_shape_pytorch():
    """ Test the shape of the spectrogram output using PyTorch's STFT """
    batch, channels, samples = 1, 2, 2*44100
    n_fft, hop_length = 4096, 1024
    waveform = torch.randn(batch, channels, samples)  # (batch, channels, samples)
    spec = spectro(waveform, n_fft=n_fft, hop_length=hop_length, pad=0, stft_type='pytorch')
    assert spec.shape == (1, 2, n_fft // 2 + 1, samples // hop_length + 1, 2)  # (batch, channels, freq_bins, time_steps, 2)
```

### Step 2: Error with the custom test

I got an error in the custom STFT test when used inside demucs due to the input format difference.
Custom stft accepts audio as a (batches, channels, samples) tensor, while [pytorch stft](https://docs.pytorch.org/docs/stable/generated/torch.stft.html) accepts 1D or 2D tensors (batches, samples).

So before passing the input for pytorch's stft to custom stft, I reshape it via `x.view(-1, 1, length)` based on the solutions discussed [here](https://discuss.pytorch.org/t/best-way-to-convolve-on-different-channels-with-a-single-kernel/16501/4?u=dhunstack).

### Step 3: Numerical Comparison Test

Finally I wrote the test to compare two implementations inside Demucs with a 20 second random waveform.

```python
def test_compare_spectrograms():
    """ Compare the spectrograms from PyTorch and Custom STFT """
    batch, channels, samples = 1, 2, 20*44100
    n_fft, hop_length = 4096, 1024
    waveform = torch.randn(batch, channels, samples)  # (batch, channels, samples)

    spec_pytorch = spectro(waveform, n_fft=n_fft, hop_length=hop_length, pad=0, stft_type='pytorch')
    spec_custom = spectro(waveform, n_fft=n_fft, hop_length=hop_length, pad=0, stft_type='custom')

    # Calculate the mean difference between the two spectrograms
    mean_diff = torch.abs(spec_pytorch - spec_custom).mean().item()
    print("\nSTFT Result: Mean Difference =", mean_diff)

    assert torch.allclose(spec_pytorch, spec_custom, atol=1e-3), "Spectrograms do not match!"
```

I've also been thinking of refactoring the custom STFT function since it's too verbose now. But this will be in another blog post.
