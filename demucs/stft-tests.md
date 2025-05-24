# Testing STFT implementation

Today I worked on writing tests for new STFT implementation inside Demucs.
I had earlier adapted the custom STFT function to the format needed by Demucs, detailed in my [previous blog](../pytorch/reimplementing-pytorch-stft.md).

I realised one key difference in the two implementations, custom stft accepts audio as a (batches, channels, samples) tensor, while [pytorch stft](https://docs.pytorch.org/docs/stable/generated/torch.stft.html) accepts 1D or 2D tensors (batches, samples).

So before passing the input for pytorch's stft to custom stft, I reshape it via `x.view(-1, 1, length)` based on the solutions discussed [here](https://discuss.pytorch.org/t/best-way-to-convolve-on-different-channels-with-a-single-kernel/16501/4?u=dhunstack).

I've also been thinking of refactoring the custom STFT function since it's too verbose now. But this will be in another blog post.