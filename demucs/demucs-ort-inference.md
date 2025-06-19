# Demucs ORT Inference

Today I adapted the PyTorch inference scripts to work with ONNX Runtime.

## Investigating apply_model function

The `apply_model` function handles chunking with overlaps and inteprolation between chunks, as well as the "shift trick".
I overloaded the function to work with `ort` inference as well, by calling the `ort_session` in place of torch model.

## ONNX Runtime Creation

```python
import onnx
import onnxruntime as ort

# Load the ONNX model
model_path = os.path.join(model_dir, f"{model_name}.onnx")
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
# Create an ONNX Runtime session
ort_session = ort.InferenceSession(model_path)
```

## ONNX Runtime Inference

```python
if onnx_flag:
    # Prepare input for ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: padded_mix.cpu().numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]
    out = th.tensor(ort_out, device=device)
else:
    # Use the PyTorch model
    out = model(padded_mix)
```
