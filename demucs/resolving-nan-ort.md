# Resolving nan outputs in ONNX Runtime inference

After overloading Demucs inference script to use ONNX Runtime, I encountered a strange error.
While the PyTorch model worked perfectly, the ONNX exported version gave all `nan` when run with ONNX Runtime.
I'll document my debugging process below to resolve the issue.

## Tracing through ONNX Graph computation

Debugging through ONNX Runtime execution is not as straightforward as deubgging PyTorch's forward pass.
One way is to modify the ONNX graph to capture intermediate results in the final outputs, as described [here](https://github.com/onnx/onnx/issues/3277#issuecomment-1050600445).
But in my case since I wasn't even sure of which nodes to track due to all `nan` outputs, I used the method described [here](https://onnx.ai/sklearn-onnx/auto_tutorial/plot_fbegin_investigate.html#python-runtime-to-look-into-every-node).

This is a Python Runtime to evaluate ONNX graph using numpy. Setting `verbose=3` traces the model's executions with intermediate output ranges displayed.

```python
from onnx.reference import ReferenceEvaluator
oinf = ReferenceEvaluator(onnx_model, verbose=3)
oinf.run(None, {ort_session.get_inputs()[0].name: audio.cpu().numpy()})
```

I was quickly able to locate the first instance of nan in the execution log.

```
Div(/Gather_43_output_0, /Add_5_output_0) -> /Div_6_output_0
 + /Div_6_output_0: float32:(8, 2049, 340) in [nan, nan]
Atan(/Div_6_output_0) -> /Atan_output_0
 + /Atan_output_0: float32:(8, 2049, 340) in [nan, nan]
```

## Fixing erroneous PyTorch code

This corresponds to the `phase = torch.atan2(imag,phase)` call in ISTFT forward pass.
Internally, ONNX calculates `Div(imag,phase)` before `Atan`, and `Div` gets nan values probably due to some `phase` values being close to 0.

I modified the code to `phase = torch.atan2(imag, real + 1e-7)`, based on the discussion [here](https://discuss.pytorch.org/t/how-to-avoid-nan-output-from-atan2-during-backward-pass/176890).
This fixed the `nan` issue, as evidenced in the new execution log

```
Div(/Gather_43_output_0, /Add_5_output_0) -> /Div_6_output_0
 + /Div_6_output_0: float32:(8, 2049, 340) in [-3613678.0, 9165606.0]
Atan(/Div_6_output_0) -> /Atan_output_0
 + /Atan_output_0: float32:(8, 2049, 340) in [-1.570796012878418, 1.570796251296997]
 ```


## Fixing ONNX Reference Evaluator class

Running `ReferenceEvaluator` with `verbose=3` leads to `ValueError` sometimes during execution, which doesn't happen with `verbose=2`.

This is a subtle but known issue when using `onnx.reference.ReferenceEvaluator(verbose=3)`: `verbose=3` enables tensor value logging, which may trigger additional NumPy operations (like .min() or .max()) for display that wouldn’t run during normal model execution. If tensor has zero elements, tensor.min() or tensor.max() raises:
```python
ValueError: zero-size array to reduction operation minimum which has no identity
```

To make it work, I modified the local `onnx/reference/reference_evaluator.py` to wrap logging like this:

```python
# return f"{a.dtype}:{a.shape} in [{a.min()}, {a.max()}]"
try:
    return f"{a.dtype}:{a.shape} in [{a.min()}, {a.max()}]"
except ValueError:
    return f"{a.dtype}:{a.shape} in [(empty tensor)]"
```