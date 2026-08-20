# Experiment template

Copy the relevant JSON file to `experiments/<name>/metrics.json`, then add:

```text
forward.py
weights.url
weights.sha256
dataset.json
conclusion.md
CPU
```

`forward.py` is model-specific and cannot be generated safely: it must build the
real architecture, strictly load the committed checkpoint digest and declare the
real batch-1 `INPUT_SHAPE`. Validate the completed directory with
`make validate EXPERIMENT=experiments/<name>`.

`dataset.json` selects only the validation prefixes required from the team
dataset bucket. `forward.py` must also implement
`evaluate_quality(predict, dataset_root)` and compute the task metric on that
fixed subset for both PyTorch and ONNX Runtime.

Start with `CPU`. Add `GPU` only in a later pull request after the CPU candidate
has been merged; a candidate must never contain both markers.
