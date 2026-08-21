from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from model_bench.bundle import TENSOR_NAME
from model_bench.quality import normalize_task_metrics


def load_forward_module(path: Path) -> ModuleType:
    module_name = f"experiment_{path.parent.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import forward module: {path}")
    module = importlib.util.module_from_spec(spec)
    experiment_path = str(path.parent.resolve())
    repository_path = str(path.resolve().parents[2])
    sys.path.insert(0, experiment_path)
    sys.path.insert(0, repository_path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(repository_path)
        sys.path.remove(experiment_path)
    return module


def find_model_class(module: ModuleType) -> type[Any]:
    import torch

    classes = [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
        and issubclass(cls, torch.nn.Module)
        and cls is not torch.nn.Module
    ]
    if len(classes) != 1:
        raise ValueError(
            f"{module.__file__} must declare exactly one torch.nn.Module class; "
            f"found {len(classes)}"
        )
    return classes[0]


def instantiate_model(model_class: type[Any], weights_path: Path) -> Any:
    signature = inspect.signature(model_class)
    if "weights_path" in signature.parameters:
        return model_class(weights_path=weights_path)
    if len(signature.parameters) == 1:
        return model_class(weights_path)
    raise ValueError(f"{model_class.__name__} constructor must accept weights_path")


def input_names(module: ModuleType) -> tuple[str, ...]:
    configured = getattr(module, "INPUT_NAMES", None)
    if configured is None:
        configured = (getattr(module, "INPUT_NAME", "pixel_values"),)
    names = tuple(str(value) for value in configured)
    if not names or len(set(names)) != len(names):
        raise ValueError("INPUT_NAMES must contain unique names")
    if any(TENSOR_NAME.fullmatch(name) is None for name in names):
        raise ValueError("INPUT_NAMES contain a name unsupported by benchmark runtimes")
    return names


def output_names(module: ModuleType) -> tuple[str, ...]:
    configured = getattr(module, "OUTPUT_NAMES", None)
    if configured is None:
        configured = (getattr(module, "OUTPUT_NAME", "logits"),)
    names = tuple(str(value) for value in configured)
    if not names or len(set(names)) != len(names):
        raise ValueError("OUTPUT_NAMES must contain unique names")
    if any(TENSOR_NAME.fullmatch(name) is None for name in names):
        raise ValueError(
            "OUTPUT_NAMES contain a name unsupported by benchmark runtimes"
        )
    return names


def input_shapes(module: ModuleType) -> tuple[tuple[int, ...], ...]:
    configured = getattr(module, "INPUT_SHAPES", None)
    if configured is None:
        single_shape = getattr(module, "INPUT_SHAPE", None)
        if single_shape is None:
            raise ValueError("forward.py must define INPUT_SHAPE or INPUT_SHAPES")
        configured = (single_shape,)
    shapes = tuple(tuple(int(dimension) for dimension in shape) for shape in configured)
    if not shapes or any(
        not shape or any(value <= 0 for value in shape) for shape in shapes
    ):
        raise ValueError(f"Invalid INPUT_SHAPES: {shapes}")
    if any(shape[0] != 1 for shape in shapes):
        raise ValueError("Every benchmark input must have batch dimension 1")
    return shapes


def make_inputs(
    module: ModuleType,
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
    seed: int,
) -> tuple[Any, ...]:
    import torch

    factory = getattr(module, "make_inputs", None)
    if factory is not None:
        generated = factory(seed)
        if isinstance(generated, dict):
            try:
                values = tuple(generated[name] for name in names)
            except KeyError as exc:
                raise ValueError(f"make_inputs is missing input {exc.args[0]}") from exc
        elif isinstance(generated, (tuple, list)):
            values = tuple(generated)
        else:
            values = (generated,)
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values = tuple(
            torch.randn(*shape, dtype=torch.float32, generator=generator)
            for shape in shapes
        )
    if len(values) != len(names):
        raise ValueError(f"Expected {len(names)} model inputs, got {len(values)}")
    supported_dtypes = {
        torch.bool,
        torch.float16,
        torch.float32,
        torch.int8,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    normalized = []
    for name, shape, value in zip(names, shapes, values, strict=True):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Model input {name!r} must be a torch.Tensor")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"Model input {name!r} has shape {tuple(value.shape)}, expected {shape}"
            )
        if value.dtype not in supported_dtypes:
            raise ValueError(
                f"Model input {name!r} has unsupported dtype {value.dtype}"
            )
        if value.device.type != "cpu":
            raise ValueError(f"Model input {name!r} must be on CPU")
        normalized.append(value.detach().contiguous())
    return tuple(normalized)


def flatten_outputs(value: Any, names: tuple[str, ...]) -> tuple[Any, ...]:
    if isinstance(value, dict):
        try:
            outputs = tuple(value[name] for name in names)
        except KeyError as exc:
            raise ValueError(f"Model output is missing {exc.args[0]}") from exc
    elif isinstance(value, (tuple, list)):
        outputs = tuple(value)
    else:
        outputs = (value,)
    if len(outputs) != len(names):
        raise ValueError(f"Expected {len(names)} model outputs, got {len(outputs)}")
    return outputs


def evaluate_quality(
    module: ModuleType,
    predict: Any,
    dataset_root: Path,
    task: str,
) -> dict[str, float]:
    evaluator = quality_evaluator(module)
    metrics = evaluator(predict, dataset_root)
    return normalize_task_metrics(metrics, task, "evaluate_quality result")


def quality_evaluator(module: ModuleType) -> Any:
    evaluator = getattr(module, "evaluate_quality", None)
    if evaluator is None or not callable(evaluator):
        raise ValueError(
            "forward.py must define evaluate_quality(predict, dataset_root)"
        )
    signature = inspect.signature(evaluator)
    try:
        signature.bind(object(), Path("dataset"))
    except TypeError as exc:
        raise ValueError(
            "evaluate_quality must accept predict and dataset_root"
        ) from exc
    return evaluator
