# Model benchmark pipeline

Репозиторий хранит эксперименты и сравнивает качество, latency и throughput после
конверсии PyTorch → ONNX Runtime / TensorRT.

## Эксперимент

```text
experiments/<name>/
├── forward.py          # модель, входы и evaluate_quality
├── weights.url         # s3://.../checkpoints/<name>/checkpoint.pt
├── weights.sha256
├── dataset.json        # validation subset в Object Storage
├── metrics.json        # исходные метрики обучения
├── conclusion.md
├── CPU                 # запустить ONNX Runtime
├── GPU                 # запустить TensorRT FP32 и дополнительный FP16
└── PRIMARY             # включить в автоматический benchmark pull request
```

`CPU`, `GPU` и `PRIMARY` — пустые маркерные файлы. `CPU` и `GPU` можно использовать
одновременно. Без `PRIMARY` эксперимент сохраняется в репозитории и запускается
вручную через `Actions → Model benchmark → Run workflow`.

`forward.py` должен экспортировать batch-1 модель, `INPUT_SHAPE` и
`evaluate_quality(predict, dataset_root)`. Checkpoint — чистый `state_dict`.
Шаблоны входных файлов находятся в [`templates/experiment`](templates/experiment/).

## Pipeline

Pull request в `main` для изменённых `PRIMARY`-экспериментов:

1. проверяет структуру и SHA-256 checkpoint;
2. считает исходное качество PyTorch на validation subset;
3. экспортирует ONNX и проверяет численную и task-level parity;
4. считает CPU benchmark для `CPU`;
5. собирает TensorRT FP32 и дополнительный FP16, повторно проверяет качество и
   считает GPU benchmark для `GPU`;
6. сохраняет `report.json`, `latency_ms.csv` и `latency_histogram.svg` на 7 дней.

FP32/FP16 performance публикуется только после успешной проверки качества.
Результат принимается при `conversion_status=validated`, `quality.passed=true` и
`status=completed`.

## Локальная проверка

Требуются Python 3.12 и `uv==0.11.32`.

```bash
make install-dev
make check
make validate EXPERIMENT=experiments/<name>
make export EXPERIMENT=experiments/<name> \
  DATASET=/path/to/dataset WEIGHTS=/path/to/checkpoint.pt
make benchmark-cpu BUNDLE=bundles/<name>
# На настроенной GPU VM:
make benchmark-gpu BUNDLE=bundles/<name> DATASET=/path/to/dataset
```
