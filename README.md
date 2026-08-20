# Model benchmark pipeline

Репозиторий хранит ML-эксперименты и запускает воспроизводимые benchmark:

- `CPU` — ONNX Runtime на GitHub-hosted runner;
- `GPU` — TensorRT FP32 и дополнительный FP16 на NVIDIA V100 в Yandex Cloud.

## Эксперимент

Для запуска создайте директорию:

```text
experiments/<experiment_name>/
├── forward.py
├── weights.url
├── weights.sha256
├── dataset.json
├── metrics.json
├── conclusion.md
└── CPU
```

Имя эксперимента должно быть в `snake_case`. Вместо `CPU` можно использовать
`GPU`, но два маркера одновременно запрещены. Маркер — пустой файл.

`weights.url` содержит URI checkpoint:

```text
s3://<bucket>/checkpoints/<experiment_name>/checkpoint.pt
```

`weights.sha256` содержит lowercase SHA-256 checkpoint. Checkpoint должен быть
чистым `state_dict`.

`dataset.json` задаёт корень validation dataset в `datasets-studcamp` и только
нужные подпапки. CI скачивает эти объекты и дважды вызывает объявленную в
`forward.py` функцию `evaluate_quality(predict, dataset_root)`: для PyTorch и
для ONNX Runtime. Конверсия отклоняется, если task-метрики расходятся сильнее
`QUALITY_ATOL` (по умолчанию `1e-4`).

`forward.py` объявляет ровно один `torch.nn.Module`. Его конструктор принимает
`weights_path`, строго загружает checkpoint, а файл задаёт реальную batch-1 форму:

```python
INPUT_SHAPE = (1, 3, 512, 512)
```

Для нестандартных или нескольких входов задайте `make_inputs(seed)`,
`INPUT_NAMES` и `INPUT_SHAPES`. Готовые шаблоны `metrics.json` находятся в
[`templates/experiment`](templates/experiment/).

## Что запускает CI

Pull request в `main` с изменённым экспериментом выполняет:

1. проверку структуры и SHA-256 checkpoint;
2. экспорт PyTorch в ONNX;
3. сравнение выходов и task-метрик PyTorch и ONNX Runtime;
4. CPU- или GPU-benchmark в зависимости от маркера;
5. публикацию отчёта, CSV и SVG-гистограммы на 7 дней.

Начинайте с `CPU`. После успешного CPU PR создайте новую ветку от `main`, замените
`CPU` на `GPU` и откройте отдельный PR. GPU job сам включает остановленную VM,
собирает TensorRT FP32 и FP16 engines, выполняет проверки и останавливает VM.

Изменение notebook не запускает benchmark. Ручной запуск доступен через
`Actions → Model benchmark → Run workflow` с путём
`experiments/<experiment_name>`.

## Локальные команды

Требуются Python 3.12 и `uv==0.11.32`.

```bash
make install-dev
make check
make validate EXPERIMENT=experiments/<experiment_name>
```

Экспорт с локальным checkpoint:

```bash
make export \
  EXPERIMENT=experiments/<experiment_name> \
  DATASET=/path/to/downloaded/validation-dataset \
  WEIGHTS=/path/to/checkpoint.pt
```

CPU-benchmark готового bundle:

```bash
make benchmark \
  EXPERIMENT=experiments/<experiment_name> \
  BUNDLE=bundles/<experiment_name>
```

## Результат

Artifact содержит:

```text
report.json
latency_ms.csv
latency_histogram.svg
```

Принимать результат можно только при `conversion_status=validated`,
`quality.passed=true` и `status=completed`. В отчёте сохраняются качество до и
после конверсии, latency p50/p90/p95/p99, throughput и SVG-гистограмма latency.
