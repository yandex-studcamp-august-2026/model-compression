# d_js_a050_t4

**Автор:** misha
**Гипотеза:** Jensen-Shannon will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + js, alpha=0.5, temperature=4.0.

## Результат
mIoU: 0.714840233 (baseline 0.717357457, -0.002517223)

## Статус гипотезы
Не подтвердилась: mIoU ниже CE-only baseline.

## Дальнейшие шаги
Не продолжать эту конфигурацию; проверить feature или boundary KD.
