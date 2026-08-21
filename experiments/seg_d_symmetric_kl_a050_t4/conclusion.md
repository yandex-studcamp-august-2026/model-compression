# d_symmetric_kl_a050_t4

**Автор:** misha
**Гипотеза:** symmetric KL will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + symmetric_kl, alpha=0.5, temperature=4.0.

## Результат
mIoU: 0.710496902 (baseline 0.717357457, -0.006860554)

## Статус гипотезы
Не подтвердилась: mIoU ниже CE-only baseline.

## Дальнейшие шаги
Не продолжать эту конфигурацию; проверить feature или boundary KD.
