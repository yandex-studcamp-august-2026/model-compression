# t_forward_kl_a050_t8

**Автор:** misha
**Гипотеза:** temperature=8 will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + forward_kl, alpha=0.5, temperature=8.0.

## Результат
mIoU: 0.703621209 (baseline 0.717357457, -0.013736248)

## Статус гипотезы
Не подтвердилась: mIoU ниже CE-only baseline.

## Дальнейшие шаги
Не продолжать эту конфигурацию; проверить feature или boundary KD.
