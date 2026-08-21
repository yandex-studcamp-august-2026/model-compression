# kd_ref_forward_kl_a050_t4

**Автор:** misha
**Гипотеза:** reference will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + forward_kl, alpha=0.5, temperature=4.0.

## Результат
mIoU: 0.711220801 (baseline 0.717357457, -0.006136656)

## Статус гипотезы
Не подтвердилась: mIoU ниже CE-only baseline.

## Дальнейшие шаги
Не продолжать эту конфигурацию; проверить feature или boundary KD.
