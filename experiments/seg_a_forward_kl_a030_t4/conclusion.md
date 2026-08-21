# a_forward_kl_a030_t4

**Автор:** misha
**Гипотеза:** alpha=0.3 will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + forward_kl, alpha=0.3, temperature=4.0.

## Результат
mIoU: 0.713316381 (baseline 0.717357457, -0.004041076)

## Статус гипотезы
Не подтвердилась: mIoU ниже CE-only baseline.

## Дальнейшие шаги
Не продолжать эту конфигурацию; проверить feature или boundary KD.
