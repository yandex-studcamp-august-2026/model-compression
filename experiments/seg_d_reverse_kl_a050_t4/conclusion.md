# d_reverse_kl_a050_t4

**Автор:** misha
**Гипотеза:** reverse KL will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + reverse_kl, alpha=0.5, temperature=4.0.

## Результат
mIoU: 0.708821654 (baseline 0.717357457, -0.008535802)

## Статус гипотезы
Не подтвердилась: mIoU ниже CE-only baseline.

## Дальнейшие шаги
Не продолжать эту конфигурацию; проверить feature или boundary KD.
