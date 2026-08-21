# a_forward_kl_a090_t4

**Автор:** misha
**Гипотеза:** alpha=0.9 will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + forward_kl, alpha=0.9, temperature=4.0.

## Результат
mIoU: 0.709273815 (baseline 0.717357457, -0.008083642)

## Статус гипотезы
Не подтвердилась: mIoU ниже CE-only baseline.

## Дальнейшие шаги
Не продолжать эту конфигурацию; проверить feature или boundary KD.
