# a_forward_kl_a070_t4

**Автор:** misha
**Гипотеза:** alpha=0.7 will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + forward_kl, alpha=0.7, temperature=4.0.

## Результат
mIoU: 0.709819615 (baseline 0.717357457, -0.007537842)

## Статус гипотезы
Не подтвердилась: mIoU ниже CE-only baseline.

## Дальнейшие шаги
Не продолжать эту конфигурацию; проверить feature или boundary KD.
