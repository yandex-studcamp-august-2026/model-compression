# s0_pretrained_student

**Автор:** misha
**Гипотеза:** The pretrained B0 provides the lower student baseline.
**Метод:** Готовый Cityscapes checkpoint без дополнительного обучения.

## Результат
mIoU: 0.702437580 (baseline 0.717357457, -0.014919877)

## Статус гипотезы
Не подтвердилась: mIoU ниже CE-only baseline.

## Дальнейшие шаги
Не продолжать эту конфигурацию; проверить feature или boundary KD.
