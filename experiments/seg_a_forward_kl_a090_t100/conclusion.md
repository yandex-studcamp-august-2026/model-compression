# a_forward_kl_a090_t100

**Автор:** misha
**Гипотеза:** alpha=0.9 will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + forward_kl, alpha=0.9, temperature=1.0.

## Результат
mIoU: 0.719806075 (baseline 0.717357457, +0.002448618)

## Статус гипотезы
Неоднозначно: есть небольшой прирост, но выполнен один запуск.

## Дальнейшие шаги
Проверить class-wise IoU на том же evaluation split.
