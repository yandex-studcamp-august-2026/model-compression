# d_reverse_kl_a010_t100

**Автор:** misha
**Гипотеза:** divergence=reverse_kl will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + reverse_kl, alpha=0.1, temperature=1.0.

## Результат
mIoU: 0.718774974 (baseline 0.717357457, +0.001417518)

## Статус гипотезы
Неоднозначно: есть небольшой прирост, но выполнен один запуск.

## Дальнейшие шаги
Проверить class-wise IoU на том же evaluation split.
