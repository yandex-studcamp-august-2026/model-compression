# t_forward_kl_a050_t1

**Автор:** misha
**Гипотеза:** temperature=1 will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + forward_kl, alpha=0.5, temperature=1.0.

## Результат
mIoU: 0.719673693 (baseline 0.717357457, +0.002316236)

## Статус гипотезы
Неоднозначно: есть небольшой прирост, но выполнен один запуск.

## Дальнейшие шаги
Проверить class-wise IoU на том же evaluation split.
