# a_forward_kl_a010_t100

**Автор:** misha
**Гипотеза:** alpha=0.1 will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + forward_kl, alpha=0.1, temperature=1.0.

## Результат
mIoU: 0.717678368 (baseline 0.717357457, +0.000320911)

## Статус гипотезы
Неоднозначно: результат практически совпадает с baseline.

## Дальнейшие шаги
Перейти от перебора KL к feature или boundary distillation.
