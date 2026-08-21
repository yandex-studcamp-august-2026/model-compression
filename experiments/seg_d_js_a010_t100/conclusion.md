# d_js_a010_t100

**Автор:** misha
**Гипотеза:** divergence=js will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + js, alpha=0.1, temperature=1.0.

## Результат
mIoU: 0.717403591 (baseline 0.717357457, +0.000046134)

## Статус гипотезы
Неоднозначно: результат практически совпадает с baseline.

## Дальнейшие шаги
Перейти от перебора KL к feature или boundary distillation.
