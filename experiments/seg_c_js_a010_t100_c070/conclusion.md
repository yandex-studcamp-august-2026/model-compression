# c_js_a010_t100_c070

**Автор:** misha
**Гипотеза:** teacher confidence threshold=0.7 will improve mIoU relative to the CE-only SegFormer-B0 baseline.
**Метод:** CE + js, alpha=0.1, temperature=1.0. Используется confidence threshold 0.7.

## Результат
mIoU: 0.717412531 (baseline 0.717357457, +0.000055075)

## Статус гипотезы
Неоднозначно: результат практически совпадает с baseline.

## Дальнейшие шаги
Перейти от перебора KL к feature или boundary distillation.
