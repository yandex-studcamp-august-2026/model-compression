# _baseline

**Автор:** misha
**Гипотеза:** CE-only training establishes the student baseline without KD.
**Метод:** SegFormer-B0 обучен только с ground-truth CrossEntropyLoss, без KD.

## Результат
mIoU: 0.717357457 (baseline 0.717357457, +0.000000000)

## Статус гипотезы
Baseline для всех экспериментов с дистилляцией.

## Дальнейшие шаги
Использовать как контрольную модель для качества и скорости.
