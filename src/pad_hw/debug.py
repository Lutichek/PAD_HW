"""Скрипт для отладки линтера и тайпчекера."""

from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    """Среднее арифметическое последовательности чисел."""
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def normalize(values: Sequence[float]) -> list[float]:
    """Min-max нормализация значений в отрезок [0, 1]."""
    low, high = min(values), max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


def describe(name: str, values: Sequence[float]) -> str:
    """Короткая текстовая сводка по выборке."""
    return f"{name}: n={len(values)}, mean={mean(values):.3f}"


if __name__ == "__main__":
    sample = [1.0, 2.5, 4.0, 10.0]
    print(describe("sample", sample))
    print(normalize(sample))
