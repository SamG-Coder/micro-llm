# Held-out prompts — local coding assistant

Five prompts. These must never run in the trace hour.

1. Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping inclusive intervals and returns them sorted. Include 3 doctest examples. No explanation after the code.

2. This function is wrong. Fix it and keep the name.

```python
def binary_search(xs, target):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < target:
            lo = mid
        else:
            hi = mid
    return lo
```

It infinite-loops when the target is missing. Return the insertion index.

3. Write a pytest module for a function `slugify(text: str) -> str` that lowercases, replaces runs of non-alnum with a single hyphen, and strips leading/trailing hyphens. Do not implement slugify. Tests only.

4. Refactor this to a single function with the same behavior, no globals.

```python
BUF = []
def add(x):
    BUF.append(x)
def total():
    return sum(BUF)
def reset():
    BUF.clear()
```

5. In C++20, can `std::string_view` bind to a temporary `std::string` returned from a function? Write a 10-line program that demonstrates the lifetime rule, and say yes or no in one sentence before the code.
