# Code Drills — Tier 0: Basics (Variables, Strings, Control Flow, Functions)

The bonus track: pure language mechanics, zero ML. Every entry is a question followed immediately by its code answer — cover the answer, write your own first, then compare. This file assumes nothing; it's the layer *underneath* `python-utilities-practice.md` and every ML doc on this hub, which all assume you already read/write Python fluently. Start here if any of that feels shaky.

Part of the **Code Drills** bonus tier — see also `code-drills-data-structures.md`, `code-drills-oop-intermediate.md`, `code-drills-numpy-pandas.md`, `code-drills-classical-ml.md`, `code-drills-deep-learning.md` for the climb from here to CNNs and LSTMs.

---

## Cluster 1 — Variables & Types

**1. Assign a variable and check its type at runtime.**
```python
x = 42
type(x)          # <class 'int'>
isinstance(x, int)   # True — prefer isinstance() over type()==... for checks, it respects subclasses
```

**2. Convert between int, float, and str.**
```python
int("42")        # 42
float("3.14")     # 3.14
str(42)           # "42"
int(3.9)          # 3 — truncates toward zero, does NOT round
```

**3. Swap two variables without a temp variable.**
```python
a, b = 1, 2
a, b = b, a       # a=2, b=1 — Python builds a tuple (b, a) then unpacks it in one step
```

**4. Unpack multiple values from one assignment.**
```python
a, b, c = 1, 2, 3
first, *rest = [1, 2, 3, 4]   # first=1, rest=[2, 3, 4] — star-unpacking soaks up the remainder
```

**5. Check if a variable is `None`.**
```python
x = None
x is None         # True — use `is`, not `==`, for None/True/False comparisons (identity, not equality)
```

**6. Integer division vs. float (true) division.**
```python
7 / 2         # 3.5   — always returns a float
7 // 2        # 3     — floor division, drops the remainder
7 % 2         # 1     — the remainder itself
```

**7. Use the modulo operator to check even/odd.**
```python
def is_even(n):
    return n % 2 == 0
```

**8. Format a value into a string with an f-string.**
```python
name, score = "Sam", 91.567
f"{name} scored {score:.1f}%"   # 'Sam scored 91.6%' — :.1f rounds to 1 decimal place
```

**9. Concatenate a string and a number safely.**
```python
age = 30
"Age: " + str(age)     # must convert first — "Age: " + age raises TypeError
f"Age: {age}"           # f-strings convert automatically — the preferred way
```

**10. Write a one-line conditional (ternary) expression.**
```python
n = 7
label = "even" if n % 2 == 0 else "odd"
```

**11. Chain comparisons the Pythonic way.**
```python
x = 5
0 < x < 10        # True — equivalent to (0 < x) and (x < 10), evaluated once
```

**12. Check the truthiness of common "empty" values.**
```python
bool(0), bool(""), bool([]), bool(None), bool({})   # all False
bool(1), bool("0"), bool([0])                         # all True — "0" and [0] are non-empty, so truthy
```

---

## Cluster 2 — Strings

**13. Split a string into a list of words (whitespace-delimited).**
```python
"the quick brown fox".split()     # ['the', 'quick', 'brown', 'fox']
```

**14. Split a string on a specific delimiter, then unpack directly into variables.**
```python
line = "Sam,91,Engineering"
name, score, dept = line.split(",")
```

**15. Join a list of strings back into one, with a separator.**
```python
",".join(["Sam", "91", "Engineering"])    # 'Sam,91,Engineering'
```

**16. Slice a string — first n chars, last n chars, reversed.**
```python
s = "hello world"
s[:5]     # 'hello'   — first 5
s[-5:]    # 'world'   — last 5
s[::-1]   # 'dlrow olleh' — step -1 walks backward
```

**17. Strip leading/trailing whitespace (and specific characters).**
```python
"  hello  ".strip()        # 'hello'
"--hello--".strip("-")     # 'hello' — strips only the chars given, from both ends
```

**18. Change case: upper, lower, title.**
```python
"Hello World".upper()    # 'HELLO WORLD'
"Hello World".lower()    # 'hello world'
"hello world".title()    # 'Hello World'
```

**19. Check if a substring exists inside a string.**
```python
"ell" in "hello"     # True — `in` works on strings just like it does on lists
```

**20. Replace a substring.**
```python
"2026-08-02".replace("-", "/")    # '2026/08/02'
```

**21. Find the index of a substring (and handle "not found").**
```python
"hello".find("l")       # 2  — index of first match
"hello".find("z")       # -1 — find() returns -1 instead of raising
"hello".index("z")      # raises ValueError — index() is the strict version
```

**22. Count occurrences of a substring.**
```python
"mississippi".count("ss")    # 2
```

**23. Check if a string is purely numeric or purely alphabetic.**
```python
"123".isdigit()      # True
"abc".isalpha()       # True
"abc123".isalnum()    # True — alphanumeric, no symbols/spaces
```

**24. Write a multiline string.**
```python
msg = """Line one
Line two"""
```

**25. Format numbers with width/padding and percentages.**
```python
f"{7:03d}"        # '007'   — zero-padded to width 3
f"{0.4567:.1%}"    # '45.7%' — treats the value as a fraction, multiplies by 100
```

**26. Reverse a string and check if it's a palindrome.**
```python
def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]
```

**27. Turn a comma-separated string of numbers into a list of ints.**
```python
"3, 1, 4, 1, 5".split(",")                    # ['3', ' 1', ' 4', ' 1', ' 5'] — still strings, with stray spaces
[int(x) for x in "3, 1, 4, 1, 5".split(",")]  # [3, 1, 4, 1, 5] — int() strips whitespace for free
```

**28. Remove specific characters from a string.**
```python
"h3ll0 w0rld".translate(str.maketrans("", "", "0123456789"))   # 'hll wrld'
```

**29. Check if a string starts/ends with a given pattern.**
```python
"report_2026.csv".startswith("report_")   # True
"report_2026.csv".endswith(".csv")         # True
```

**30. Pad a string to a fixed width (left/right/center align).**
```python
"7".zfill(3)          # '007'
"log".ljust(10, ".")  # 'log.......'
"log".rjust(10, ".")  # '.......log'
"log".center(10, ".") # '..log.....'
```

---

## Cluster 3 — Control Flow

**31. Write a basic if/elif/else chain.**
```python
def grade(score):
    if score >= 90: return "A"
    elif score >= 80: return "B"
    else: return "C"
```

**32. Loop over a range of numbers.**
```python
for i in range(5):        # 0, 1, 2, 3, 4 — stop is exclusive
    print(i)
for i in range(2, 10, 2):  # 2, 4, 6, 8 — start, stop, step
    print(i)
```

**33. Loop over a list while tracking the index.**
```python
fruits = ["apple", "banana", "cherry"]
for i, fruit in enumerate(fruits):
    print(i, fruit)
```

**34. Use `while` with `break` to exit early.**
```python
n = 0
while True:
    if n >= 3:
        break
    print(n)
    n += 1
```

**35. Use `continue` to skip an iteration.**
```python
for n in range(10):
    if n % 2 != 0:
        continue      # skip odd numbers
    print(n)
```

**36. Use the (rare) `for...else` clause.**
```python
for n in [1, 3, 5, 7]:
    if n % 2 == 0:
        break
else:
    print("no even number found")   # runs only if the loop never hit `break`
```

**37. Build a multiplication table with nested loops.**
```python
for i in range(1, 4):
    for j in range(1, 4):
        print(i * j, end=" ")
    print()
```

**38. Loop over two lists in parallel.**
```python
names = ["Sam", "Ana"]
scores = [91, 88]
for name, score in zip(names, scores):
    print(name, score)
```

**39. Loop with a sentinel value to signal "stop".**
```python
data = [4, 8, 15, -1, 16]
for x in data:
    if x == -1:
        break
    print(x)
```

**40. Use a guard clause instead of nesting.**
```python
def process(x):
    if x is None:            # guard: handle the bad case first, return early
        return "invalid"
    return x * 2              # main logic stays unindented, easier to read
```

**41. Use `match`/`case` (Python 3.10+) for multi-branch dispatch.**
```python
def describe(status):
    match status:
        case 200: return "OK"
        case 404: return "Not Found"
        case _:   return "Unknown"     # `_` is the wildcard/default case
```

**42. Combine multiple conditions with `any()`/`all()`.**
```python
nums = [2, 4, 6, 8]
all(n % 2 == 0 for n in nums)    # True  — every element even
any(n > 7 for n in nums)          # True  — at least one > 7
```

**43. Use the walrus operator `:=` to assign inside a condition.**
```python
data = [1, 2, 3, 4, 5]
i = 0
while (n := data[i]) < 4:    # assigns n AND tests it in one expression
    print(n)
    i += 1
```

**44. Preview: a list comprehension is a for-loop compressed to one line.**
```python
squares = []
for x in range(5):
    squares.append(x * x)

squares = [x * x for x in range(5)]    # identical result — full depth in code-drills-data-structures.md
```

**45. Filter items inside a loop vs. inside a comprehension.**
```python
nums = [1, -2, 3, -4, 5]
positives = [n for n in nums if n > 0]    # [1, 3, 5] — the `if` at the end filters
```

---

## Cluster 4 — Functions

**46. Define a function with positional arguments.**
```python
def add(a, b):
    return a + b
```

**47. Give a parameter a default value.**
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

greet("Sam")               # 'Hello, Sam!'
greet("Sam", "Hi")          # 'Hi, Sam!'
```

**48. Call a function using keyword arguments (order stops mattering).**
```python
greet(greeting="Hi", name="Sam")   # same result, arguments matched by name not position
```

**49. Accept a variable number of positional arguments with `*args`.**
```python
def total(*args):
    return sum(args)

total(1, 2, 3)    # 6 — args is the tuple (1, 2, 3) inside the function
```

**50. Accept a variable number of keyword arguments with `**kwargs`.**
```python
def describe(**kwargs):
    return kwargs

describe(name="Sam", age=30)   # {'name': 'Sam', 'age': 30} — kwargs is a dict
```

**51. Mix positional, default, `*args`, and `**kwargs` in one signature.**
```python
def build(a, b=10, *args, **kwargs):
    return a, b, args, kwargs

build(1, 2, 3, 4, x=5)   # (1, 2, (3, 4), {'x': 5}) — this is also the required ordering
```

**52. Return multiple values from a function.**
```python
def min_max(nums):
    return min(nums), max(nums)     # actually returns a tuple

lo, hi = min_max([3, 1, 4, 1, 5])   # unpacked immediately on the caller's side
```

**53. Write a docstring the standard way.**
```python
def add(a, b):
    """Return the sum of a and b."""
    return a + b

add.__doc__    # 'Return the sum of a and b.'
```

**54. Write a one-line anonymous function with `lambda`.**
```python
square = lambda x: x * x
square(5)    # 25 — equivalent to a def, but expression-only, no statements allowed inside
```

**55. Pass a function as an argument to another function.**
```python
def apply_twice(f, x):
    return f(f(x))

apply_twice(lambda x: x + 3, 10)    # 16 — functions are first-class values, just like ints or strings
```

**56. Use `map()` to apply a function across an iterable.**
```python
list(map(lambda x: x * 2, [1, 2, 3]))   # [2, 4, 6] — map() is lazy, wrap in list() to see results
```

**57. Use `filter()` to keep only items matching a condition.**
```python
list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4]))   # [2, 4]
```

**58. Sort a list of dicts by a specific field using `key=`.**
```python
people = [{"name": "Sam", "age": 30}, {"name": "Ana", "age": 25}]
sorted(people, key=lambda p: p["age"])     # sorted by age ascending
sorted(people, key=lambda p: p["age"], reverse=True)   # descending
```

**59. Write a recursive function.**
```python
def factorial(n):
    if n <= 1:          # base case — without this, infinite recursion
        return 1
    return n * factorial(n - 1)
```

**60. Spot the classic mutable-default-argument bug.**
```python
def add_item(item, bucket=[]):   # BUG: the default list is created ONCE, at def-time, and reused every call
    bucket.append(item)
    return bucket

add_item("a")   # ['a']
add_item("b")   # ['a', 'b'] — surprise! same list carried over from the first call

def add_item_fixed(item, bucket=None):
    if bucket is None:
        bucket = []      # fresh list every call — the standard fix
    bucket.append(item)
    return bucket
```

**61. Distinguish local scope from global scope.**
```python
x = 10
def show():
    x = 20        # this creates a NEW local x — does not touch the global one
    print(x)      # 20

show()
print(x)          # still 10
```

**62. Modify a global variable from inside a function with `global`.**
```python
counter = 0
def increment():
    global counter     # without this line, `counter += 1` below would raise UnboundLocalError
    counter += 1
```

**63. Write a closure — a function that remembers a variable from its enclosing scope.**
```python
def make_multiplier(factor):
    def multiply(x):
        return x * factor    # `factor` is remembered even after make_multiplier() has returned
    return multiply

times3 = make_multiplier(3)
times3(10)    # 30
```

**64. Add type hints to a function signature.**
```python
def add(a: int, b: int) -> int:
    return a + b
# hints are documentation + tooling support only — Python does NOT enforce them at runtime
```

**65. Reduce a list to a single value with `functools.reduce`.**
```python
from functools import reduce
reduce(lambda acc, x: acc + x, [1, 2, 3, 4], 0)   # 10 — same as sum(), shown for the general pattern
```

**66. Unpack a list/dict directly into a function call.**
```python
def add3(a, b, c):
    return a + b + c

args = [1, 2, 3]
add3(*args)             # unpacks the list as three positional args

kwargs = {"a": 1, "b": 2, "c": 3}
add3(**kwargs)          # unpacks the dict as keyword args, matched by key name
```

---

**Next in the Code Drills tier:** `code-drills-data-structures.md` (lists/dicts/sets/comprehensions, JSON, files, exceptions).
