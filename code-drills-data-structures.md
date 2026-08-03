# Code Drills — Tier 0: Data Structures, JSON, Files, Exceptions

Continues `code-drills-basics.md`. Same format: question, then code answer — cover and self-test before reading. This is the cluster that directly answers "how do I convert JSON to a dict" and everything adjacent to it.

---

## Cluster 1 — Lists & Tuples

**1. Create a list, index it, and slice it.**
```python
nums = [10, 20, 30, 40, 50]
nums[0]      # 10   — first element
nums[-1]     # 50   — last element
nums[1:3]    # [20, 30] — slice, stop is exclusive
```

**2. Add and remove items from a list.**
```python
nums = [1, 2, 3]
nums.append(4)        # [1, 2, 3, 4]        — add to the end
nums.insert(0, 0)     # [0, 1, 2, 3, 4]      — insert at a specific index
nums.remove(2)         # [0, 1, 3, 4]         — removes the first VALUE that matches, not an index
nums.pop()             # returns 4, list is now [0, 1, 3] — pop() removes+returns the last item
nums.pop(0)            # returns 0, list is now [1, 3]     — pop(i) removes+returns at index i
```

**3. Know when to use a tuple instead of a list.**
```python
point = (3, 4)        # tuples are immutable — use them for fixed, "this shouldn't change" data
point[0] = 5           # raises TypeError — lists allow this, tuples don't
# rule of thumb: coordinates, DB rows, function returns -> tuple. A growing collection -> list.
```

**4. Copy a list correctly — avoid the shared-reference trap.**
```python
original = [1, 2, 3]
alias = original            # NOT a copy — alias and original point to the SAME list
alias.append(4)
original                    # [1, 2, 3, 4] — surprise! original changed too

real_copy = original.copy()  # or original[:] or list(original)
real_copy.append(5)
original                     # unaffected — real_copy is an independent list
```

**5. Sort a list — in place vs. producing a new list.**
```python
nums = [3, 1, 4, 1, 5]
nums.sort()                    # sorts IN PLACE, returns None
sorted_copy = sorted([3, 1, 4])  # returns a NEW sorted list, original untouched
nums.sort(reverse=True)         # descending
```

**6. Reverse a list.**
```python
nums = [1, 2, 3]
nums.reverse()        # in place -> [3, 2, 1]
nums[::-1]             # or via slicing -> new reversed list, original untouched
```

**7. Check membership and find an index.**
```python
nums = [10, 20, 30]
20 in nums            # True
nums.index(20)        # 1 — raises ValueError if not found (unlike string .find())
```

**8. Flatten a nested (2D) list into a flat one.**
```python
matrix = [[1, 2], [3, 4], [5, 6]]
flat = [x for row in matrix for x in row]    # [1, 2, 3, 4, 5, 6] — outer loop first, then inner
```

**9. Remove duplicates from a list while preserving order.**
```python
nums = [3, 1, 4, 1, 5, 3]
list(dict.fromkeys(nums))    # [3, 1, 4, 5] — dicts remember insertion order (Python 3.7+); set() would lose order
```

**10. Combine two lists — concatenate vs. pair up element-wise.**
```python
a, b = [1, 2, 3], [4, 5, 6]
a + b                # [1, 2, 3, 4, 5, 6] — concatenation
list(zip(a, b))       # [(1, 4), (2, 5), (3, 6)] — pairs elements by position
```

**11. Unpack part of a list with `*`.**
```python
nums = [1, 2, 3, 4, 5]
first, second, *rest = nums     # first=1, second=2, rest=[3, 4, 5]
*init, last = nums              # init=[1, 2, 3, 4], last=5
```

**12. "Unzip" a list of tuples into two separate lists.**
```python
pairs = [(1, "a"), (2, "b"), (3, "c")]
nums, letters = zip(*pairs)      # nums=(1, 2, 3), letters=('a', 'b', 'c') — zip(*...) is its own inverse
```

---

## Cluster 2 — Dicts & Sets

**13. Create a dict and access, add, and update keys.**
```python
person = {"name": "Sam", "age": 30}
person["name"]          # 'Sam'
person["city"] = "Austin"   # adds a new key
person["age"] = 31           # updates an existing key
```

**14. Access a key safely, with a default if it's missing.**
```python
person = {"name": "Sam"}
person["age"]           # raises KeyError
person.get("age")        # None — no error
person.get("age", 0)     # 0    — explicit default
```

**15. Iterate over a dict's keys, values, and items.**
```python
d = {"a": 1, "b": 2}
for k in d:                    # iterates keys by default
    print(k)
for v in d.values():
    print(v)
for k, v in d.items():         # the usual pattern — both at once
    print(k, v)
```

**16. Merge two dicts.**
```python
defaults = {"color": "blue", "size": "M"}
overrides = {"size": "L"}
merged = defaults | overrides     # {'color': 'blue', 'size': 'L'} — Python 3.9+, right side wins on conflict
merged2 = {**defaults, **overrides}   # same result, works on older Python too
```

**17. Check whether a key exists in a dict.**
```python
"age" in person          # True/False — checks KEYS, not values, by default
```

**18. Delete a key — two ways, one of them safe.**
```python
d = {"a": 1, "b": 2}
del d["a"]                # raises KeyError if "a" isn't there
d.pop("b", None)           # returns the value (or None if missing) — no exception either way
```

**19. Invert a dict — swap its keys and values.**
```python
d = {"a": 1, "b": 2}
{v: k for k, v in d.items()}   # {1: 'a', 2: 'b'} — only safe if original values are unique
```

**20. Count item frequency using a plain dict.**
```python
words = ["a", "b", "a", "c", "b", "a"]
counts = {}
for w in words:
    counts[w] = counts.get(w, 0) + 1   # {'a': 3, 'b': 2, 'c': 1}
```

**21. Count item frequency the fast way with `Counter`.**
```python
from collections import Counter
counts = Counter(["a", "b", "a", "c", "b", "a"])
counts               # Counter({'a': 3, 'b': 2, 'c': 1})
counts.most_common(2)  # [('a', 3), ('b', 2)] — top 2, most frequent first
```

**22. Create a set and use it to deduplicate a list.**
```python
nums = [1, 2, 2, 3, 3, 3]
unique = set(nums)      # {1, 2, 3} — unordered, no duplicates, O(1) membership check
```

**23. Combine sets — union, intersection, difference.**
```python
a = {1, 2, 3}
b = {2, 3, 4}
a | b     # {1, 2, 3, 4}  — union
a & b     # {2, 3}         — intersection
a - b     # {1}            — in a but not b
a ^ b     # {1, 4}         — symmetric difference (in exactly one of the two)
```

**24. Group items by a key using `defaultdict`.**
```python
from collections import defaultdict
words = ["apple", "avocado", "banana", "blueberry", "cherry"]
groups = defaultdict(list)
for w in words:
    groups[w[0]].append(w)     # no need to check "is this key here yet" — defaultdict handles it
# groups = {'a': ['apple', 'avocado'], 'b': ['banana', 'blueberry'], 'c': ['cherry']}
```

---

## Cluster 3 — Comprehensions

**25. Write a basic list comprehension.**
```python
squares = [x * x for x in range(6)]    # [0, 1, 4, 9, 16, 25]
```

**26. Add a filter condition to a list comprehension.**
```python
evens = [x for x in range(10) if x % 2 == 0]    # [0, 2, 4, 6, 8]
```

**27. Write a dict comprehension.**
```python
words = ["apple", "kiwi", "banana"]
lengths = {w: len(w) for w in words}    # {'apple': 5, 'kiwi': 4, 'banana': 6}
```

**28. Write a set comprehension.**
```python
{x % 3 for x in range(10)}    # {0, 1, 2} — duplicates collapse automatically, like any set
```

**29. Nest comprehensions to flatten a 2D structure.**
```python
matrix = [[1, 2, 3], [4, 5, 6]]
[x for row in matrix for x in row if x % 2 == 0]   # [2, 4, 6] — loop order matches nested for-loops
```

**30. Use a generator expression instead of a list comprehension when you don't need the whole list in memory.**
```python
total = sum(x * x for x in range(1_000_000))   # no square brackets -> lazy, computes one value at a time
# a list comprehension here would build a 1,000,000-element list just to sum it and throw it away
```

---

## Cluster 4 — JSON ⇄ Dict

**31. Parse a JSON string into a Python dict.**
```python
import json
raw = '{"name": "Sam", "age": 30, "active": true}'
data = json.loads(raw)     # {'name': 'Sam', 'age': 30, 'active': True} — JSON true/null -> Python True/None
data["name"]                # 'Sam'
```

**32. Convert a Python dict into a JSON string.**
```python
person = {"name": "Sam", "age": 30, "active": True}
json.dumps(person)     # '{"name": "Sam", "age": 30, "active": true}' — Python True -> JSON true
```

**33. Pretty-print JSON for readability.**
```python
json.dumps(person, indent=2)
# {
#   "name": "Sam",
#   "age": 30,
#   "active": true
# }
```

**34. Read a JSON file straight into a dict.**
```python
with open("config.json") as f:
    config = json.load(f)     # note: json.load (file object) vs json.loads (string) — easy to mix up
```

**35. Write a dict out to a JSON file.**
```python
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)    # json.dump (file) vs json.dumps (string) — same naming pattern
```

**36. Access a nested value inside parsed JSON safely.**
```python
data = json.loads('{"user": {"name": "Sam", "roles": ["admin", "editor"]}}')
data["user"]["name"]          # 'Sam'
data["user"]["roles"][0]      # 'admin'
data.get("user", {}).get("email", "unknown")   # 'unknown' — chained .get() avoids a KeyError crash on missing keys
```

**37. Parse a JSON array of objects into a list of dicts.**
```python
raw = '[{"id": 1, "name": "Sam"}, {"id": 2, "name": "Ana"}]'
records = json.loads(raw)
names = [r["name"] for r in records]    # ['Sam', 'Ana'] — comprehension straight over parsed JSON
```

**38. Serialize a type JSON doesn't natively support (e.g. `datetime`).**
```python
from datetime import datetime
event = {"name": "deploy", "when": datetime.now()}
json.dumps(event, default=str)    # default=str: fall back to str() for anything json.dumps doesn't know
```

---

## Cluster 5 — Files & Paths

**39. Read an entire file into a string.**
```python
with open("notes.txt") as f:     # `with` auto-closes the file even if an error happens inside the block
    content = f.read()
```

**40. Read a file line by line.**
```python
with open("notes.txt") as f:
    for line in f:                 # memory-efficient — reads one line at a time, not the whole file
        print(line.strip())         # .strip() removes the trailing '\n'
```

**41. Write a string to a file (overwrite).**
```python
with open("out.txt", "w") as f:    # "w" truncates the file first if it already exists
    f.write("hello\n")
```

**42. Append to an existing file.**
```python
with open("log.txt", "a") as f:    # "a" adds to the end, never truncates
    f.write("new entry\n")
```

**43. Read a CSV file into a list of dicts.**
```python
import csv
with open("people.csv") as f:
    reader = csv.DictReader(f)      # uses the first row as column headers automatically
    rows = [row for row in reader]  # each row is an OrderedDict-like mapping of header -> value
```

**44. Check whether a file exists, using `pathlib`.**
```python
from pathlib import Path
Path("config.json").exists()    # True/False — preferred over os.path in modern Python
```

**45. Build file paths safely with `pathlib` (not string concatenation).**
```python
from pathlib import Path
base = Path("D:/nvidia")
full = base / "config.json"     # `/` is overloaded to join paths — works cross-platform (no manual "\\" vs "/")
```

---

## Cluster 6 — Exceptions

**46. Catch an error with try/except.**
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    result = None
```

**47. Catch a specific exception and read its message.**
```python
try:
    int("not a number")
except ValueError as e:
    print(f"conversion failed: {e}")    # `as e` binds the exception object so you can inspect it
```

**48. Catch multiple exception types in one block.**
```python
try:
    risky_call()
except (ValueError, TypeError) as e:
    print(f"bad input: {e}")
```

**49. Use `finally` to guarantee cleanup runs either way.**
```python
f = open("data.txt")
try:
    process(f)
finally:
    f.close()      # runs whether process() succeeds, raises, or even returns early — always executes
```

**50. Use `else` on a try block — code that runs only if NO exception occurred.**
```python
try:
    value = int(user_input)
except ValueError:
    print("invalid input")
else:
    print(f"got a valid number: {value}")    # only runs if the try block succeeded cleanly
```

**51. Raise your own exception with a message.**
```python
def set_age(age):
    if age < 0:
        raise ValueError("age cannot be negative")
    return age
```

**52. Define a custom exception class.**
```python
class InsufficientFundsError(Exception):
    pass

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFundsError(f"tried to withdraw {amount}, only {balance} available")
    return balance - amount
```

**53. Re-raise an exception after logging it, preserving the original traceback.**
```python
try:
    risky_call()
except ValueError:
    print("logging the error here")
    raise             # bare `raise` re-raises the SAME exception, traceback intact
```

**54. Chain a new exception onto the original one for context.**
```python
try:
    parse_config()
except ValueError as e:
    raise RuntimeError("config parsing failed") from e   # keeps both tracebacks linked, not just the new one
```

**55. Validate input up front and fail loudly instead of silently continuing.**
```python
def divide(a, b):
    if b == 0:
        raise ValueError("b cannot be zero")   # fail fast and clearly, rather than returning None/NaN silently
    return a / b
```

---

**Next in the Code Drills tier:** `code-drills-oop-intermediate.md` (classes, decorators, generators, closures).
