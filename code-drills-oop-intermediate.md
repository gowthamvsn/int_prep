# Code Drills — Tier 1: OOP, Decorators, Generators, Context Managers

Continues `code-drills-data-structures.md`. This is the cluster that closes the gap to reading real library code — every `nn.Module` subclass, every `Dataset`, every `with torch.no_grad()` is built from exactly the four patterns in this file: classes, decorators, generators, context managers. Drill #15 draws the direct line to PyTorch.

---

## Cluster 1 — Classes & OOP

> 🔗 **Theory:** [PyTorch Deep Dive — Autograd Internals](/topic/practice-pytorch-deep#cluster-1-autograd-internals-custom-functions-and-hooks) (drill #15 below is the direct bridge from a plain class to `nn.Module`)

**1. Define a class with a constructor and instance attributes.**
```python
class Dog:
    def __init__(self, name, age):    # __init__ runs automatically when Dog(...) is called
        self.name = name               # self.X = ... attaches an attribute to THIS instance
        self.age = age

rex = Dog("Rex", 3)
rex.name    # 'Rex'
```

**2. Write an instance method that uses `self`.**
```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        return f"{self.name} says woof!"   # self gives the method access to THIS instance's data

rex = Dog("Rex", 3)
rex.bark()    # 'Rex says woof!' — Python passes `rex` in automatically as `self`
```

**3. Distinguish a class attribute (shared) from an instance attribute (per-object).**
```python
class Dog:
    species = "Canis familiaris"    # class attribute — ONE copy, shared by every instance

    def __init__(self, name):
        self.name = name             # instance attribute — a separate copy per object

a, b = Dog("Rex"), Dog("Fido")
a.species, b.species     # both 'Canis familiaris' — same shared value
a.name, b.name             # 'Rex', 'Fido' — independent per instance
```

**4. Write a `@classmethod` as an alternate constructor.**
```python
class Dog:
    def __init__(self, name, age):
        self.name, self.age = name, age

    @classmethod
    def from_birth_year(cls, name, birth_year, current_year=2026):
        return cls(name, current_year - birth_year)   # cls is the class itself, not an instance

rex = Dog.from_birth_year("Rex", 2023)   # age=3, without the caller doing the subtraction themselves
```

**5. Write a `@staticmethod` — a utility function that lives on the class but needs no `self`.**
```python
class Dog:
    @staticmethod
    def is_valid_name(name):
        return isinstance(name, str) and len(name) > 0

Dog.is_valid_name("Rex")    # True — called on the class directly, no instance needed
```

**6. Control how an object prints, with `__repr__` and `__str__`.**
```python
class Dog:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Dog(name={self.name!r})"    # unambiguous, for developers/debugging (repr(obj), or in a list)

    def __str__(self):
        return f"a dog named {self.name}"     # readable, for end users (print(obj), str(obj))

print(Dog("Rex"))          # 'a dog named Rex'      — uses __str__
[Dog("Rex")]                # [Dog(name='Rex')]      — uses __repr__ inside containers
```

**7. Inherit from a base class and extend its constructor.**
```python
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)    # runs Animal's __init__ first, so self.name gets set there
        self.breed = breed
```

**8. Override a parent method in a subclass.**
```python
class Animal:
    def speak(self):
        return "..."

class Dog(Animal):
    def speak(self):              # same method name — replaces the parent's version for Dog instances
        return "Woof!"

Dog().speak()    # 'Woof!'
```

**9. Follow the "protected"/"private" naming convention.**
```python
class Account:
    def __init__(self, balance):
        self._balance = balance     # single underscore: "internal, but still accessible" — a convention, not enforced
        self.__secret = "shh"        # double underscore: name-mangled to _Account__secret, harder to collide/access by accident

a = Account(100)
a._balance        # 100 — works fine, just a signal "don't touch this from outside"
a.__secret         # raises AttributeError — must use a._Account__secret
```

**10. Use `@property` to expose a computed value like a plain attribute.**
```python
class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return 3.14159 * self.radius ** 2   # looks like an attribute, computed on access, no stale cache

c = Circle(2)
c.area    # 12.56636 — no parentheses; called like an attribute, not a method
```

**11. Overload operators so custom objects work with `+`, `==`, `len()`.**
```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)   # defines what `v1 + v2` means

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y       # defines what `v1 == v2` means

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

Vector(1, 2) + Vector(3, 4)     # Vector(4, 6)
Vector(1, 2) == Vector(1, 2)    # True — without __eq__, this compares object IDENTITY and would be False
```

**12. Define an abstract base class that forces subclasses to implement a method.**
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        ...

class Circle(Shape):
    def __init__(self, r): self.r = r
    def area(self): return 3.14159 * self.r ** 2

Shape()          # raises TypeError — can't instantiate an ABC directly
Circle(2).area()  # 12.56636 — Circle is fine, it implemented the required method
```

**13. Use `@dataclass` to skip writing boilerplate `__init__`/`__repr__`/`__eq__` by hand.**
```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

p1, p2 = Point(1, 2), Point(1, 2)
p1                # Point(x=1, y=2) — free __repr__
p1 == p2           # True — free field-by-field __eq__, unlike drill #11's manual version
```

**14. Make an object callable — `obj(x)` instead of `obj.some_method(x)` — with `__call__`.**
```python
class Doubler:
    def __call__(self, x):
        return x * 2

double = Doubler()
double(5)    # 10 — `double(5)` is secretly `double.__call__(5)`
```

**15. See the exact shape `nn.Module` uses — this drill is pure OOP, no `torch` needed, to isolate the pattern.**
```python
class Layer:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x):              # every PyTorch layer defines forward() — the actual computation
        return x * self.weight

    def __call__(self, x):             # __call__ is what makes `layer(x)` work instead of `layer.forward(x)`
        return self.forward(x)          # PyTorch's __call__ also runs hooks, so ALWAYS call layer(x), never layer.forward(x) directly

layer = Layer(weight=2)
layer(5)    # 10 — looks like a function call, is actually __call__ -> forward() under the hood
# this IS the mechanism: nn.Module.__call__ wraps your forward() the same way drill #14 wraps a plain function
```

---

## Cluster 2 — Decorators

**16. Write your own decorator — a function that wraps another function.**
```python
import time

def timer(func):
    def wrapper(*args, **kwargs):     # accepts anything, so it works on ANY decorated function
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.4f}s")
        return result
    return wrapper

@timer                # sugar for: slow_add = timer(slow_add)
def slow_add(a, b):
    return a + b

slow_add(2, 3)    # prints timing, then returns 5
```

**17. Write a decorator that itself takes arguments (a decorator factory).**
```python
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n - 1):
                func(*args, **kwargs)
            return func(*args, **kwargs)   # return the LAST call's result
        return wrapper
    return decorator

@repeat(3)              # repeat(3) returns `decorator`, which then wraps greet
def greet():
    print("hi")

greet()    # prints "hi" three times
```

**18. Preserve a decorated function's name/docstring with `functools.wraps`.**
```python
from functools import wraps

def timer(func):
    @wraps(func)          # without this, slow_add.__name__ would become 'wrapper' after decorating
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@timer
def slow_add(a, b):
    """Add two numbers."""
    return a + b

slow_add.__name__    # 'slow_add' — preserved, thanks to @wraps
```

**19. Use `functools.lru_cache` to memoize an expensive/recursive function for free.**
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n < 2: return n
    return fib(n - 1) + fib(n - 2)     # naive recursion recomputes the same values repeatedly without caching

fib(30)    # fast — repeated calls with the same n are served straight from the cache, no recomputation
```

---

## Cluster 3 — Generators & Iterators

**20. Write a generator function with `yield`.**
```python
def countdown(n):
    while n > 0:
        yield n           # pauses here, hands back n, resumes from this exact point on the next call
        n -= 1

for i in countdown(3):
    print(i)      # 3, 2, 1
```

**21. Understand why a generator saves memory vs. a list.**
```python
def squares_list(n):
    return [x * x for x in range(n)]     # builds the ENTIRE list in memory immediately

def squares_gen(n):
    for x in range(n):
        yield x * x                       # computes ONE value at a time, only when asked

sum(squares_gen(10_000_000))    # never holds 10M numbers in memory at once — the list version would
```

**22. Pull values from a generator manually with `next()`.**
```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

gen = countdown(2)
next(gen)    # 2
next(gen)    # 1
next(gen)    # raises StopIteration — a `for` loop catches this automatically, manual next() doesn't
```

**23. Write a custom iterator class with `__iter__`/`__next__` (what `for` calls under the hood).**
```python
class Countdown:
    def __init__(self, start):
        self.n = start

    def __iter__(self):           # a `for` loop calls this ONCE, to get an iterator
        return self

    def __next__(self):           # then calls this repeatedly, until StopIteration
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n + 1

for i in Countdown(3):
    print(i)      # 3, 2, 1 — same output as drill #20's generator, written the manual way
```

**24. Delegate to another generator with `yield from`.**
```python
def inner():
    yield 1
    yield 2

def outer():
    yield from inner()   # equivalent to: for x in inner(): yield x
    yield 3

list(outer())    # [1, 2, 3]
```

---

## Cluster 4 — Context Managers

**25. Write a custom context manager as a class (`__enter__`/`__exit__`).**
```python
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self                  # whatever __enter__ returns is bound to `as x`

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        print(f"elapsed: {time.time() - self.start:.4f}s")
        return False                  # False (or None) means "don't suppress exceptions" — let them propagate

with Timer() as t:
    sum(range(1_000_000))
# __exit__ runs automatically here, even if the block raised an exception
```

**26. Write the same context manager the short way, with `@contextlib.contextmanager`.**
```python
from contextlib import contextmanager
import time

@contextmanager
def timer():
    start = time.time()
    yield                              # everything before yield = __enter__; everything after = __exit__
    print(f"elapsed: {time.time() - start:.4f}s")

with timer():
    sum(range(1_000_000))
# this is exactly what `with torch.no_grad():` is built from — a generator-based context manager
```

---

## Cluster 5 — Misc Structural Patterns

**27. Compose objects ("has-a") instead of always reaching for inheritance ("is-a").**
```python
class Engine:
    def start(self):
        return "vroom"

class Car:
    def __init__(self):
        self.engine = Engine()      # Car HAS an Engine — composition, not "Car inherits from Engine"

    def start(self):
        return self.engine.start()   # delegate to the composed object
```

**28. Check types at runtime with `isinstance()`/`issubclass()` — the right way to check "is this a Dog."**
```python
class Animal: pass
class Dog(Animal): pass

isinstance(Dog(), Animal)     # True  — also True for the exact class, and any subclass
type(Dog()) == Animal          # False — too strict, breaks for subclasses; avoid this pattern
issubclass(Dog, Animal)        # True  — class-to-class check, no instance needed
```

**29. Pass extra constructor args up through `super().__init__(*args, **kwargs)` without naming each one.**
```python
class Base:
    def __init__(self, a, b):
        self.a, self.b = a, b

class Derived(Base):
    def __init__(self, c, *args, **kwargs):
        super().__init__(*args, **kwargs)   # forwards whatever it received onward, no need to re-list a, b
        self.c = c

Derived(c=3, a=1, b=2)
```

**30. Guard "script vs. imported module" behavior with `if __name__ == "__main__"`.**
```python
def main():
    print("running as a script")

if __name__ == "__main__":    # True only when this file is RUN directly, False when it's imported elsewhere
    main()
```

**31. Use `itertools` for common generator building blocks instead of hand-rolling loops.**
```python
from itertools import count, cycle, islice

counter = count(start=10, step=2)          # infinite: 10, 12, 14, 16, ...
list(islice(counter, 4))                    # [10, 12, 14, 16] — islice caps an infinite generator to N items

colors = cycle(["red", "green", "blue"])    # infinite repeat: red, green, blue, red, green, blue, ...
list(islice(colors, 5))                     # ['red', 'green', 'blue', 'red', 'green']
```

---

**Next in the Code Drills tier:** `code-drills-numpy-pandas.md` (array/DataFrame drills — terser companion to the full `numpy-practice.md`/`pandas-practice.md` deep dives).
