# Python Utilities Practice — Built as a Chain, Not a List

These are the "surrounding" skills that make the modeling code elsewhere on this hub actually deployable. Each cluster is one continuous thread — every question builds on the answer before it, and closes with a worked summary example.

---

> 🔗 **Hands-on reps:** [Code Drills 1 — Basics](/topic/code-drills-basics) and [Code Drills 2 — Data Structures, JSON, Files, Exceptions](/topic/code-drills-data-structures)

## Cluster 1 — Datetime

### 1. How do you get the current time and format it as a readable string?
```python
from datetime import datetime
now = datetime.now()
now.strftime("%Y-%m-%d %H:%M:%S")     # -> e.g. '2026-07-18 14:32:07'
```
`%Y-%m-%d` is the ISO-8601 date format. It sorts correctly as plain text — lexicographic order matches chronological order — unlike `%m/%d/%Y`. That's exactly why it's the standard for filenames, log timestamps, and database columns.

### 2. How do you go the other direction — a formatted string into a real datetime object?
```python
dt = datetime.strptime("2026-07-18", "%Y-%m-%d")
```
**Mnemonic:** the letter right after "str" tells you the direction. **p**arse = **strp**time, string to datetime. **f**ormat = **strf**time, datetime to string.

### 3. How do you compute the difference between two real datetime objects?
```python
from datetime import timedelta
delta = datetime(2026, 8, 1) - datetime(2026, 7, 18)
delta.days          # 14 -- an integer number of days
delta.total_seconds()   # if you need finer-than-day resolution
```

### 4. A `timedelta` is what you get from subtracting two dates. How do you use it to add or subtract time safely?
```python
future = datetime.now() + timedelta(days=30, hours=6)
```
Manually incrementing a `.day` attribute breaks at month and year boundaries — day 31 of a 30-day month doesn't exist. `timedelta` handles all of that calendar arithmetic correctly, leap years included, in a way manual field arithmetic just can't.

### 5. Does a datetime object always know which timezone it represents?
```python
from datetime import timezone
utc_now = datetime.now(timezone.utc)      # explicit, unambiguous UTC time
naive_now = datetime.now()                 # NO timezone info attached -- ambiguous, avoid for stored data
```
No. A "naive" datetime — one with no timezone attached — is silently ambiguous about which timezone it represents. Comparing a naive datetime to an aware one raises an error. Comparing two naive datetimes from *different* timezones — a UTC server against a user's local browser time, say — without realizing it, produces wrong-but-silent results, with no error at all.

**A naive datetime is a phone number with no country code.**
```
"2:00 PM" (naive)          "2:00 PM UTC" (aware)          "2:00 PM EST" (aware)
     │                             │                              │
     ▼                             ▼                              ▼
  Which "2:00 PM"?             Unambiguous —              Unambiguous —
  Server's? User's?            same instant                same instant,
  No way to know                everywhere                 different clock reading
                                                             elsewhere
```
It might work by accident, as long as everyone's in the same place. It breaks silently the moment anything crosses a boundary — a server in one region, a user's browser in another. Store and compare timezone-aware datetimes in a real system, always.

### Summary example
A maintenance log records `datetime.now()` — naive — on a server running in UTC. The operations dashboard displaying it assumes local Central time.

1. A reading gets logged at "14:00" server time.
2. It's displayed as "14:00," with no conversion applied.
3. That's silently 5-6 hours off from what a Central-time viewer would assume — and no error gets raised anywhere, because nothing about a naive datetime carries the information needed to catch the mismatch.

Using `datetime.now(timezone.utc)` from the start, and converting explicitly at display time, removes the ambiguity entirely.

---

## Cluster 2 — Regex

### 1. How do you check whether a string matches a pattern at all?
```python
import re
bool(re.search(r"\d{4}-\d{2}-\d{2}", "log entry 2026-07-18: OK"))   # True -- search finds it ANYWHERE in the string
bool(re.match(r"\d{4}-\d{2}-\d{2}", "log entry 2026-07-18: OK"))    # False -- match only anchors at the START
```
`re.match` only checks the very beginning of the string. It fails silently on a pattern that appears anywhere else. `re.search` scans the whole string instead. Reaching for `match` when you meant `search` is a frequent, quiet bug.

**Where each function is even allowed to find a hit:**
```
string:   "log entry 2026-07-18: OK"
           ▲
  match()  └── can ONLY succeed if the pattern starts EXACTLY here → fails, pattern starts later

string:   "log entry 2026-07-18: OK"
           ────────▶ ────────▶ ────────▶  scans forward until it finds a hit anywhere
  search()                     ▲
                                └── succeeds here, position 10
```
`match` has its feet nailed to position 0. It never even looks further into the string if the pattern doesn't start immediately there. `search` walks the whole string, looking for the pattern to start anywhere.

### 2. Once a match succeeds, how do you extract specific parts of it, not just confirm it happened?
```python
m = re.search(r"(\d{4})-(\d{2})-(\d{2})", "date: 2026-07-18")
m.group(0)     # '2026-07-18' -- the FULL match
m.group(1)     # '2026'       -- first parenthesized group
m.groupdict()  # use named groups (?P<year>\d{4}) etc. for readable access instead of positional numbers
```

### 3. `search` finds the first match. What if the pattern appears multiple times, and you need all of them?
```python
re.findall(r"\b\w+@\w+\.\w+\b", "contact a@x.com or b@y.com")   # ['a@x.com', 'b@y.com']
```

### 4. Instead of extracting matches, how do you replace every match with something else?
```python
re.sub(r"\s+", " ", "too    many     spaces")    # 'too many spaces' -- collapse repeated whitespace
```

### 5. Every pattern above was written as `r"..."`. Why does that matter?
```python
re.search(r"\d+", text)    # correct: \d is a regex token
re.search("\d+", text)     # works but triggers a DeprecationWarning: \d isn't a valid Python escape sequence
```
Here's what's happening underneath:
1. In a plain, non-raw string, Python tries to interpret `\d`, `\s`, and similar sequences as Python escape sequences first.
2. Most of them aren't valid Python escapes, so they pass through unchanged — which is why it *seems* to work.
3. But `\n` and `\t` *are* valid Python escapes, and they'd get silently transformed before regex ever even sees the pattern.

That's fragile enough to avoid entirely. Always use `r"..."` for regex patterns, no exceptions.

### Summary example
Extracting and normalizing dates from a messy log file.
1. `re.findall(r"\d{4}-\d{2}-\d{2}", log_text)` — not `re.match`, since dates appear mid-line — pulls every date anywhere in the text.
2. `re.sub(r"\s+", " ", line)` first collapses any irregular whitespace, so the date pattern matches reliably.
3. Every pattern is written as `r"..."`, so `\d` is never accidentally mangled by Python's own string-escaping before regex gets a chance to see it.

---

## Cluster 3 — File I/O

### 1. How do you read and write the most common structured text format, JSON?
```python
import json
with open("config.json") as f:
    config = json.load(f)          # file -> Python dict/list
json_str = json.dumps(config, indent=2)   # Python object -> JSON string, indent=2 for human-readable output
```

### 2. What if the file is CSV instead, and you don't want to pull in all of pandas for a lightweight case?
```python
import csv
with open("data.csv", newline="") as f:
    reader = csv.DictReader(f)      # each row becomes a dict keyed by header
    rows = [row for row in reader]
```
`newline=""` matters specifically on Windows. Without it, Python's universal newline translation can interact with the csv module's own line-ending handling and produce extra blank rows — Python's own csv module documentation calls this out for exactly this reason.

### 3. For a real data pipeline, not a one-off script, why reach for Parquet instead of CSV?
```python
df.to_parquet("data.parquet", engine="pyarrow", index=False)
df2 = pd.read_parquet("data.parquet")
```
Parquet stores actual data types — a date column stays a date, instead of a string that gets re-parsed on every reload. It compresses far better than plain text. And since it's columnar, you can read back just the specific columns you need without loading the whole file. CSV is a plain-text, lowest-common-denominator format with none of these properties — fine for a quick one-off export, not for a real pipeline.

### 4. What if the data lives in a SQL database rather than a file at all?
```python
import sqlite3
conn = sqlite3.connect("app.db")
df = pd.read_sql("SELECT * FROM sensor_readings WHERE wear_pct > ?", conn, params=(40,))
conn.close()
```
`params=(40,)` instead of an f-string isn't just a style preference. An f-string like `f"...WHERE wear_pct > {value}"` is a direct SQL-injection vector the instant `value` comes from anywhere user-controlled. A parameterized query lets the database driver handle escaping safely. Make this habit automatic even in a "just my own script" context — scripts have a way of quietly becoming real services.

### Summary example
A pipeline reads sensor thresholds from a small `config.json`, pulls the actual sensor readings via a parameterized `pd.read_sql` query — never an f-string, even though "it's just an internal script" — and writes the cleaned result to Parquet rather than CSV. Why Parquet here specifically: this same file gets re-read by three other pipeline stages, and Parquet's preserved types and columnar reads mean each stage can cheaply pull just the columns it needs, instead of re-parsing a full CSV from scratch every time.

---

## Cluster 4 — Performance and Correctness Patterns

### 1. How do you time a piece of code properly, not just eyeball it?
```python
import time
start = time.perf_counter()
# ... code to time ...
elapsed = time.perf_counter() - start
```
`time.perf_counter()` is a monotonic clock, built specifically for measuring elapsed intervals. It's unaffected by system clock adjustments — an NTP sync, a manual clock change — that can occasionally make `time.time()` differences come out negative or wrong.

### 2. You can measure total time. How do you find out which specific part of the code is actually slow?
```python
import cProfile
cProfile.run("my_slow_function()")     # prints per-function call counts and cumulative time
```
Intuition about "which part is slow" is wrong surprisingly often. A profiler shows the actual bottleneck — frequently something unglamorous, like a repeated DataFrame copy inside a loop — rather than whichever part merely *looks* complicated.

### 3. Profiling a growing-DataFrame-in-a-loop pattern specifically tends to reveal a classic antipattern. What is it, and what's the fix?
```python
# slow: creates a new, larger DataFrame from scratch on every iteration
result = pd.DataFrame()
for chunk in chunks:
    result = pd.concat([result, process(chunk)])

# fast: append to a plain list, concatenate ONCE at the end
pieces = [process(chunk) for chunk in chunks]
result = pd.concat(pieces, ignore_index=True)
```
`pd.concat` inside a loop re-copies the entire growing DataFrame on every single iteration. That's O(n²) total work for something that should be O(n). Collecting pieces in a plain Python list — cheap, O(1) amortized append — and concatenating once at the end is a straightforward, very common real speedup. It's exactly the kind of thing `cProfile` above would surface as the actual bottleneck.

### 4. Beyond performance, how do type hints catch a whole class of bugs before runtime?
```python
def wear_risk_score(wear_pct: float, age_days: int, depot: str = "unknown") -> float:
    """Return a 0-1 risk score for scheduling maintenance."""
    ...
```
Tools like mypy, IDEs, and linters can catch a call to this function that passes a string where `wear_pct` should be a float — before runtime, not as a confusing error deep inside the function body, and not as a wrong result that silently "works" because Python's duck typing happened to let it through.

### 5. Type hints catch argument-type mistakes. What's a correctness bug that even correct type hints won't catch — the mutable default argument trap?
```python
# WRONG: the empty list is created ONCE at function definition time and reused across every call
def add_reading(value, readings=[]):
    readings.append(value)
    return readings

# RIGHT: default to None, create a fresh list inside the function body every call
def add_reading(value, readings=None):
    if readings is None:
        readings = []
    readings.append(value)
    return readings
```
Here's the mechanism, step by step:
1. Default argument values get evaluated exactly once — when the function is *defined*, not each time it's called.
2. A mutable default, like a list or dict, is that same one object, shared across every call that doesn't pass its own.
3. So it silently accumulates. Calling `add_reading(5)` twice returns `[5, 5]` on the second call, not a fresh `[5]`.

This is a real, classic Python bug, and it surprises almost everyone the first time they hit it.

**A mutable default argument is a shared whiteboard bolted to the function, not a fresh sheet of paper handed out per call:**
```
def add_reading(value, readings=[]):    ← this [] is built ONCE, at def-time,
                                            and lives for the lifetime of the function

call 1: add_reading(5)  → readings = [5]           (same [] object, now has one entry)
call 2: add_reading(9)  → readings = [5, 9]        (SAME [] object again — 5 is still there!)
call 3: add_reading(2)  → readings = [5, 9, 2]     (still the same whiteboard, never erased)
```
`readings=None` plus `if readings is None: readings=[]` hands each call a fresh sheet instead.

### Summary example
A logging utility accumulates readings across repeated calls without the caller realizing it, because it was defined as `def log(value, history=[])`.
1. Profiling with `cProfile` might even show this function getting suspiciously slower over time — the shared list keeps growing across the whole program's lifetime, forever.
2. The fix is the same `readings=None` pattern from above.
3. Timing the fix's effect with `time.perf_counter()`, before and after, confirms the growth — and the slowdown — actually stops.

---

## Cluster 5 — Concurrency Pitfalls (What AI-Engineer Interviews Actually Probe)

AI-engineer interviews lean on async Python constantly, because most GenAI backends are I/O-bound — they spend most of their time waiting on LLM API calls. These come up as "fundamental, should just know it" questions, not bonus points.

### 1. Two threads increment a shared counter at the same time, and the final count is wrong. What actually happened, and what's the standard fix?
This is a **race condition**. Here's the sequence that breaks it:
1. Both threads read the counter's current value.
2. Both compute "value + 1" from that same stale read.
3. Both write back — and one increment quietly gets lost.

The standard fix is a **lock** (`threading.Lock`, sometimes called a mutex). A thread has to acquire the lock before touching the shared value. Any other thread that tries to acquire it while it's held just waits its turn. Only one thread touches the data at a time, by construction, not by hoping the timing works out.

A second, complementary fix: use an **immutable** data structure — a tuple instead of a list. Since it can't be modified in place, anything that wants to "change" it has to create a new copy. There's nothing shared left for two threads to race over.

### 2. Locks solve races within one process. What's the Python-specific wrinkle that catches people off guard here — something unique to this language?
The **Global Interpreter Lock**, or GIL. It's a language-level mutex that allows only one thread to execute Python bytecode at a time, in any Python program, no matter how many threads you spawn.

That means multi-threading in Python gives you **concurrency** — multiple things making progress by interleaving — but not true **parallelism** — multiple things literally executing at the same instant — for CPU-bound work. To actually parallelize CPU-heavy work across cores, you need `multiprocessing`: separate Python interpreter processes, each with its own GIL. The GIL exists to keep Python's memory management (reference counting) simple and thread-safe, without needing a lock on every single object.

### 3. The GIL limits threads for CPU-bound work. So why is async/await still worth using in an AI backend at all?
Because most AI-backend work isn't CPU-bound. It's **waiting** — on an LLM API response, a database query, a network call — and the GIL doesn't block waiting.

`asyncio` lets one thread hold many in-flight "waiting" tasks at once, switching to whichever one becomes ready. That's exactly the shape of a backend making several concurrent LLM calls or DB lookups. This gets you concurrency without needing multiple threads or processes at all — the right tool specifically because the bottleneck is I/O, not computation.

### 4. Async is the right tool for I/O-bound work. What are the concrete ways people break it in practice?
- **Blocking the event loop with a CPU-heavy task inside an async function.** One long synchronous computation freezes every other task waiting on that same event loop — defeating the entire point.
- **Using a synchronous library inside async code.** `time.sleep()` instead of `asyncio.sleep()`, or the synchronous `requests` library instead of an async-native one like `httpx` — both block the whole event loop exactly like the CPU-heavy case above, just less obviously.
- **Scheduling a task and forgetting to `await` it.** The task can fail silently. Nothing is watching for its result, so the failure produces no error and no log — just a task that quietly never happened.
- **Harder debugging, in general.** Interleaved async execution makes stack traces and step-through debugging genuinely more confusing than synchronous code. Worth naming as a real cost, not just extra latency to fix for free.

### Summary example
A FastAPI backend handling chat requests uses `requests.get()` inside an `async def` route handler to call an external LLM API.

1. Under load, response times degrade far more than the API's own latency would explain.
2. The synchronous `requests` call is blocking the entire event loop on every single request, so no other request can make progress while one is waiting on the network.
3. Switching to `httpx`'s async client fixes it.

Separately, a shared in-memory request counter, incremented from multiple worker threads, occasionally undercounts — a classic race condition, fixed with a `threading.Lock` around the increment. Neither bug throws an exception. Both silently produce wrong behavior, which is the common thread across this entire cluster.

---

## Practice Q&A (Self-Test)

**Q1. Why is `%Y-%m-%d` specifically worth memorizing as a date format, over something like `%m/%d/%Y`?**
A: `%Y-%m-%d` is the ISO-8601 date format, and it sorts correctly as plain text — lexicographic order equals chronological order — unlike `%m/%d/%Y`. That's why it's the standard for filenames, log timestamps, and database columns.

**Q2. What's the mnemonic for remembering which of `strptime`/`strftime` parses and which formats?**
A: The letter right after "str" tells you the direction: **p**arse = **strp**time (string to datetime), and **f**ormat = **strf**time (datetime to string).

**Q3. Why should you use `timedelta` for date arithmetic instead of manually incrementing a `.day` attribute?**
A: Manually incrementing `.day` breaks at month and year boundaries — day 31 of a 30-day month doesn't exist, for example. `timedelta` handles all calendar arithmetic correctly, leap years included, so it should be used for adding or subtracting time rather than manual field manipulation.

**Q4. Why are "naive" datetimes (no timezone attached) a real production bug source?**
A: A naive datetime is silently ambiguous about which timezone it represents. Comparing a naive datetime to an aware one raises an error, and comparing two naive datetimes from different timezones — a UTC server versus a user's local browser time — without realizing it, produces wrong-but-silent results. Real systems should store and compare timezone-aware datetimes.

**Q5. What's the difference between `re.match` and `re.search`, and why is confusing them a real gotcha?**
A: `re.match` only checks the beginning of the string, while `re.search` scans the whole string for the pattern anywhere. Reaching for `match` when you meant `search` is a frequent, quiet bug, because `match` fails silently on a pattern that appears anywhere other than the very start.

**Q6. Why should regex patterns always be written as raw strings (`r"..."`)?**
A: In a plain, non-raw string, Python tries to interpret sequences like `\d` as Python escape sequences first. Most regex tokens aren't valid Python escapes, so they pass through — but that's fragile, since sequences like `\n` or `\t` *are* valid Python escapes and would get silently transformed before regex ever sees them. Raw strings avoid this entirely, with no exceptions needed.

**Q7. Why does `csv.DictReader` usage on Windows specifically call for `open("data.csv", newline="")`?**
A: Without `newline=""`, Python's universal newline translation can interact with the csv module's own line-ending handling and produce extra blank rows. This exact argument is called out in Python's own csv module documentation for this reason.

**Q8. Why prefer Parquet over CSV for a real data pipeline rather than a quick one-off export?**
A: Parquet stores actual data types — a date column stays a date rather than a string that needs re-parsing on every reload — compresses far better than plain text, and, being columnar, lets you read back only the specific columns you need. CSV is a plain-text, lowest-common-denominator format with none of these properties, fine for small exports but not for a real pipeline.

**Q9. Why is `pd.read_sql(query, conn, params=(40,))` safer than building the query with an f-string, and why does that habit matter even for "just my own script"?**
A: An f-string like `f"...WHERE wear_pct > {value}"` is a direct SQL-injection vector the moment `value` comes from anywhere user-controlled; a parameterized query lets the database driver handle escaping safely. This habit should be automatic even in a "just my own script" context, since scripts have a way of becoming real services.

**Q10. Why does growing a DataFrame with `pd.concat` inside a loop cause a real performance problem, and what's the fix?**
A: `pd.concat` inside a loop re-copies the entire growing DataFrame on every iteration, turning what should be O(n) work into O(n²). The fix is to collect each piece in a plain Python list — cheap, O(1) amortized append — and call `pd.concat` once at the end.

**Q11. Why does calling `add_reading(5)` twice with `def add_reading(value, readings=[])` return `[5, 5]` on the second call instead of `[5]` both times?**
A: Default argument values are evaluated exactly once, at function definition time — the empty list `[]` is created a single time and shared across every call that doesn't pass its own `readings` argument, so it silently accumulates. The fix is defaulting to `None` and creating a fresh list inside the function body on each call.
