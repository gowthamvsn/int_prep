# LeetCode "Design a Data Structure" — 12 Real Interview Problems, Answers Included

A distinct cluster from `leetcode-arrays-strings.md` and `leetcode-dp-trees-graphs.md`: instead of "compute one answer," these ask you to **design a class** that supports several operations efficiently, or to run a graph/DP algorithm you already know on a slightly different question (return the order, not just yes/no; return all answers, not just one). This exact list — LRU Cache, Trie, Design Twitter/TinyURL/Rate Limiter, Serialize/Deserialize a tree, Median of Two Sorted Arrays, Alien Dictionary, Course Schedule II, Kth Largest in a Stream, Online Stock Span, Word Break II — is one of the most frequently reported coding-round sets for AI Engineer interviews specifically, not just generic SWE loops. Same terse format as the other LeetCode files: problem, solution, one-line why.

## Design a Data Structure

**1. LRU Cache** — `get`/`put` in O(1), evict the least-recently-used item when full.
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```
*Technique: `OrderedDict.move_to_end` + `popitem(last=False)` gives O(1) "mark as most recently used" and O(1) eviction of the least recently used. The near-universal follow-up — "implement it without `OrderedDict`" — wants a hashmap plus a doubly linked list doing the exact same two operations by hand.*

**2. Trie (Prefix Tree)** — `insert`, `search` (exact word), `starts_with` (prefix).
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True

    def search(self, word):
        node = self._find(word)
        return node is not None and node.is_word

    def starts_with(self, prefix):
        return self._find(prefix) is not None

    def _find(self, s):
        node = self.root
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node
```
*Technique: each node is just a dict of children plus an end-of-word flag. `insert`/`search`/`starts_with` are the same "walk one character at a time" loop — O(word length), independent of how many words are stored, which is the whole reason a trie beats a plain set of strings for prefix queries.*

**3. Design Twitter** — post a tweet, get the 10 most recent tweets from people you follow (including yourself), follow/unfollow.
```python
import heapq
from collections import defaultdict

class Twitter:
    def __init__(self):
        self.time = 0
        self.tweets = defaultdict(list)      # user -> [(time, tweetId), ...]
        self.following = defaultdict(set)

    def post_tweet(self, user_id, tweet_id):
        self.tweets[user_id].append((self.time, tweet_id))
        self.time -= 1        # decreasing counter doubles as a max-heap sort key

    def get_news_feed(self, user_id):
        heap = []
        for u in self.following[user_id] | {user_id}:
            for t in self.tweets[u][-10:]:
                heapq.heappush(heap, t)
        return [tid for _, tid in heapq.nsmallest(10, heap)]

    def follow(self, follower_id, followee_id):
        if follower_id != followee_id:
            self.following[follower_id].add(followee_id)

    def unfollow(self, follower_id, followee_id):
        self.following[follower_id].discard(followee_id)
```
*Technique: only ever merge each followed user's most recent 10 tweets, never their full history — that bound is what keeps `get_news_feed` fast even for a user who's posted thousands of times. The decreasing global counter is a small trick: it's simultaneously a timestamp and a sort key, so the smallest values in the heap are automatically the newest tweets.*

**4. Design TinyURL** — `encode(long_url)` → short code, `decode(short_url)` → original.
```python
import random, string

class TinyURL:
    def __init__(self):
        self.code_to_url = {}
        self.alphabet = string.ascii_letters + string.digits

    def encode(self, long_url):
        code = "".join(random.choices(self.alphabet, k=6))
        while code in self.code_to_url:
            code = "".join(random.choices(self.alphabet, k=6))
        self.code_to_url[code] = long_url
        return "http://tiny.url/" + code

    def decode(self, short_url):
        code = short_url.rsplit("/", 1)[-1]
        return self.code_to_url[code]
```
*Technique: a random fixed-length code plus a collision check is the standard single-machine answer — 62^6 ≈ 56 billion possible codes makes collisions rare enough that retry-on-collision is fine. The real follow-up worth knowing: at true multi-server scale you'd switch to a global counter encoded in base62, since checking for collisions across machines is the expensive part a random code was trying to avoid in the first place.*

**5. Design a Rate Limiter (token bucket)** — allow a request only if a token is available; tokens refill over time.
```python
import time

class RateLimiter:
    def __init__(self, max_tokens, refill_rate_per_sec):
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.refill_rate = refill_rate_per_sec
        self.last_check = time.time()

    def allow_request(self):
        now = time.time()
        elapsed = now - self.last_check
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_check = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```
*Technique: tokens refill continuously, proportional to elapsed time, rather than resetting in discrete windows — that's what makes token-bucket smoother than a naive "N requests per fixed window" counter, which lets a client burst 2×N requests right across a window boundary (N at the end of one window, N at the start of the next).*

**6. Serialize and Deserialize a Binary Tree**
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def serialize(root):
    vals = []
    def dfs(node):
        if not node:
            vals.append("#")
            return
        vals.append(str(node.val))
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return ",".join(vals)

def deserialize(data):
    vals = iter(data.split(","))
    def build():
        val = next(vals)
        if val == "#":
            return None
        node = TreeNode(int(val))
        node.left = build()
        node.right = build()
        return node
    return build()
```
*Technique: pre-order traversal (root, then left, then right) with an explicit `"#"` sentinel for every null child is what makes deserialization unambiguous — rebuilding consumes values in exactly the order they were written, with no separate index bookkeeping needed.*

**7. Kth Largest Element in a Stream** — `add(val)` returns the current kth-largest value after insertion.
```python
import heapq

class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.heap = nums[:]
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val):
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]
```
*Technique: maintain a MIN-heap of exactly the k largest values seen so far — the smallest element in that heap (the root) is always the kth-largest overall, and each `add` costs O(log k) instead of re-sorting the entire stream on every call.*

**8. Online Stock Span** — for each day's price, how many consecutive prior days (including today) had a price ≤ today's?
```python
class StockSpanner:
    def __init__(self):
        self.stack = []   # (price, span)

    def next(self, price):
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        self.stack.append((price, span))
        return span
```
*Technique: a monotonic decreasing stack of (price, span) pairs lets each new price "absorb" every smaller-or-equal price directly behind it in one pop loop. This is amortized O(1) per call — a single call's while-loop can look expensive, but every price is pushed exactly once and popped at most once across the entire stream.*

## Topological Sort — Ordering Under Constraints

**9. Alien Dictionary** — given words already sorted per an unknown alphabet's rules, reconstruct that alphabet's letter order.
```python
from collections import defaultdict, deque

def alien_order(words):
    graph = defaultdict(set)
    in_degree = {c: 0 for w in words for c in w}
    for w1, w2 in zip(words, words[1:]):
        min_len = min(len(w1), len(w2))
        if w1[:min_len] == w2[:min_len] and len(w1) > len(w2):
            return ""   # invalid: a word can't come before its own strict prefix
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    order = []
    while queue:
        c = queue.popleft()
        order.append(c)
        for nxt in graph[c]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
    return "".join(order) if len(order) == len(in_degree) else ""
```
*Technique: only the FIRST differing character between each pair of adjacent words gives a real ordering constraint — build one graph edge per adjacent pair from that, then it's plain Kahn's-algorithm topological sort. The invalid case almost everyone misses on a first attempt: a longer word appearing before its own strict prefix (e.g. `"abc"` before `"ab"`), which can never happen in a validly-sorted dictionary.*

**10. Course Schedule II** — given prerequisites, return a valid course order (or `[]` if impossible).
```python
from collections import defaultdict, deque

def find_order(num_courses, prerequisites):
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    queue = deque([c for c in range(num_courses) if in_degree[c] == 0])
    order = []
    while queue:
        c = queue.popleft()
        order.append(c)
        for nxt in graph[c]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
    return order if len(order) == num_courses else []
```
*Technique: the identical Kahn's-algorithm topological sort as Course Schedule I (cycle detection only, already in `leetcode-dp-trees-graphs.md`) — the only change is returning the actual order list instead of a yes/no boolean, which is exactly why interviewers like pairing the two: it checks whether you understood the algorithm or just memorized "3-state DFS says no cycle."*

## Hard Array/String Follow-ups

**11. Median of Two Sorted Arrays**
```python
def find_median_sorted_arrays(a, b):
    if len(a) > len(b):
        a, b = b, a
    n1, n2 = len(a), len(b)
    lo, hi = 0, n1
    half = (n1 + n2 + 1) // 2
    while lo <= hi:
        i = (lo + hi) // 2
        j = half - i
        a_left = a[i - 1] if i > 0 else float("-inf")
        a_right = a[i] if i < n1 else float("inf")
        b_left = b[j - 1] if j > 0 else float("-inf")
        b_right = b[j] if j < n2 else float("inf")
        if a_left <= b_right and b_left <= a_right:
            if (n1 + n2) % 2:
                return max(a_left, b_left)
            return (max(a_left, b_left) + min(a_right, b_right)) / 2
        elif a_left > b_right:
            hi = i - 1
        else:
            lo = i + 1
```
*Technique: binary search on the partition point of the SMALLER array (never a merge) is what gets this to O(log(min(n, m))) instead of O(n+m). "Just merge them and take the middle" is a correct answer to a different, easier question than the one actually being asked here — the O(log) requirement is usually stated explicitly precisely to rule that shortcut out.*

**12. Word Break II** — return ALL ways to segment a string into dictionary words (not just whether it's possible).
```python
def word_break_ii(s, word_dict):
    word_set = set(word_dict)
    memo = {}

    def backtrack(start):
        if start == len(s):
            return [""]
        if start in memo:
            return memo[start]
        sentences = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                for rest in backtrack(end):
                    sentences.append(word if not rest else word + " " + rest)
        memo[start] = sentences
        return sentences

    return backtrack(0)
```
*Technique: same "can this prefix be segmented" idea as Word Break I (a boolean DP, already covered), but memoizing the LIST of valid completions from each start index — not just a yes/no — turns what looks like exponential re-exploration into one real computation per index. The boolean version's memo can't be reused here because two different call sites reaching the same index still need their own full sentence completions, not just confirmation that "some" completion exists.*

---

## Practice Q&A (Self-Test)

**Q1. Why does `LRUCache.get` call `move_to_end` even when the key's value isn't changing?**
A: Reading a key is itself a "use" — LRU eviction is about recency of access, not just recency of writes. Skipping `move_to_end` on `get` would make the cache track least-recently-*written* instead of least-recently-*used*, evicting keys that are actually being read constantly.

**Q2. In the Trie, why is `is_word` a separate flag instead of just checking "does this node have no children"?**
A: A word can be a strict prefix of another stored word (e.g. both `"car"` and `"card"` are inserted) — the node for `"car"` has children (the path continuing to `"card"`) but must still register as a complete word on its own. A childless node isn't the same condition as a complete word.

**Q3. In Design Twitter, why cap the per-user tweet slice at `[-10:]` before pushing into the heap, instead of pushing every tweet a followed user has ever posted?**
A: The feed only ever needs the top 10 overall, and each individual user's own tweets are already time-ordered by insertion — their 10 most recent are the only ones that could possibly make the final top 10. Pushing their entire history would make the heap size scale with total tweets ever posted instead of `10 × (number followed)`.

**Q4. Why does the Rate Limiter compute `elapsed * refill_rate` instead of just resetting `tokens` to `max_tokens` every fixed interval?**
A: A hard reset creates the "double burst at the window boundary" problem — a client could send `max_tokens` requests right before a reset and another `max_tokens` right after, doubling the intended rate for a brief window. Continuous, elapsed-time-proportional refilling has no such boundary to exploit.

**Q5. Alien Dictionary and Course Schedule II both end in the exact same Kahn's-algorithm while-loop. What's actually different between the two problems?**
A: Only how the graph gets built. Course Schedule II is handed the edges directly as `(course, prerequisite)` pairs. Alien Dictionary has to first DERIVE the edges by comparing adjacent words character-by-character and taking only the first difference — the topological sort itself, once the graph exists, is identical code.

**Q6. Why can't Word Break II reuse Word Break I's boolean memo table directly?**
A: Word Break I's memo answers "can index `i` reach the end" — a single true/false per index. Word Break II needs "what are ALL the full sentences reachable from index `i`" — a list per index, since two different earlier positions that both recurse into the same index `i` still need that index's complete set of onward sentences, not just a confirmation that at least one exists.
