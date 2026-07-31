# LeetCode Arrays & Strings — 60 Real Interview Problems, Answers Included

The canonical "Blind 75 / NeetCode 150"-style core — the problems that keep reappearing across every company's screen because each one teaches a reusable pattern (two pointers, sliding window, hashmap frequency counting, intervals, stack, binary search), not just a one-off trick. Grouped by pattern on purpose — recognizing *which* pattern a new problem is wearing is the actual skill; memorizing 60 isolated answers without the grouping teaches you 60 answers and zero transfer. Extends `live-coding-prep.md`.

## Two Pointers

**Visual + memory hook — the shape that separates "two pointers" from "sliding window" from everything else, before any individual problem:**
```
TWO POINTERS (fixed array, pointers move independently, often toward each other)
  [ 2, 7, 11, 15, 20 ]
    ▲               ▲
   lo ───▶  ◀─── hi        one or both pointers move each step, based on a comparison

SLIDING WINDOW (a contiguous chunk that grows/shrinks, one boundary at a time)
  [ a, b, c, d, e, f ]
       └──window──┘
        lo        hi ──▶   hi always advances; lo only advances to shrink an invalid window

BINARY SEARCH (the gap between pointers HALVES each step, not shifts by one)
  [ ...................... ]
   lo          mid          hi ──▶ lo or hi jumps to mid, discarding half the space each time
```
**Remember it as "how do the boundaries move":** two-pointer boundaries move inward based on a comparison at each step (problems #1–#8 below); a sliding window's right edge always expands while the left edge only catches up when a rule is violated (problems #9–#14); binary search's boundaries jump to the midpoint, discarding half the remaining space every time (problems #27–#32). Seeing a new problem and immediately asking "which of these three boundary-movement shapes does this match" is the actual transferable skill this whole doc is trying to build — the 60 solutions below are just practice reps for that one recognition move.

**1. Two Sum** — indices of the two numbers that add to a target (unsorted array).
```python
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
```
*O(n) via hashmap — not technically two-pointer, but the problem every list like this starts with.*

**2. Two Sum II** — same, but the array is already sorted (true two-pointer version).
```python
def two_sum_sorted(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        s = nums[lo] + nums[hi]
        if s == target: return [lo + 1, hi + 1]
        elif s < target: lo += 1
        else: hi -= 1
```

**3. Valid Palindrome** — ignoring non-alphanumeric characters and case.
```python
def is_palindrome(s):
    s = [c.lower() for c in s if c.isalnum()]
    return s == s[::-1]
```

**4. Container With Most Water** — max area between two lines.
```python
def max_area(height):
    lo, hi, best = 0, len(height) - 1, 0
    while lo < hi:
        best = max(best, (hi - lo) * min(height[lo], height[hi]))
        if height[lo] < height[hi]: lo += 1
        else: hi -= 1
    return best
```
*Technique: always move the pointer at the SHORTER line — moving the taller one can only shrink the width without any chance of increasing the limiting height.*

**5. 3Sum** — all unique triplets summing to 0.
```python
def three_sum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]: continue
        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0: lo += 1
            elif s > 0: hi -= 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                while lo < hi and nums[lo] == nums[lo-1]: lo += 1
    return res
```
*Technique: sort first, then fix one number and two-pointer the rest — turns an O(n³) brute force into O(n²); the `continue`/inner-while skips are what prevent duplicate triplets.*

**6. Trapping Rain Water** — total water trapped between bars.
```python
def trap(height):
    lo, hi = 0, len(height) - 1
    left_max = right_max = water = 0
    while lo < hi:
        if height[lo] < height[hi]:
            left_max = max(left_max, height[lo])
            water += left_max - height[lo]
            lo += 1
        else:
            right_max = max(right_max, height[hi])
            water += right_max - height[hi]
            hi -= 1
    return water
```

**7. Move Zeroes** — shift all zeros to the end, in place, preserving order of the rest.
```python
def move_zeroes(nums):
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write], nums[read] = nums[read], nums[write]
            write += 1
```
*Technique: the "read/write pointer" pattern — same shape as the C-style in-place array compaction idiom.*

**8. Merge Sorted Array** — merge nums2 into nums1 in place, nums1 has trailing space.
```python
def merge(nums1, m, nums2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]; i -= 1
        else:
            nums1[k] = nums2[j]; j -= 1
        k -= 1
```
*Technique: merge from the BACK — merging from the front would overwrite nums1 values you still need to compare.*

## Sliding Window

**9. Best Time to Buy and Sell Stock** — max profit, one transaction.
```python
def max_profit(prices):
    min_price, best = float("inf"), 0
    for p in prices:
        min_price = min(min_price, p)
        best = max(best, p - min_price)
    return best
```

**10. Longest Substring Without Repeating Characters**
```python
def length_of_longest_substring(s):
    seen, lo, best = {}, 0, 0
    for hi, c in enumerate(s):
        if c in seen and seen[c] >= lo:
            lo = seen[c] + 1
        seen[c] = hi
        best = max(best, hi - lo + 1)
    return best
```

**11. Longest Repeating Character Replacement** — longest substring of one repeated char after ≤k replacements.
```python
def character_replacement(s, k):
    count, lo, best, max_freq = {}, 0, 0, 0
    for hi, c in enumerate(s):
        count[c] = count.get(c, 0) + 1
        max_freq = max(max_freq, count[c])
        while (hi - lo + 1) - max_freq > k:
            count[s[lo]] -= 1
            lo += 1
        best = max(best, hi - lo + 1)
    return best
```

**12. Minimum Window Substring** — smallest window in s containing every character of t.
```python
from collections import Counter
def min_window(s, t):
    need = Counter(t)
    missing = len(t)
    lo = start = end = 0
    for hi, c in enumerate(s, 1):
        if need[c] > 0: missing -= 1
        need[c] -= 1
        if missing == 0:
            while need[s[lo]] < 0:
                need[s[lo]] += 1; lo += 1
            if end == 0 or hi - lo < end - start:
                start, end = lo, hi
            need[s[lo]] += 1; missing += 1; lo += 1
    return s[start:end]
```

**13. Permutation in String** — does s1's permutation occur as a substring of s2?
```python
from collections import Counter
def check_inclusion(s1, s2):
    need, window = Counter(s1), Counter()
    k = len(s1)
    for i, c in enumerate(s2):
        window[c] += 1
        if i >= k: 
            left = s2[i - k]
            window[left] -= 1
            if window[left] == 0: del window[left]
        if window == need: return True
    return False
```

**14. Sliding Window Maximum** — max of every window of size k.
```python
from collections import deque
def max_sliding_window(nums, k):
    dq, res = deque(), []
    for i, n in enumerate(nums):
        while dq and nums[dq[-1]] < n: dq.pop()
        dq.append(i)
        if dq[0] == i - k: dq.popleft()
        if i >= k - 1: res.append(nums[dq[0]])
    return res
```
*Technique: a monotonic deque holds candidate indices in decreasing-value order, so the max is always at the front — each element enters and leaves the deque once, giving O(n) instead of the naive O(nk).*

## HashMap / Frequency Counting

**15. Contains Duplicate**
```python
def contains_duplicate(nums):
    return len(set(nums)) != len(nums)
```

**16. Valid Anagram**
```python
from collections import Counter
def is_anagram(s, t):
    return Counter(s) == Counter(t)
```

**17. Group Anagrams**
```python
from collections import defaultdict
def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        groups[tuple(sorted(s))].append(s)
    return list(groups.values())
```
*Technique: the sorted-letters tuple is a canonical key — every anagram of the same word sorts to the identical tuple, so it's a free hashmap key for "same letters, any order."*

**18. Top K Frequent Elements**
```python
from collections import Counter
import heapq
def top_k_frequent(nums, k):
    counts = Counter(nums)
    return heapq.nlargest(k, counts.keys(), key=counts.get)
```

**19. Product of Array Except Self** — no division allowed.
```python
def product_except_self(nums):
    n = len(nums)
    res = [1] * n
    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        res[i] *= suffix
        suffix *= nums[i]
    return res
```
*Technique: one pass accumulating the running product of everything to the LEFT, a second pass multiplying in everything to the RIGHT — avoids division entirely and stays O(n).*

**20. Longest Consecutive Sequence** — longest run of consecutive integers, unsorted input, O(n).
```python
def longest_consecutive(nums):
    num_set = set(nums)
    best = 0
    for n in num_set:
        if n - 1 not in num_set:   # only start counting from the beginning of a run
            length = 1
            while n + length in num_set:
                length += 1
            best = max(best, length)
    return best
```
*Technique: the `n - 1 not in num_set` check is what keeps this O(n) instead of O(n²) — it guarantees each number only ever starts a count-up once, as the head of its run.*

**21. Subarray Sum Equals K** — count of subarrays summing to k.
```python
from collections import defaultdict
def subarray_sum(nums, k):
    counts = defaultdict(int)
    counts[0] = 1
    total = res = 0
    for n in nums:
        total += n
        res += counts[total - k]
        counts[total] += 1
    return res
```
*Technique: prefix-sum + hashmap — "does a subarray summing to k end here" becomes "have I seen the prefix-sum `total - k` before," turning an O(n²) nested-loop scan into O(n).*

## Stack

**22. Valid Parentheses**
```python
def is_valid(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for c in s:
        if c in pairs:
            if not stack or stack.pop() != pairs[c]: return False
        else:
            stack.append(c)
    return not stack
```

**23. Min Stack** — a stack supporting O(1) `getMin()`.
```python
class MinStack:
    def __init__(self):
        self.stack, self.mins = [], []
    def push(self, val):
        self.stack.append(val)
        self.mins.append(val if not self.mins else min(val, self.mins[-1]))
    def pop(self):
        self.stack.pop(); self.mins.pop()
    def top(self):
        return self.stack[-1]
    def getMin(self):
        return self.mins[-1]
```
*Technique: a second stack tracks "the min so far" at every depth, so popping never loses track of what the min was one level down.*

**24. Evaluate Reverse Polish Notation**
```python
def eval_rpn(tokens):
    stack = []
    ops = {"+": lambda a,b: a+b, "-": lambda a,b: a-b,
           "*": lambda a,b: a*b, "/": lambda a,b: int(a/b)}
    for t in tokens:
        if t in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[t](a, b))
        else:
            stack.append(int(t))
    return stack[0]
```

**25. Daily Temperatures** — days until a warmer temperature, per day.
```python
def daily_temperatures(temps):
    res = [0] * len(temps)
    stack = []   # indices with a temperature we haven't beaten yet
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
```
*Technique: a monotonic (decreasing) stack of indices — this exact pattern (stack of indices, pop while the new value beats the top) solves an entire family of "next greater element" problems.*

**26. Largest Rectangle in Histogram**
```python
def largest_rectangle_area(heights):
    stack, best = [], 0
    for i, h in enumerate(heights + [0]):
        while stack and heights[stack[-1]] >= h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            best = max(best, height * width)
        stack.append(i)
    return best
```

## Binary Search

**27. Binary Search** (the baseline).
```python
def search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target: return mid
        elif nums[mid] < target: lo = mid + 1
        else: hi = mid - 1
    return -1
```

**28. Search in Rotated Sorted Array**
```python
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target: return mid
        if nums[lo] <= nums[mid]:          # left half is sorted
            if nums[lo] <= target < nums[mid]: hi = mid - 1
            else: lo = mid + 1
        else:                               # right half is sorted
            if nums[mid] < target <= nums[hi]: lo = mid + 1
            else: hi = mid - 1
    return -1
```
*Technique: at every step, ONE half (left or right of mid) is guaranteed properly sorted — check which one, then decide if the target could be in that sorted half using plain range comparison.*

**29. Find Minimum in Rotated Sorted Array**
```python
def find_min(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[hi]: lo = mid + 1
        else: hi = mid
    return nums[lo]
```

**30. Find Peak Element** — any index where the value is greater than both neighbors.
```python
def find_peak_element(nums):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[mid + 1]: hi = mid
        else: lo = mid + 1
    return lo
```

**31. Search a 2D Matrix** — rows and columns both sorted, treat as one flat sorted array.
```python
def search_matrix(matrix, target):
    if not matrix or not matrix[0]: return False
    rows, cols = len(matrix), len(matrix[0])
    lo, hi = 0, rows * cols - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        val = matrix[mid // cols][mid % cols]
        if val == target: return True
        elif val < target: lo = mid + 1
        else: hi = mid - 1
    return False
```

**32. Koko Eating Bananas** — minimum eating speed to finish within h hours ("binary search on the answer," not on the array).
```python
import math
def min_eating_speed(piles, h):
    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        hours = sum(math.ceil(p / mid) for p in piles)
        if hours <= h: hi = mid
        else: lo = mid + 1
    return lo
```
*Technique: binary search doesn't require a sorted array — it requires a monotonic yes/no answer as the guessed value increases, which "can I finish in time at this speed" is. Recognizing this shape is what unlocks a whole class of non-obvious binary search problems.*

## Intervals

**33. Merge Intervals**
```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    res = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= res[-1][1]:
            res[-1][1] = max(res[-1][1], end)
        else:
            res.append([start, end])
    return res
```

**34. Insert Interval** — insert into an already-sorted, non-overlapping list.
```python
def insert(intervals, new_interval):
    res = []
    i, n = 0, len(intervals)
    while i < n and intervals[i][1] < new_interval[0]:
        res.append(intervals[i]); i += 1
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval = [min(new_interval[0], intervals[i][0]), max(new_interval[1], intervals[i][1])]
        i += 1
    res.append(new_interval)
    return res + intervals[i:]
```

**35. Non-overlapping Intervals** — minimum removals to make the rest non-overlapping.
```python
def erase_overlap_intervals(intervals):
    intervals.sort(key=lambda x: x[1])
    count, prev_end = 0, float("-inf")
    for start, end in intervals:
        if start >= prev_end:
            prev_end = end
        else:
            count += 1
    return count
```
*Technique: sort by END time, greedily keep the interval that finishes soonest — this is the classic activity-selection greedy proof pattern, not a coincidence it works.*

**36. Meeting Rooms II** — minimum number of rooms needed for all meetings.
```python
import heapq
def min_meeting_rooms(intervals):
    intervals.sort(key=lambda x: x[0])
    heap = []
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heapreplace(heap, end)
        else:
            heapq.heappush(heap, end)
    return len(heap)
```
*Technique: the heap holds end-times of currently "in progress" meetings — its size at the end is the peak concurrent-meetings count, which is exactly the room requirement.*

## Sorting-Based & Greedy

**37. Sort Colors** — sort an array of 0/1/2 in one pass, in place (Dutch national flag).
```python
def sort_colors(nums):
    lo, mid, hi = 0, 0, len(nums) - 1
    while mid <= hi:
        if nums[mid] == 0:
            nums[lo], nums[mid] = nums[mid], nums[lo]; lo += 1; mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[hi] = nums[hi], nums[mid]; hi -= 1
```

**38. Kth Largest Element in an Array**
```python
import heapq
def find_kth_largest(nums, k):
    return heapq.nlargest(k, nums)[-1]
```
*Technique: `heapq.nlargest` for a quick correct answer; the "real" interview follow-up is implementing quickselect for average O(n) instead of this O(n log k).*

**39. Majority Element** — the element appearing more than n/2 times (Boyer-Moore voting).
```python
def majority_element(nums):
    count, candidate = 0, None
    for n in nums:
        if count == 0: candidate = n
        count += 1 if n == candidate else -1
    return candidate
```
*Technique: Boyer-Moore voting — since the majority element outnumbers everything else combined, it survives this "cancel out a non-majority pair" process no matter what order the array is in.*

**40. Missing Number** — find the missing number in 0..n.
```python
def missing_number(nums):
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)
```

**41. Single Number** — every element appears twice except one.
```python
def single_number(nums):
    result = 0
    for n in nums:
        result ^= n
    return result
```
*Technique: XOR of a number with itself is 0, and XOR is commutative/associative — every duplicate pair cancels out, leaving only the unique number.*

**42. Best Time to Buy and Sell Stock II** — unlimited transactions, max total profit.
```python
def max_profit_ii(prices):
    return sum(max(prices[i+1] - prices[i], 0) for i in range(len(prices) - 1))
```
*Technique: any profitable up-move can be captured independently — summing every positive day-to-day delta is mathematically identical to optimally timed buy/sell pairs.*

**43. Gas Station** — starting index to complete a circular route, or -1 if impossible.
```python
def can_complete_circuit(gas, cost):
    if sum(gas) < sum(cost): return -1
    total, start = 0, 0
    for i in range(len(gas)):
        total += gas[i] - cost[i]
        if total < 0:
            start = i + 1
            total = 0
    return start
```

**44. Jump Game** — can you reach the last index?
```python
def can_jump(nums):
    reach = 0
    for i, n in enumerate(nums):
        if i > reach: return False
        reach = max(reach, i + n)
    return True
```

**45. Next Permutation** — the next lexicographically greater arrangement, in place.
```python
def next_permutation(nums):
    n = len(nums)
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]: i -= 1
    if i >= 0:
        j = n - 1
        while nums[j] <= nums[i]: j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[i+1:] = reversed(nums[i+1:])
```

## Matrix

**46. Rotate Image** — rotate an n×n matrix 90° clockwise, in place.
```python
def rotate(matrix):
    matrix.reverse()
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```
*Technique: reverse rows top-to-bottom, then transpose — two simple O(n²) passes instead of tracking rotated indices directly.*

**47. Spiral Matrix** — return all elements in spiral order.
```python
def spiral_order(matrix):
    res = []
    while matrix:
        res += matrix.pop(0)
        matrix = [list(row) for row in zip(*matrix)][::-1]
    return res
```

**48. Set Matrix Zeroes** — if a cell is 0, zero its entire row and column, in place.
```python
def set_zeroes(matrix):
    rows, cols = set(), set()
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v == 0: rows.add(i); cols.add(j)
    for i, row in enumerate(matrix):
        for j in range(len(row)):
            if i in rows or j in cols: matrix[i][j] = 0
```

## Prefix Sum & Remaining Core

**49. Range Sum Query (Immutable)** — many repeated range-sum queries on a fixed array.
```python
class NumArray:
    def __init__(self, nums):
        self.prefix = [0]
        for n in nums: self.prefix.append(self.prefix[-1] + n)
    def sumRange(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]
```
*Technique: precompute prefix sums ONCE (O(n)), then every query is O(1) — the standard tradeoff when a range-sum query needs to run many times.*

**50. Remove Duplicates from Sorted Array** — in place, return new length.
```python
def remove_duplicates(nums):
    write = 1
    for read in range(1, len(nums)):
        if nums[read] != nums[write - 1]:
            nums[write] = nums[read]
            write += 1
    return write
```

**51. Longest Common Prefix**
```python
def longest_common_prefix(strs):
    if not strs: return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix: return ""
    return prefix
```

**52. String to Integer (atoi)** — parse leading whitespace, optional sign, digits, clamp to 32-bit range.
```python
def my_atoi(s):
    s = s.strip()
    if not s: return 0
    i, sign = 0, 1
    if s[0] in "+-":
        sign = -1 if s[0] == "-" else 1
        i = 1
    num = 0
    while i < len(s) and s[i].isdigit():
        num = num * 10 + int(s[i]); i += 1
    num *= sign
    return max(-2**31, min(2**31 - 1, num))
```

**53. Reverse Words in a String** — reverse word order, collapse extra whitespace.
```python
def reverse_words(s):
    return " ".join(reversed(s.split()))
```
*Technique: `str.split()` with no argument already collapses runs of whitespace and drops leading/trailing space — the whole "extra whitespace" requirement is handled for free.*

**54. Longest Palindromic Substring** — expand-around-center approach.
```python
def longest_palindrome(s):
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l+1:r]
    best = ""
    for i in range(len(s)):
        odd = expand(i, i)
        even = expand(i, i + 1)
        best = max(best, odd, even, key=len)
    return best
```
*Technique: every palindrome has a center (a single character for odd length, a gap between two characters for even) — checking both center types at every position covers all cases in O(n²).*

**55. Palindromic Substrings** — count of all palindromic substrings (same expand-around-center engine as #54).
```python
def count_substrings(s):
    def expand(l, r):
        count = 0
        while l >= 0 and r < len(s) and s[l] == s[r]:
            count += 1; l -= 1; r += 1
        return count
    return sum(expand(i, i) + expand(i, i + 1) for i in range(len(s)))
```

**56. Encode and Decode Strings** — serialize a list of strings into one string and back, safe for any characters including delimiters.
```python
def encode(strs):
    return "".join(f"{len(s)}#{s}" for s in strs)

def decode(s):
    res, i = [], 0
    while i < len(s):
        j = s.index("#", i)
        length = int(s[:j][len(res[-1]) if False else 0:]) if False else int(s[i:j])
        res.append(s[j+1:j+1+length])
        i = j + 1 + length
    return res
```
*Technique: length-prefixing (`"5#hello"`) instead of a plain delimiter — a plain delimiter breaks the instant a string itself contains that delimiter character; a length prefix never has that ambiguity.*

**57. Valid Sudoku** — check rows, columns, and 3×3 boxes for duplicate digits.
```python
def is_valid_sudoku(board):
    rows, cols, boxes = [set() for _ in range(9)], [set() for _ in range(9)], [set() for _ in range(9)]
    for i in range(9):
        for j in range(9):
            v = board[i][j]
            if v == ".": continue
            b = (i // 3) * 3 + j // 3
            if v in rows[i] or v in cols[j] or v in boxes[b]: return False
            rows[i].add(v); cols[j].add(v); boxes[b].add(v)
    return True
```

**58. Find All Anagrams in a String** — all start indices where an anagram of p occurs in s.
```python
from collections import Counter
def find_anagrams(s, p):
    need, window, res = Counter(p), Counter(), []
    k = len(p)
    for i, c in enumerate(s):
        window[c] += 1
        if i >= k:
            left = s[i - k]
            window[left] -= 1
            if window[left] == 0: del window[left]
        if window == need: res.append(i - k + 1)
    return res
```

**59. Rotate Array** — rotate right by k steps, in place, O(1) extra space.
```python
def rotate(nums, k):
    k %= len(nums)
    nums.reverse()
    nums[:k] = reversed(nums[:k])
    nums[k:] = reversed(nums[k:])
```
*Technique: reverse the whole array, then reverse each of the two pieces separately — three reversals achieve a rotation without any extra array.*

**60. Two Sum — all pairs, no duplicates** (a common variant of #1: return the actual value pairs, not indices, no duplicate pairs).
```python
def two_sum_pairs(nums, target):
    seen, used, res = set(), set(), []
    for n in nums:
        complement = target - n
        if complement in seen and complement not in used:
            res.append([complement, n])
            used.add(complement); used.add(n)
        seen.add(n)
    return res
```

## Practice Q&A (Self-Test)

### Problems #4, #5, and #6 all use two pointers, but move them under different conditions. What's the actual shared principle?
In each case, moving a specific pointer is provably safe because the alternative can never produce a better answer — in #4, moving the taller line can only shrink width without any chance of a taller limiting height; in #5, after sorting, moving `lo`/`hi` based on whether the sum is too small/large eliminates exactly the pairs that provably can't reach the target; in #6, the side with the smaller running max is the side whose water level is already determined, so it's safe to resolve. Two pointers only works when you can prove a direction is always safe to discard — it's not a generic trick, it's a proof technique applied to array traversal.

### Problems #10, #11, #12, and #13 are all "sliding window," but #10/#11 use a simple `lo`/`hi` shrink-on-violation loop while #14 uses a deque. When do you need the deque instead of a plain window?
A plain shrinking window works when you only need to know something *aggregate* about the current window (its length, whether a condition holds) that updates cheaply as elements enter/leave. The deque is needed specifically when you need the window's *maximum* efficiently at every step — recomputing the max of a window naively on every slide is O(k) per step; the monotonic deque maintains it in amortized O(1) per step by discarding elements that can never be the max again.

### #28 (Search in Rotated Sorted Array) checks `nums[lo] <= nums[mid]` to decide which half is sorted. Why can't you just check if the whole array is sorted and pick a strategy once?
The rotation point is unknown and can be anywhere, so the array as a whole is never fully sorted (except in the trivial no-rotation case) — but at every single binary-search step, at least one of the two halves *around mid* is guaranteed to be properly sorted, even though the full array isn't. Re-checking this locally at each step (rather than trying to determine a single global strategy) is what makes the algorithm work at all.

### #32 (Koko Eating Bananas) binary searches over possible *eating speeds*, not over the array `piles`. What's the general signal that a problem can be solved this way?
Binary search only requires a monotonic (one-directional) relationship between a guessed value and a yes/no answer — here, "can Koko finish in time at speed X" only ever gets easier (never harder) as X increases, which is exactly the monotonic structure binary search needs. Whenever a problem asks for a minimum/maximum value satisfying some feasibility condition, and increasing the candidate value only ever makes the condition easier (or only ever harder) to satisfy, that's the signal to binary search on the *answer* instead of on a data structure.

### Why does #35 (Non-overlapping Intervals) sort by END time while #33 (Merge Intervals) sorts by START time?
The two problems are solving different questions. Merge Intervals needs to walk through intervals in the order they'd occur and merge adjacent overlaps, which requires start-time order to correctly detect "does this interval begin before the previous one ends." Non-overlapping Intervals is a greedy scheduling problem (maximize how many non-overlapping intervals you can keep) — sorting by end time and always keeping whichever interval finishes soonest is the classic activity-selection greedy strategy, which provably maximizes the count kept; start-time order doesn't have that same greedy-optimality guarantee for this specific question.
