# LeetCode DP, Trees & Graphs — 43 Real Interview Problems, Answers Included

Less universally asked of data scientists than SQL/Pandas/arrays, but they still show up — especially at companies that run a general SWE-style loop for DS/ML roles, or for anyone building actual graph/tree-shaped features (org charts, dependency graphs, `practice-langgraph`'s state graphs). Same terse format: problem, solution, one-line why. Recursion is the throughline — DP is recursion with memory, trees/graphs are recursion (or its iterative BFS/DFS cousin) over a branching structure instead of a line.

**Visual + memory hook — DP's entire value proposition is this picture: the SAME subtree gets recomputed over and over without memory, and collapses to one node with it:**
```
climb_stairs(4), no memory                    climb_stairs(4), memoized
                                               (or the bottom-up loop version,
        f(4)                                   same idea, no recursion needed)
       /    \
     f(3)   f(2)                                    f(4)
    /  \    /  \                                   /    \
  f(2) f(1) f(1) f(0)                            f(3)   f(2) ◀── already computed
  /  \                                            /  \        once, just READ this
f(1) f(0)                                       f(2) f(1)     time, not recomputed
                                                 /  \
7 calls to f(2)/f(1)/f(0) — most of them       f(1) f(0)
recomputing an answer already computed          only 5 distinct subproblems ever
one branch over, wasted work growing            actually computed, each ONCE —
exponentially with the input size               everything else is a cache hit
```
**Remember it as "does this recursion tree have repeated subtrees, and can I afford to remember them":** that question, asked before writing any code, is what tells you a problem is DP-shaped in the first place (repeated overlapping subproblems) rather than plain recursion or backtracking (each subtree genuinely distinct, nothing to cache). Every `dp[]` array or `@lru_cache` in this section is doing exactly one thing — turning a re-drawn branch back into a single already-answered node.

## Dynamic Programming — 1D

**1. Climbing Stairs** — ways to reach step n, 1 or 2 steps at a time.
```python
def climb_stairs(n):
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
```
*Technique: this is literally Fibonacci — `ways(n) = ways(n-1) + ways(n-2)` because the last move was either a 1-step or a 2-step.*

**2. House Robber** — max sum, no two adjacent elements.
```python
def rob(nums):
    prev, curr = 0, 0
    for n in nums:
        prev, curr = curr, max(curr, prev + n)
    return curr
```

**3. House Robber II** — same, but houses are in a circle (first and last are adjacent).
```python
def rob_ii(nums):
    if len(nums) == 1: return nums[0]
    def rob_line(houses): 
        prev, curr = 0, 0
        for n in houses:
            prev, curr = curr, max(curr, prev + n)
        return curr
    return max(rob_line(nums[1:]), rob_line(nums[:-1]))
```
*Technique: "circular, no two adjacent" reduces to two runs of the plain linear version — once excluding the first house, once excluding the last — because a valid circular selection can never include both ends.*

**4. Min Cost Climbing Stairs**
```python
def min_cost_climbing_stairs(cost):
    a, b = 0, 0
    for i in range(2, len(cost) + 1):
        a, b = b, min(b + cost[i-1], a + cost[i-2])
    return b
```

**5. Longest Increasing Subsequence**
```python
import bisect
def length_of_lis(nums):
    tails = []
    for n in nums:
        i = bisect.bisect_left(tails, n)
        if i == len(tails): tails.append(n)
        else: tails[i] = n
    return len(tails)
```
*Technique: `tails[i]` = smallest possible tail value of an increasing subsequence of length `i+1` — this greedy-with-binary-search approach gets O(n log n) instead of the O(n²) naive DP.*

**6. Coin Change** — fewest coins to make an amount (or -1).
```python
def coin_change(coins, amount):
    dp = [0] + [float("inf")] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1
```

**7. Word Break** — can s be segmented into dictionary words?
```python
def word_break(s, word_dict):
    words = set(word_dict)
    dp = [True] + [False] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[-1]
```

**8. Decode Ways** — number of ways to decode a digit string (A=1..Z=26).
```python
def num_decodings(s):
    if not s or s[0] == "0": return 0
    prev, curr = 1, 1
    for i in range(1, len(s)):
        temp = 0
        if s[i] != "0": temp += curr
        if 10 <= int(s[i-1:i+1]) <= 26: temp += prev
        prev, curr = curr, temp
    return curr
```

## Dynamic Programming — 2D / Knapsack

**9. Unique Paths** — grid, top-left to bottom-right, only right/down moves.
```python
def unique_paths(m, n):
    dp = [1] * n
    for _ in range(m - 1):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[-1]
```

**10. Unique Paths II** — same, with obstacles (marked 1) blocking cells.
```python
def unique_paths_with_obstacles(grid):
    n = len(grid[0])
    dp = [0] * n
    dp[0] = 1
    for row in grid:
        for j in range(n):
            if row[j] == 1: dp[j] = 0
            elif j > 0: dp[j] += dp[j - 1]
    return dp[-1]
```

**11. Minimum Path Sum**
```python
def min_path_sum(grid):
    rows, cols = len(grid), len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0: continue
            elif i == 0: grid[i][j] += grid[i][j-1]
            elif j == 0: grid[i][j] += grid[i-1][j]
            else: grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    return grid[-1][-1]
```

**12. Longest Common Subsequence**
```python
def longest_common_subsequence(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1] + 1
            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]
```

**13. Edit Distance** — min insert/delete/replace operations to transform word1 into word2.
```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```

**14. 0/1 Knapsack** — max value, weight capacity W, each item used at most once (the DP every other knapsack-shaped problem, including #15/#16 below, derives from).
```python
def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    return dp[n][W]
```

**15. Partition Equal Subset Sum** — can the array split into two equal-sum halves? (0/1 knapsack in disguise: "can I hit exactly sum/2.")
```python
def can_partition(nums):
    total = sum(nums)
    if total % 2: return False
    target = total // 2
    dp = {0}
    for n in nums:
        dp |= {n + x for x in dp if n + x <= target}
    return target in dp
```

**16. Target Sum** — ways to assign +/- to each number to reach a target (also knapsack in disguise).
```python
from collections import defaultdict
def find_target_sum_ways(nums, target):
    counts = defaultdict(int)
    counts[0] = 1
    for n in nums:
        next_counts = defaultdict(int)
        for total, c in counts.items():
            next_counts[total + n] += c
            next_counts[total - n] += c
        counts = next_counts
    return counts[target]
```

## Backtracking

**17. Subsets** — all possible subsets (the power set).
```python
def subsets(nums):
    res = [[]]
    for n in nums:
        res += [s + [n] for s in res]
    return res
```

**18. Permutations**
```python
def permute(nums):
    if len(nums) <= 1: return [nums]
    res = []
    for i in range(len(nums)):
        rest = nums[:i] + nums[i+1:]
        for p in permute(rest):
            res.append([nums[i]] + p)
    return res
```

**19. Combination Sum** — unique combinations summing to target, numbers reusable.
```python
def combination_sum(candidates, target):
    res = []
    def backtrack(start, path, remaining):
        if remaining == 0: res.append(path[:]); return
        if remaining < 0: return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])   # i, not i+1: reuse allowed
            path.pop()
    backtrack(0, [], target)
    return res
```
*Technique: passing `i` (not `i+1`) into the recursive call is what allows the same number to be reused — the single-character difference between "combination with reuse" and "combination without reuse."*

**20. Word Search** — does the word exist as a path of adjacent cells in a grid?
```python
def exist(board, word):
    rows, cols = len(board), len(board[0])
    def backtrack(r, c, i):
        if i == len(word): return True
        if r < 0 or c < 0 or r >= rows or c >= cols or board[r][c] != word[i]: return False
        temp, board[r][c] = board[r][c], "#"
        found = any(backtrack(r+dr, c+dc, i+1) for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)])
        board[r][c] = temp
        return found
    return any(backtrack(r, c, 0) for r in range(rows) for c in range(cols))
```
*Technique: temporarily marking the visited cell (`"#"`) instead of a separate visited-set is a common space-saving trick — restore it on backtrack so other paths can still use that cell.*

**21. Generate Parentheses** — all valid combinations of n pairs.
```python
def generate_parenthesis(n):
    res = []
    def backtrack(s, open_count, close_count):
        if len(s) == 2 * n: res.append(s); return
        if open_count < n: backtrack(s + "(", open_count + 1, close_count)
        if close_count < open_count: backtrack(s + ")", open_count, close_count + 1)
    backtrack("", 0, 0)
    return res
```

**22. Letter Combinations of a Phone Number**
```python
def letter_combinations(digits):
    if not digits: return []
    mapping = {"2":"abc","3":"def","4":"ghi","5":"jkl","6":"mno","7":"pqrs","8":"tuv","9":"wxyz"}
    res = [""]
    for d in digits:
        res = [p + c for p in res for c in mapping[d]]
    return res
```

**23. N-Queens (count only)** — number of ways to place n non-attacking queens.
```python
def total_n_queens(n):
    cols, diag1, diag2 = set(), set(), set()
    def backtrack(row):
        if row == n: return 1
        count = 0
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2: continue
            cols.add(col); diag1.add(row - col); diag2.add(row + col)
            count += backtrack(row + 1)
            cols.remove(col); diag1.remove(row - col); diag2.remove(row + col)
        return count
    return backtrack(0)
```
*Technique: `row - col` is constant along one diagonal direction and `row + col` along the other — two sets replace an O(n) diagonal-clash check with O(1).*

## Trees

**24. Maximum Depth of Binary Tree**
```python
def max_depth(root):
    if not root: return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

**25. Invert Binary Tree**
```python
def invert_tree(root):
    if not root: return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```

**26. Same Tree**
```python
def is_same_tree(p, q):
    if not p and not q: return True
    if not p or not q or p.val != q.val: return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
```

**27. Subtree of Another Tree**
```python
def is_subtree(root, sub_root):
    if not root: return False
    if is_same_tree(root, sub_root): return True
    return is_subtree(root.left, sub_root) or is_subtree(root.right, sub_root)
```

**28. Validate Binary Search Tree**
```python
def is_valid_bst(root, low=float("-inf"), high=float("inf")):
    if not root: return True
    if not (low < root.val < high): return False
    return is_valid_bst(root.left, low, root.val) and is_valid_bst(root.right, root.val, high)
```
*Technique: passing down a valid (low, high) RANGE, not just comparing to the immediate parent — a node can be greater than its direct parent but still invalid if it violates a constraint from further up the tree.*

**29. Binary Tree Level Order Traversal** (BFS)
```python
from collections import deque
def level_order(root):
    if not root: return []
    res, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        res.append(level)
    return res
```
*Technique: snapshotting `len(queue)` before the inner loop is what separates one level from the next — without it, you get one flat BFS order instead of a list-of-levels.*

**30. Lowest Common Ancestor of a BST**
```python
def lowest_common_ancestor(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val: root = root.left
        elif p.val > root.val and q.val > root.val: root = root.right
        else: return root
```
*Technique: uses the BST ORDERING property directly — no general tree-traversal needed, since "both targets are smaller" or "both are larger" tells you which side to descend into.*

**31. Binary Tree Right Side View** — the values visible looking from the right.
```python
from collections import deque
def right_side_view(root):
    if not root: return []
    res, queue = [], deque([root])
    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:
                res.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
    return res
```
*Technique: same level-order BFS as #29, just keeping the LAST node processed at each level — that's always the rightmost one.*

**32. Kth Smallest Element in a BST**
```python
def kth_smallest(root, k):
    stack = []
    node = root
    while stack or node:
        while node:
            stack.append(node); node = node.left
        node = stack.pop()
        k -= 1
        if k == 0: return node.val
        node = node.right
```
*Technique: an in-order traversal of a BST visits nodes in sorted order for free — the kth pop off the stack during that traversal is the kth smallest value, no separate sort needed.*

**33. Diameter of Binary Tree** — longest path between any two nodes (in edges).
```python
def diameter_of_binary_tree(root):
    best = 0
    def depth(node):
        nonlocal best
        if not node: return 0
        left, right = depth(node.left), depth(node.right)
        best = max(best, left + right)
        return 1 + max(left, right)
    depth(root)
    return best
```
*Technique: the diameter through any given node is `left_depth + right_depth`, computed as a side effect while computing depth anyway — no separate pass needed.*

**34. Balanced Binary Tree** — is every subtree's left/right depth within 1 of each other?
```python
def is_balanced(root):
    def check(node):
        if not node: return 0
        left = check(node.left)
        if left == -1: return -1
        right = check(node.right)
        if right == -1: return -1
        if abs(left - right) > 1: return -1
        return 1 + max(left, right)
    return check(root) != -1
```
*Technique: returning -1 as a sentinel "already found unbalanced" value lets the recursion short-circuit and bail out early, instead of computing full depth everywhere and checking balance in a slower second pass.*

**35. Path Sum** — does a root-to-leaf path summing exactly to target exist?
```python
def has_path_sum(root, target_sum):
    if not root: return False
    if not root.left and not root.right: return root.val == target_sum
    remaining = target_sum - root.val
    return has_path_sum(root.left, remaining) or has_path_sum(root.right, remaining)
```

## Graphs

**36. Number of Islands** — count connected groups of "1"s in a grid (DFS flood fill).
```python
def num_islands(grid):
    rows, cols = len(grid), len(grid[0])
    def dfs(r, c):
        if r < 0 or c < 0 or r >= rows or c >= cols or grid[r][c] != "1": return
        grid[r][c] = "0"
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]: dfs(r+dr, c+dc)
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1
                dfs(r, c)
    return count
```

**37. Clone Graph**
```python
def clone_graph(node):
    if not node: return None
    seen = {}
    def dfs(n):
        if n in seen: return seen[n]
        copy = Node(n.val)
        seen[n] = copy
        for nb in n.neighbors:
            copy.neighbors.append(dfs(nb))
        return copy
    return dfs(node)
```
*Technique: a `seen` map from original node to its clone — needed both to avoid infinite recursion on cycles and to make sure shared neighbors point to the SAME clone, not separate duplicate copies.*

**38. Course Schedule** — can all courses be finished given prerequisite pairs (cycle detection)?
```python
def can_finish(num_courses, prerequisites):
    graph = {i: [] for i in range(num_courses)}
    for course, pre in prerequisites: graph[course].append(pre)
    state = {}   # 0=visiting, 1=done
    def has_cycle(node):
        if state.get(node) == 0: return True
        if state.get(node) == 1: return False
        state[node] = 0
        for pre in graph[node]:
            if has_cycle(pre): return True
        state[node] = 1
        return False
    return not any(has_cycle(c) for c in range(num_courses))
```
*Technique: the 3-state marking (unvisited / currently-in-progress / fully-done) distinguishes "revisiting a node still on the current path" (a real cycle) from "revisiting a node already fully resolved elsewhere" (not a cycle) — a 2-state visited/unvisited check alone can't tell these apart.*

**39. Pacific Atlantic Water Flow** — cells from which water can reach both oceans.
```python
def pacific_atlantic(heights):
    if not heights: return []
    rows, cols = len(heights), len(heights[0])
    pacific, atlantic = set(), set()
    def dfs(r, c, visited, prev_height):
        if (r, c) in visited or r < 0 or c < 0 or r >= rows or c >= cols or heights[r][c] < prev_height:
            return
        visited.add((r, c))
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            dfs(r+dr, c+dc, visited, heights[r][c])
    for r in range(rows):
        dfs(r, 0, pacific, heights[r][0]); dfs(r, cols-1, atlantic, heights[r][cols-1])
    for c in range(cols):
        dfs(0, c, pacific, heights[0][c]); dfs(rows-1, c, atlantic, heights[rows-1][c])
    return list(pacific & atlantic)
```
*Technique: flow the search BACKWARD from each ocean's border inland (water can flow from a cell to a lower-or-equal neighbor, so searching backward means "can I reach this cell going uphill") — searching forward from every single cell would be far more expensive.*

**40. Graph Valid Tree** — n nodes, list of edges: do they form a valid tree (connected, no cycles)?
```python
def valid_tree(n, edges):
    if len(edges) != n - 1: return False   # a tree has exactly n-1 edges
    graph = {i: [] for i in range(n)}
    for a, b in edges:
        graph[a].append(b); graph[b].append(a)
    seen = set()
    def dfs(node):
        seen.add(node)
        for nb in graph[node]:
            if nb not in seen: dfs(nb)
    dfs(0)
    return len(seen) == n
```
*Technique: the edge-count check (`n-1`) is a fast necessary-but-not-sufficient filter; the DFS-reachability check confirms the remaining necessary condition (fully connected) — together they're sufficient, since a connected graph with exactly n-1 edges cannot contain a cycle.*

**41. Number of Connected Components in an Undirected Graph**
```python
def count_components(n, edges):
    graph = {i: [] for i in range(n)}
    for a, b in edges:
        graph[a].append(b); graph[b].append(a)
    seen = set()
    def dfs(node):
        seen.add(node)
        for nb in graph[node]:
            if nb not in seen: dfs(nb)
    count = 0
    for node in range(n):
        if node not in seen:
            count += 1
            dfs(node)
    return count
```

**42. Rotting Oranges** — minutes until all fresh oranges rot (multi-source BFS).
```python
from collections import deque
def oranges_rotting(grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2: queue.append((r, c, 0))
            elif grid[r][c] == 1: fresh += 1
    minutes = 0
    while queue:
        r, c, minutes = queue.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                queue.append((nr, nc, minutes + 1))
    return minutes if fresh == 0 else -1
```
*Technique: seeding the BFS queue with ALL initially-rotten oranges at once ("multi-source BFS"), not just one — the rot spreads from every source simultaneously, matching how the problem actually behaves minute by minute.*

**43. Word Ladder** — shortest transformation sequence length from beginWord to endWord, one letter at a time, each step a valid dictionary word (shortest path = BFS).
```python
from collections import deque
def ladder_length(begin_word, end_word, word_list):
    words = set(word_list)
    if end_word not in words: return 0
    queue = deque([(begin_word, 1)])
    while queue:
        word, steps = queue.popleft()
        if word == end_word: return steps
        for i in range(len(word)):
            for c in "abcdefghijklmnopqrstuvwxyz":
                candidate = word[:i] + c + word[i+1:]
                if candidate in words:
                    words.remove(candidate)
                    queue.append((candidate, steps + 1))
    return 0
```
*Technique: BFS specifically, not DFS — BFS explores in order of increasing distance, so the first time you reach `end_word` is guaranteed to be via the shortest path; DFS would find *a* path, not necessarily the shortest one.*

## Practice Q&A (Self-Test)

### #3 (House Robber II) solves a circular-array problem by calling the linear version twice. Why does that actually cover every case correctly?
Because the constraint is "no two adjacent houses, including the wraparound pair (first, last)," any valid selection can never include both the first and last house simultaneously. So the true optimal answer is always captured by either "the best selection that excludes the last house" or "the best selection that excludes the first house" — running the plain linear solution once per exclusion and taking the max covers both possibilities, and one of them is guaranteed to equal the true circular optimum.

### #17 (Subsets) builds `res += [s + [n] for s in res]` in a loop. Why does this generate every subset without any explicit recursion?
Each iteration doubles the result set: for every subset already built without the new number, it adds a matching subset WITH the new number appended. Starting from `[[]]`  and doing this once per input number produces every combination of "include or exclude" across all numbers — exactly `2^n` subsets — without needing to write the include/exclude choice as an explicit recursive branch.

### #28 (Validate BST) passes down a `(low, high)` range rather than just comparing each node to its immediate parent. What specific case does comparing-to-parent-only get wrong?
A node can be locally consistent with its immediate parent (e.g., greater than a left-side parent) while still violating a BST constraint from higher up the tree (e.g., it needs to be less than a grandparent it's nested under, on the grandparent's left side). Only tracking an accumulated valid range from the root down catches these non-local violations; comparing only to the direct parent misses them.

### #36 (Number of Islands) and #38 (Course Schedule) both use DFS, but #38 needs a 3-state visited marker while #36 only needs a 2-state one (visited or not). Why the difference?
#36 only needs to know "have I already counted this land cell" — once visited, it's permanently accounted for, and a simple visited-set suffices. #38 needs to distinguish a genuine cycle (revisiting a node that's still on the CURRENT recursive path) from simply reaching an already-fully-resolved node via a different path (not a cycle) — a plain 2-state check can't tell those apart, which is exactly why cycle detection specifically needs the "currently in progress" state as a third option.

### Why does #43 (Word Ladder) require BFS specifically, while #36/#37/#39 (Islands, Clone Graph, Pacific Atlantic) all use plain DFS just fine?
#43 is asking for the SHORTEST path length — BFS explores nodes in strict order of increasing distance from the start, so the first time it reaches the target is guaranteed to be via the shortest possible route. #36/#37/#39 only care about reachability/connectivity (which cells belong to the same island, which nodes are copies of each other, which cells can reach an ocean) — for a pure reachability question, the order of exploration doesn't matter, so DFS (simpler to write recursively) works exactly as well as BFS would.
