# Git Commands for Real Scenarios

Not a full git tutorial — a lookup by *situation*: "I'm in this mess, what's the command." Same one-line-command spirit as `module-cheatsheet.md`, for the tool everything else gets built and shared through.

### "I want to save my work in progress without committing it"
```bash
git stash                  # shelve all uncommitted changes, working directory becomes clean
git stash -u                # also stash untracked (new) files, not just modified ones
git stash pop               # reapply the most recent stash and remove it from the stash list
git stash list               # see everything currently stashed
```
Use when you need to switch branches with a dirty working directory but aren't ready to commit yet.

### "I want to start a new feature/experiment branch"
```bash
git checkout -b feature/new-retriever      # create and switch to a new branch, from current HEAD
git switch -c feature/new-retriever        # same thing, the newer/clearer syntax
```

### "I committed to the wrong branch"
```bash
git log --oneline -1        # note the commit hash
git reset --hard HEAD~1     # remove it from the current (wrong) branch — only safe if not pushed yet
git checkout correct-branch
git cherry-pick <commit-hash>   # replay that exact commit onto the correct branch
```

### "I want to undo my last commit but keep the changes staged"
```bash
git reset --soft HEAD~1
```
`--soft` keeps changes staged (ready to re-commit, e.g. with a fixed message). `--mixed` (the default if you omit a flag) keeps changes but unstages them. `--hard` throws the changes away entirely — the one to be careful with.

### "I already pushed a commit and need to undo it, without rewriting shared history"
```bash
git revert <commit-hash>
```
`revert` creates a *new* commit that undoes the changes from an earlier one — safe on shared branches, because nobody else's history gets rewritten. `reset` rewrites history and force-pushing it after others have pulled the old history causes real problems for them — that's the core reason `revert` is the right tool once something's pushed and shared, and `reset` is fine only for commits still local to you.

### "I have a merge conflict — what do I actually do?"

**Visual + memory hook — a conflict is just two arrows landing on the same line, and every command in this file is a different way of drawing the graph:**
```
                  A───B───C  (main)
                 /         \
  ...──X────────┘           M   ←  git merge: draws ONE new commit
                 \         /       joining both tips — conflict if
                  D───E───F        B..C and D..F both touched the
                  (feature-branch) same lines

                  A───B───C
                 /         \
  ...──X────────┘        D'──E'──F'  ←  git rebase: REPLAYS D,E,F
                                        as brand-new commits on top
                                        of C — same conflict risk,
                                        but a straight line after,
                                        not a diamond
```
**Remember it as:** every git command in this doc is either *adding a commit* (`commit`, `merge`), *moving a label* (`checkout`, `reset`, `branch`), or *rewriting commits that already exist* (`rebase`, `commit --amend`, `filter-repo`) — a conflict happens whenever two commits that touched the same lines get joined into one, whether that join is a merge's diamond shape or a rebase's replayed straight line. Once you can sketch the little dot-and-arrow graph above for whatever command you're about to run, "what will this actually do to my history" stops being a guess.

```bash
git merge feature-branch
# CONFLICT (content): Merge conflict in model.py
```
Open the conflicted file — git marks the disagreement inline:
```
<<<<<<< HEAD
learning_rate = 0.001
=======
learning_rate = 0.0005
>>>>>>> feature-branch
```
Everything between `<<<<<<< HEAD` and `=======` is *your current branch's* version; everything between `=======` and `>>>>>>> feature-branch` is the *incoming* version. Edit the file to keep whichever is correct (or a manual combination of both), delete the `<<<<<<<`/`=======`/`>>>>>>>` marker lines, then:
```bash
git add model.py
git commit          # completes the merge
```

### "I want a cleaner history than 15 tiny 'wip' commits before I merge"
```bash
git rebase -i HEAD~15
```
Opens an editor listing the last 15 commits; change `pick` to `squash` (or `s`) on the ones you want folded into the commit above them, save, and git combines them into fewer, more meaningful commits before you write the final combined message. **Never rebase commits that have already been pushed and pulled by someone else** — it rewrites commit hashes, and anyone who already has the old ones gets a diverged, hard-to-reconcile history.

### "I want to find which commit introduced a bug, across hundreds of commits"
```bash
git bisect start
git bisect bad                 # current commit is broken
git bisect good v1.2.0         # this earlier tag/commit was known good
# git checks out a commit halfway between — test it, then:
git bisect good                # or: git bisect bad
# repeat — git binary-searches down to the exact breaking commit
git bisect reset               # done, return to where you started
```
Turns "somewhere in these 300 commits" into roughly `log2(300) ≈ 9` manual tests.

### "I want to see exactly who changed this line and when"
```bash
git blame model.py             # every line, annotated with commit + author + date
git log -p --follow -- model.py    # full history of a file, including through renames
```

### "I accidentally committed a large model file / secret and need it gone from history"
```bash
git filter-repo --path secrets.env --invert-paths
```
(the modern replacement for the older `git filter-branch`/BFG approach). This rewrites history to remove the file from every commit, not just the latest one — required because simply deleting it in a new commit leaves it fully recoverable from history. After this, anyone with a clone needs to re-clone rather than pull, since history has changed. If the secret was ever pushed, treat it as compromised and rotate it regardless of the history cleanup.

### "This repo has model files/datasets that are too large for a normal git repo"
```bash
git lfs install
git lfs track "*.pt" "*.h5" "*.onnx"
git add .gitattributes
git add model.pt
git commit -m "add model checkpoint via LFS"
```
Git LFS stores large binaries outside the normal git object store and commits only a small pointer — the same underlying idea as DVC (`mlops-practice.md`), but as a general-purpose git extension rather than an ML-pipeline-specific tool. Use LFS for "one large file needs to live in git"; reach for DVC when you need dataset *versioning* tied to experiment tracking specifically.

### A sane `.gitignore` for an ML project
```
# environments
.venv*/
__pycache__/
*.pyc

# data & artifacts (version these with DVC/LFS instead, not raw git)
*.csv
*.parquet
data/
*.pt
*.h5
*.joblib
*.onnx

# secrets
.env
*.key

# logs & experiment output
mlruns/
wandb/
*.log
```
The recurring mistake this prevents: committing a multi-GB dataset or checkpoint directly, discovering the repo is now unusable to clone, and needing a history rewrite (see above) to fix it after the fact.

### "I want to combine two commits I already made into one"
```bash
git rebase -i HEAD~2
# mark the second one "squash" instead of "pick", save, edit the combined commit message
```

### "I want to compare two branches before merging"
```bash
git diff main..feature-branch          # what feature-branch has that main doesn't
git log main..feature-branch --oneline  # just the commit list, not the full diff
```

## Practice Q&A (Self-Test)

### You need to undo a commit that's already been pushed and pulled by two teammates. `reset` or `revert`, and why?
`revert` — it creates a new commit undoing the changes without rewriting existing history, so your teammates' already-pulled commits stay valid. `reset` rewrites history; force-pushing it after others have pulled would desync their local repos from the shared branch.

### Git reports a merge conflict. What do the `<<<<<<<`, `=======`, and `>>>>>>>` markers actually mean?
Everything between `<<<<<<< HEAD` and `=======` is your current branch's version of that section; everything between `=======` and `>>>>>>> other-branch` is the incoming branch's version. You edit the file to keep the correct content and delete all three marker lines before staging and committing.

### You suspect a bug was introduced somewhere in the last 200 commits but don't know which one. What's the fastest way to find it without checking all 200 manually?
`git bisect` — mark a known-good and known-bad commit, and it binary-searches (checking out a midpoint each time, waiting on your good/bad verdict) down to the exact commit in roughly log2(200) ≈ 8 tests instead of 200.

### Why is deleting a committed secret in a new commit not actually sufficient to remove it?
Git preserves full history — the file still exists in the earlier commit and is fully recoverable via `git log`/`git checkout <old-commit>` even after a later commit removes it. Actually removing it requires rewriting history (`git filter-repo`), and if it was ever pushed, the secret should be treated as compromised and rotated regardless.

### When would you reach for Git LFS versus DVC for a large file?
Git LFS for a general-purpose "this one large binary needs to live in the git repo" case (a model checkpoint, an asset file) with no other requirements. DVC when you specifically need dataset/model *versioning* integrated with an ML pipeline and experiment tracking — tracking which exact data version trained which exact model, not just storing a large file efficiently.
