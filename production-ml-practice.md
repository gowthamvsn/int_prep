# Production ML Practice — Serving, Monitoring, and Drift

`mlops-practice.md` gets a model versioned and registered. This doc is what happens after: how it actually serves real traffic, how you'd notice if it started quietly getting worse, and what to do about it — in plain language.

## Cluster 1 — Choosing How a Model Serves Traffic

### 1. Before anything else, how do you decide whether a model even NEEDS to answer instantly?
- **Batch inference** — run the model on a large chunk of data on a schedule (nightly, hourly) and store the predictions for later use. Right for anything that doesn't need an answer *this second* — tomorrow's demand forecast, this week's churn-risk scores, monthly credit-risk re-scoring.
- **Real-time (online) inference** — the model runs on-demand, per request, and the caller waits for the response (a fraud check at checkout, a chatbot reply). Needs the model loaded and ready (or a fast-starting serverless setup), and every millisecond of latency is directly felt by the user.

Picking batch when real-time isn't actually needed is one of the cheapest wins in production ML — batch is simpler to build, cheaper to run, and easier to debug (you can inspect the whole batch output before anyone sees it).

### 2. Given real-time serving is the right call (question 1), what does the actual request path look like, and where does the model itself fit in?
```
request -> load balancer -> API layer (auth, validation, rate limiting)
        -> feature lookup (fetch any features not in the request itself)
        -> model inference (the actual forward pass)
        -> response
```
The "model inference" box is usually the smallest part of the actual code — most of the engineering is around it: validating the input didn't arrive malformed, fetching any additional features the model needs (a user's historical average, say) fast enough to stay within latency budget, and logging the request/prediction pair for later monitoring (Cluster 2). Tools like **Triton Inference Server** or **vLLM** (for LLMs specifically) handle the inference box itself, with batching and GPU scheduling built in — see `core-technical-depth.md` and `bnsf-technical-visual.html` for how that batching actually works under the hood.

### Summary example
A checkout fraud check needs real-time serving (question 1) since the decision blocks the user's transaction; the request flows through the full stack (question 2) — auth, feature lookup for the user's historical spending average, then inference — with the model's own forward pass being the smallest part of that path, and every request/prediction pair logged on the way out specifically to feed the monitoring discussed next.

---

## Cluster 2 — Drift, Detection, and Safe Rollout

### 1. Given a model is now serving live traffic (Cluster 1), why would its accuracy degrade even if NOBODY touched the code or the model file?
A model is a snapshot of patterns in the data it was trained on. The real world keeps moving; the model doesn't, unless retrained. Two distinct kinds:
- **Data drift (covariate shift)** — the *input* distribution changes (customer demographics shift, a sensor gets recalibrated, a new product category appears) even if the true input→output relationship hasn't changed.
- **Concept drift** — the actual relationship between inputs and the target changes (fraud tactics evolve to look different than the patterns the model learned; a pandemic changes what "normal" purchasing behavior looks like).

Both cause the same symptom — accuracy quietly drops — but the fix differs: data drift might just need more representative retraining data; concept drift means the *old labels themselves* no longer reflect the new reality, so more data alone doesn't fix it — the target relationship has to be relearned.

### 2. Given both kinds of drift produce the SAME symptom (quietly dropping accuracy), how do you actually detect either one before a stakeholder notices?
- **Monitor the input distribution directly** — track feature statistics (mean, std, or a distance measure like population stability index / PSI, or KL-divergence) over time, and alert when incoming traffic looks meaningfully different from the training distribution — this catches data drift specifically.
- **Monitor prediction distribution** — if a model that used to flag 2% of transactions as fraud suddenly flags 15%, something changed upstream (in the data or the world), whether or not you know the ground truth yet.
- **Monitor against ground truth once it's available** — for fraud, the confirmed-fraud label often arrives weeks later; when it does, compute real accuracy/precision/recall on that lagged data and compare to training-time performance — this is what actually confirms concept drift, not just a symptom of it.
- **Shadow deployment** — run a new candidate model alongside the current production model on live traffic, log both predictions, but only serve the current model's output to users. Compare offline before ever letting the new model actually make decisions.

### 3. Given drift is detected and a new candidate model needs to replace the old one, why not just switch 100% of traffic over immediately?
**Canary deployment** — route a small slice of real traffic (say 5%) to the new model version while the rest keeps going to the current one, watch key metrics on that slice, and only ramp up the percentage if it holds up. This limits the blast radius of a bad model version to a small fraction of real traffic instead of 100% of it, and gives you a live comparison instead of a leap of faith.

**Visual + memory hook — the traffic split ramping over time, watched at every step:**
```
Day 1     ████████████████████████░  95% v1 (current)  ░ 5% v2 (canary)
Day 2     ████████████████░░░░░░░░  80% v1             ░░░░░░░░ 20% v2   ← metrics held, ramp up
Day 3     ████░░░░░░░░░░░░░░░░░░░░  20% v1             ░░░░░░░░░░░░░░░░░░░░ 80% v2
Day 4     ░░░░░░░░░░░░░░░░░░░░░░░░  0% v1 (retired)    ████████████████████ 100% v2

                    ▲ at EVERY step: if v2's metrics look worse, ramp stops or reverses —
                      this is a dial, not a switch
```
**Remember it as a dial you turn gradually, not a switch you flip once** — the whole point is that "100% v1 → 100% v2" never happens in one step; it happens in a series of small, individually-reversible steps, each one gated on the previous step's metrics actually holding up. A "canary" (the bird, in a coal mine) dies first and small — that's the entire metaphor: 5% of traffic is the small, contained exposure that tells you something's wrong before it's exposed to everyone.

### 4. Given monitoring (question 2) and canary rollouts (question 3) are both in place, what specific failures still slip through BOTH, silently?
- **Silent feature pipeline breakage** — an upstream data source changes a column's format or starts sending nulls; the model doesn't error, it just quietly gets fed garbage and produces worse predictions with no crash to notice.
- **Training/serving skew** — the feature computed at training time (say, "average order value over the last 30 days," computed in a batch job) is computed slightly differently at serving time (computed live, different rounding or a different time window) — model performs great offline, worse in production, because it's technically seeing different features than it was trained on.
- **Feedback loops** — a recommendation model's own past recommendations shape what users click, which becomes the next round's training data, reinforcing whatever the model already favored (and drowning out anything it initially under-recommended) — a slow, self-reinforcing drift the model can't see from its own metrics.
- **Latency creep** — a model that was fast in testing slows down as real traffic volume/batch sizes grow, eventually breaching a latency SLA nobody was watching until users complained.

### 5. Given any of question 4's failures (or question 1's drift) eventually forces a decision to revert, what actually has to be true BEFORE that moment for rollback to be fast?
Rolling back should mean pointing the registry/serving config at the previous registered model version and redeploying — a fast, well-rehearsed action, not an emergency retraining job under pressure. This is exactly why the model registry from `mlops-practice.md` matters operationally, not just organizationally: rollback speed is bounded by whether "the previous version" is a clearly labeled, ready-to-serve artifact or something someone has to go dig up.

### Summary example
A fraud model's prediction rate silently climbs from 2% to 15% flagged (question 2's prediction-distribution monitor catches this) — traced back to concept drift as fraud tactics evolved (question 1). The fix is retrained and rolled out as a 5%-then-ramping canary (question 3), while the same monitoring that caught the original drift also watches for training/serving skew in the new version (question 4) — and if the canary's metrics look worse at any step, rollback (question 5) is just pointing the registry back at the still-running previous version, not an emergency retrain under pressure.

---

## Cluster 3 — Securing and Serving the API Layer

Cluster 1's request path had one box labelled `API layer (auth, validation, rate limiting)` and then moved straight past it to feature lookup. This cluster opens that box. It matters for AI-engineer roles specifically because the API layer around a model is usually *yours* to ship — there's rarely a separate backend team who wraps your model for you. It's also directly claimable ground: NaviDoc (`my-projects-portfolio.md`) already serves through FastAPI, and its access-control module — which filters retrieval candidates by the requesting user's permissions *during* the similarity search — is exactly the "who is asking, and what are they allowed to see" question this cluster is about, applied one layer deeper than the endpoint.

### 1. Given Cluster 1's request path puts auth inside the API layer, why does an inference endpoint need auth at all — isn't "don't expose the port publicly" enough?
"Don't expose the port" is a network boundary, not an identity boundary: it answers *can this packet reach me* but never *who is asking and what are they allowed to do*. Three concrete reasons an inference endpoint in particular needs real auth:
- **Cost control.** Unlike a CRUD endpoint, a single inference request can consume real GPU seconds. An unauthenticated endpoint means anyone who finds the URL can run expensive inference **on your bill** — this is a direct financial exposure, not a hypothetical one, and it scales with how good/expensive your model is.
- **Abuse and rate-limit evasion.** Rate limits have to be keyed to *somebody*. With no identity, the only key available is source IP, which is trivially rotated (cloud VMs, proxies) — so without auth, rate limiting is barely enforceable at all. Auth is what makes a per-caller quota meaningful.
- **Compliance.** The moment the endpoint touches real user or business data — clinical notes, transactions, PII — auditable per-caller identity stops being good practice and becomes a *requirement* (HIPAA, SOC 2, GDPR all assume you can say who accessed what, when). "The port was firewalled" is not an access log.

The through-line: the network layer decides *reachability*, the API layer decides *identity and permission*, and only the second one can answer "who ran up this GPU bill" or "who read that record."

### 2. Given the API layer needs to establish caller identity (question 1), what are the actual options — and what really distinguishes an API key from a JWT from OAuth2?
This trio is a very common interview trip-up, mostly because OAuth2 isn't the same *kind* of thing as the other two.

**API key** — a long random string the caller sends on every request (`X-API-Key: sk_live_a3f...`). The server looks it up in a table to find out who it belongs to.
- *Good for:* service-to-service and internal calls, quick partner integrations, anything where the caller is a machine, not a person.
- *Costs:* it's opaque — the key itself carries **no** information (no user id, no expiry, no permissions), so **every single request costs a database lookup**. It has no built-in expiry, so a leaked key is valid until someone notices and revokes it. And revocation is coarse: one key usually means all-or-nothing access, so "let this caller keep reading but stop it writing" means minting and redistributing new keys.

**JWT (JSON Web Token)** — a *signed, self-contained* token. Three base64url-encoded segments joined by dots: `header.payload.signature`.
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyXzEyMyIsImV4cCI6MTczNTY4OTYwMCwic2NvcGUiOiJpbmZlcmVuY2U6cmVhZCJ9.mzAknCbx0eYoQjDW9xMBoPVoeFSzLFaePeCk_dWJ5UU
└──────────── header ────────────┘ └───────────────────────── payload ─────────────────────────┘ └───────────── signature ─────────────┘
       {"alg":"HS256",                  {"sub": "user_123",                                          HMAC-SHA256 over
        "typ":"JWT"}                     "exp": 1735689600,                                          header + "." + payload,
                                         "scope": "inference:read"}                                  using the server's secret
```
That payload is a real, decodable one (base64url is *encoding*, not encryption — anyone holding the token can read the claims; the signature protects **integrity**, not secrecy, so never put anything confidential in a payload). Its three claims:
- `sub` (**subject**) — who the token is about. `user_123` is the caller's identity; this is what your logs, quotas, and billing key off.
- `exp` (**expiry**) — a Unix timestamp after which the token must be rejected. This one is `2025-01-01T00:00:00Z`, so it is already expired — a token that leaks after its `exp` is worthless, which is exactly why short-lived tokens beat static keys.
- `scope` — what this caller is permitted to do (`inference:read`). Permissions travel *inside* the token.

The reason JWTs dominate stateless API auth at scale: the server can verify the token wasn't tampered with by **recomputing the signature with its own secret (or verifying with its public key) — no database lookup at all**. Any edit to the payload (bumping `exp`, adding `scope: admin`) invalidates the signature, and the attacker can't forge a new one without the key. So identity, expiry, and permissions all arrive with the request already verified. The tradeoff is the mirror image of the API key's: a JWT is hard to revoke *early*, because nothing is consulted at request time to ask "is this still valid?" — which is why access tokens are kept short-lived (minutes) and paired with a longer-lived refresh token that *does* hit a revocable store.

**OAuth2** — **not a token format at all.** It's a *delegation/authorization framework*: a protocol for how a user grants a **third-party application** limited access to their resources **without handing over their password**. "Log in with Google" is the everyday example — you authorize an app to read your Google profile; the app never sees your Google password, gets only the scopes you approved, and you can revoke it later from your Google account without changing that password.
- The distinction people get wrong: **OAuth2 is the flow; a JWT is often what that flow hands back** as the actual access token. They're not alternatives — they operate at different layers. "We use OAuth2" describes *how the token was obtained*; "we use JWTs" describes *what the token is*.
- Corollary worth saying out loud in an interview: OAuth2 is about **authorization** (delegated access), not authentication (proving who you are). The layer that adds identity on top of it is **OIDC (OpenID Connect)**, which is what actually makes "log in with Google" a *login* — and OIDC's ID token is, specifically, a JWT.

**Visual + memory hook — same request, three different things in the header:**
```
API key   →  X-API-Key: sk_live_a3f...     opaque; server must LOOK IT UP        "a coat-check ticket"
JWT       →  Authorization: Bearer eyJ...  self-describing; server VERIFIES it   "a signed passport"
OAuth2    →  (not a header — it's how you GOT the Bearer token above)            "the visa process"
```
**A coat-check ticket means nothing without the cloakroom's ledger; a passport is readable and tamper-evident on its own; OAuth2 isn't a document at all, it's the process that issued one.**

### 3. Given a JWT is the choice, what actually happens at request time inside Cluster 1's API-layer box?
Four steps, in this order, before any GPU work is touched:
1. Pull the token out of the `Authorization: Bearer <token>` header.
2. **Verify the signature** with the server's secret (HS256) or public key (RS256) — this is what proves the payload wasn't edited.
3. **Check `exp`** hasn't passed → expired tokens are rejected even if the signature is perfect.
4. **Check `scope`** actually permits *this* operation → a valid token for `inference:read` must not be able to trigger a fine-tune.

Steps 2 and 3 are "who are you" (**401 Unauthorized** on failure); step 4 is "you are known but not allowed to do this" (**403 Forbidden**). Returning 401 where you meant 403 is a small tell that gets noticed in review.

In FastAPI, this is a `Depends()` dependency, which means auth is declared per-route rather than remembered per-handler — forgetting it is a visible omission in the signature rather than an invisible one in the body:

```python
import jwt  # PyJWT
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

app = FastAPI()
bearer_scheme = HTTPBearer()          # parses the "Authorization: Bearer ..." header

JWT_SECRET = "load-me-from-a-secret-manager-never-from-source"
JWT_ALGORITHM = "HS256"


def require_scope(required_scope: str):
    """Build an auth dependency that verifies the JWT *and* checks one scope."""

    def _verify(
        creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    ) -> dict:
        token = creds.credentials     # the "<token>" half of "Bearer <token>"
        try:
            # verifies the signature AND the exp claim; raises on either failure
            claims = jwt.decode(
                token,
                JWT_SECRET,
                algorithms=[JWT_ALGORITHM],   # pin it — never trust the token's own alg
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token expired")
        except jwt.InvalidTokenError:         # bad signature, malformed, wrong alg
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")

        if required_scope not in claims.get("scope", "").split():
            # authenticated, but not allowed to do THIS -> 403, not 401
            raise HTTPException(status.HTTP_403_FORBIDDEN, "insufficient scope")
        return claims

    return _verify


@app.post("/v1/inference")
def run_inference(claims: dict = Depends(require_scope("inference:read"))):
    caller = claims["sub"]            # identity for logging, quotas, billing
    return {"caller": caller, "prediction": 0.87}
```
Traced through, that endpoint returns `200` for a valid unexpired `inference:read` token, `401 token expired` once `exp` passes, `401 invalid token` for a tampered/garbage/wrongly-signed token or a missing header, and `403 insufficient scope` for a perfectly valid token whose scope is (say) `jobs:write`.

Two details that are easy to get wrong and worth knowing why they're there:
- **`algorithms=[...]` is not optional.** Omitting the allow-list is the classic JWT vulnerability: an attacker re-signs the token declaring `alg: none` (or downgrades RS256→HS256 and signs with the public key as if it were the shared secret) and the library obligingly "verifies" it. Pinning the algorithm server-side closes it.
- **`exp` is checked by the library, not by you.** `jwt.decode()` validates `exp` automatically and raises `ExpiredSignatureError`; hand-rolling `if claims["exp"] < time.time()` is both redundant and easy to get wrong (missing claim, clock skew).

### 4. Given a valid token now gets through (question 3), why does the "rate limiting" part of Cluster 1's box still matter — isn't authenticated traffic trusted traffic?
No — authentication answers *who*, never *how much*. A perfectly valid token can still be attached to a client that is:
- **compromised** — the token was stolen and is being replayed by someone else, and it stays valid until `exp` (see question 2's revocation tradeoff);
- **buggy** — a retry loop with no backoff hammering your endpoint 500×/second in complete good faith, which in practice is the more common outage;
- **abusive within its rights** — a legitimate customer on a cheap tier discovering your GPU endpoint is the fastest way to run their own batch job.

Two mechanisms worth being able to contrast:
- **Token bucket** — each caller has a bucket that refills at a steady rate (say 10 tokens/sec, capacity 100); a request spends one token, and an empty bucket means rejection. Because the bucket can sit full, it **allows bursts** up to capacity while capping the long-run average. Cheap: one counter and one timestamp per caller.
- **Sliding window** — count the requests actually made in the trailing N seconds and reject past the limit. **Stricter and smoother** (no burst allowance), but more expensive: it needs per-request timestamps, not just a counter.

Token bucket is usually the right default for inference APIs: real clients are bursty, and the burst allowance absorbs that without punishing them, while the refill rate still bounds sustained GPU cost.

**Visual + memory hook — where each check sits, and what it costs to reach it:**
```
request
  │
  ├─▶ rate limit (per IP / per API key)   ~microseconds, one counter   ← CHEAPEST, so FIRST
  ├─▶ verify JWT signature + exp + scope  ~sub-millisecond crypto
  ├─▶ validate + feature lookup           ~milliseconds, hits a store
  └─▶ MODEL INFERENCE                     ~GPU seconds, $$$            ← MOST EXPENSIVE, so LAST
```
**Order the gate by what it costs to get past it — reject at the cheapest layer that can say no.** A volumetric flood should be dropped by a counter increment, not after you've done crypto, hit the feature store, and warmed a GPU. The practical wrinkle: the *tightest* per-user quota needs the identity that only auth provides, so real systems do both — a coarse pre-auth limit keyed on IP or raw API key to survive floods, and a fine per-`sub` quota after the token is verified.

### 5. Given everything above assumes a synchronous request/response (Cluster 1's path ends in `-> response`), what happens when the operation genuinely takes minutes or hours?
Holding an HTTP connection open for a 40-minute fine-tuning job is not an option: load balancers and proxies time out (typically 30–120s), a dropped connection loses the result entirely, and a held connection pins server resources doing nothing. Polling (`GET /jobs/123` every 5s) works but is wasteful and laggy — most polls return "still running."

**Webhooks** are the standard answer: **the caller registers a callback URL up front, and your service POSTs the result to that URL when the job finishes.** The synchronous call returns immediately with a job id (`202 Accepted`), the connection closes, and the result is *pushed* later. It inverts who calls whom — your service becomes the client, the caller becomes the server.

That inversion creates the security problem specific to webhooks: **the receiver has an endpoint that anyone on the internet can POST to.** Nothing about a raw POST proves it came from you rather than from someone who guessed the URL and forged a `{"status": "succeeded", "credits_refunded": 5000}` body. The fix is **HMAC signature verification**: at registration both sides hold a shared secret; the sender hashes the exact request body with that secret and puts the result in a header; the receiver recomputes the same hash over the body it received and compares. Matching signatures prove both *authenticity* (only a holder of the secret could produce it) and *integrity* (any byte changed in transit produces a different hash).

```python
import hashlib
import hmac
import json

WEBHOOK_SECRET = b"shared-secret-handed-to-the-receiver-at-registration"


def sign(raw_body: bytes, secret: bytes = WEBHOOK_SECRET) -> str:
    return "sha256=" + hmac.new(secret, raw_body, hashlib.sha256).hexdigest()


# --- sender side (your service, when the fine-tune job finishes) ---
body = json.dumps({"job_id": "ft_881", "status": "succeeded"}).encode()
headers = {"X-Signature-256": sign(body)}
# requests.post(callback_url, data=body, headers=headers)   # send the SAME bytes you signed


# --- receiver side ---
def verify(raw_body: bytes, header_sig: str, secret: bytes = WEBHOOK_SECRET) -> bool:
    expected = sign(raw_body, secret)
    return hmac.compare_digest(expected, header_sig)   # constant-time compare
```
Three things that make or break this in practice:
- **`hmac.compare_digest`, not `==`.** A normal string comparison short-circuits at the first differing byte, so its *timing* leaks how many leading bytes were right — enough, over many attempts, to reconstruct a valid signature byte by byte. `compare_digest` takes the same time regardless.
- **Sign and verify the raw bytes, not a re-serialized dict.** `json.dumps(json.loads(body))` can reorder keys or change whitespace, which changes the hash and breaks verification for a payload that was never actually tampered with. On the receiving FastAPI side that means `await request.body()`, not the parsed Pydantic model.
- **Include a timestamp in what you sign, and reject old ones.** A bare signature is still perfectly replayable — an attacker who captures one valid signed delivery can POST that same body+signature forever. Signing `f"{timestamp}.{body}"` and rejecting deliveries older than a few minutes closes the replay window. (Same category of problem as `exp` on a JWT, one layer down.)

Also: the receiver must reply fast (`200` immediately, do the work after) or your sender will time out and retry, and it must be **idempotent** — webhook delivery is at-least-once, so the same `job_id` will occasionally arrive twice, and the receiver has to treat the duplicate as a no-op rather than double-charging someone.

### Summary example
An AI engineer ships a fine-tuning-as-a-service endpoint. Auth is non-negotiable because each job burns real GPU hours on the company's bill and the training data is customer-owned (question 1). Internal services calling it get a static API key; customer-facing traffic gets short-lived JWTs, obtained through an OAuth2 flow so a customer's own dashboard can trigger jobs without ever handling the customer's password (question 2). Each request arrives as `Authorization: Bearer eyJ...`; the FastAPI `Depends()` dependency verifies the signature, rejects anything past `exp` with a 401, and rejects a caller holding only `inference:read` from hitting `POST /v1/fine-tune` with a 403 (question 3) — but rate limiting runs *before* that crypto, keyed on API key or IP, so a retry storm from one buggy client is dropped by a counter rather than by a warmed GPU, with a tighter per-`sub` token-bucket quota applied once identity is known (question 4). Because the job takes 40 minutes, the call returns `202 Accepted` with a `job_id` instead of blocking, and when training completes the service POSTs the result to the customer's registered callback URL with an HMAC-SHA256 signature over the raw body plus a timestamp — which the customer verifies with `hmac.compare_digest` before trusting a single field of it (question 5). Every one of those requests is logged against its `sub`, which is what makes the GPU bill attributable and the access log auditable.

### Common pitfalls
- **If a JWT-authenticated endpoint accepts a token that was obviously edited, it's because the `algorithms=` allow-list was omitted from `jwt.decode()`** — without it the library will honour the token's own `alg` header, including `alg: none`, and "verify" a completely unsigned payload; pinning `algorithms=["HS256"]` (or `["RS256"]`) server-side is the fix, and it's a one-line fix for a total auth bypass.
- **If revoking a compromised JWT "doesn't take effect," it's because nothing is consulted at request time — that statelessness is the whole point of the design** — the fix isn't to add a lookup to every request (that discards the reason you chose JWTs), it's short `exp` lifetimes (minutes) plus a revocable refresh token, so the blast radius of a stolen access token is bounded by the clock instead of by a database.
- **If rate limiting sits after auth and a flood still takes the service down, it's because every junk request is still paying for signature verification and a feature lookup before being told no** — put a coarse limit keyed on IP or raw API key *in front of* auth, and keep the fine per-`sub` quota behind it; the cheap check has to be the first one.
- **If webhook signature verification fails on payloads that were never tampered with, it's because the receiver re-serialized the JSON before hashing it** — key order and whitespace change the bytes and therefore the digest; hash the raw request body exactly as received (`await request.body()` in FastAPI), not `json.dumps(parsed)`.
- **If an attacker can replay a captured webhook delivery indefinitely, it's because only the body was signed, with no timestamp** — sign `timestamp + body` and reject deliveries older than a few minutes; a valid signature alone only proves the payload is *authentic*, never that it's *current*.
- **If a caller reports being charged twice for one completed job, it's because the webhook receiver assumed exactly-once delivery** — webhook senders retry on timeout or non-2xx, so delivery is at-least-once by design; the receiver has to dedupe on `job_id` (or an idempotency key) and return `200` fast, doing the real work asynchronously.
- **If someone says "we use OAuth2 instead of JWTs," it's a category error worth gently correcting** — OAuth2 is the framework for *obtaining* a token via delegated authorization; a JWT is a *format* that token often takes. They compose; they don't compete. (And OAuth2 alone is authorization, not authentication — OIDC is the layer that adds identity.)

## Practice Q&A (Self-Test)

### A fraud-detection model needs an answer in under 200ms at checkout. Batch or real-time serving?
Real-time — the decision has to happen synchronously within the user's checkout flow; a nightly batch job couldn't possibly influence a decision that needs to happen in the current request.

### Model accuracy drops even though nothing about the model or its code changed. What are the two categories of explanation, and how do you tell them apart operationally?
Data drift (the input distribution shifted) vs. concept drift (the input→output relationship itself changed). Distinguish them by comparing: if predictions on fresh inputs still match ground truth well when tested against the *old* relationship assumptions, but the inputs themselves look statistically different from training, that's data drift; if even a retrain on similarly-distributed recent inputs still underperforms because the true labels have shifted, that's concept drift.

### Offline evaluation on held-out data looked great, but the model performs noticeably worse in production. Feature values themselves look reasonable. What's a likely cause specific to production systems (not covered by normal train/test evaluation)?
Training/serving skew — the same named feature is computed differently at training time (e.g. a batch job with a clean 30-day window) versus serving time (a live computation with a subtly different window or rounding), so the model is technically fed different feature values than it was trained on, even though nothing looks obviously wrong.

### Why deploy a new model version as a canary instead of switching 100% of traffic to it immediately?
A canary limits how much real traffic is exposed to an unproven model version, so if it's actually worse, the damage is contained to a small slice while you still have a live, real-traffic comparison against the current version — rather than finding out it's bad only after every user is already affected.

### Why does model-registry discipline (from `mlops-practice.md`) matter specifically for rollback speed?
Rollback is only fast if "the previous production model" is an unambiguous, already-packaged, ready-to-serve artifact you can point traffic back to immediately. Without a registry tracking exact versions and their deployment history, "roll back" turns into first figuring out which file was actually running before — exactly the wrong time to be doing archaeology.

### The inference endpoint is already behind a firewall and not on the public internet. Why still put auth on it?
A firewall decides *reachability*; auth decides *identity and permission*, and only the second one survives contact with the three real risks: someone (internal or via a compromised host) running expensive GPU inference on your bill with no way to attribute the cost; rate limits being unenforceable because there's no per-caller key to count against; and, for anything touching user or business data, a compliance requirement to produce a per-caller access log. "The port was firewalled" is not an access log.

### What is a JWT actually made of, and why does its structure make it popular for stateless API auth?
Three base64url segments joined by dots — `header.payload.signature`. The payload carries claims like `{"sub": "user_123", "exp": 1735689600, "scope": "inference:read"}`: who the caller is, when the token stops being valid, and what it's allowed to do. The signature is computed over header+payload with the server's key, so the server can confirm nothing was tampered with **by recomputing it — no database lookup per request**, which is exactly why it scales. Note base64url is encoding, not encryption: anyone holding the token can read the claims, so nothing confidential goes in the payload.

### Is OAuth2 an alternative to JWTs? (Common trip-up.)
No — they're different layers, not competing options. OAuth2 is a **delegation/authorization framework**: a protocol by which a user grants a third-party app limited, revocable access to their resources without sharing their password ("log in with Google"). A JWT is a **token format** — and it's very often what an OAuth2 flow hands back as the access token. "We use OAuth2" describes how the token was obtained; "we use JWTs" describes what the token is. Bonus distinction: OAuth2 by itself is authorization, not authentication; OIDC is the layer built on top that adds identity (and its ID token is a JWT).

### A request arrives with a valid, correctly-signed, unexpired JWT — but for an operation the caller shouldn't be able to trigger. What does the API layer return, and why does the status code matter?
`403 Forbidden`, from a `scope` claim check — not `401 Unauthorized`. 401 means "I don't know who you are" (bad or expired token); 403 means "I know exactly who you are, and you're not allowed to do this." Signature and `exp` failures are 401s; scope failures are 403s. Collapsing the two is a small tell that gets noticed in review, and it makes client-side error handling wrong (a 401 tells a client to go refresh its token, which won't help at all if the real problem was insufficient scope).

### One line in `jwt.decode()` is the difference between real verification and a total auth bypass. Which, and what's the attack?
The `algorithms=["HS256"]` allow-list. Omit it and the library honours the algorithm declared *in the token itself* — so an attacker sets `alg: none` and submits an unsigned payload that gets happily "verified," or downgrades RS256 to HS256 and signs with your public key as though it were a shared secret. Pinning the accepted algorithm server-side closes both.

### Traffic is fully authenticated. Why is rate limiting still needed, and where in the request path should it sit?
Auth answers *who*, never *how much*. A valid token can be attached to a stolen client, a buggy retry loop with no backoff, or a legitimate customer abusing a cheap tier — and a JWT stays valid until `exp` even after it's known to be compromised. Place it by cost: a coarse limit keyed on IP or raw API key goes *before* signature verification and inference, so volumetric abuse is rejected by a counter increment instead of after crypto, a feature lookup, and GPU time; a tighter per-`sub` quota then runs after auth, since that's the first point a real per-caller identity exists. Token bucket (refills at a fixed rate, allows bursts up to capacity, one counter per caller) is the usual default for bursty inference clients; sliding window is stricter and smoother but needs per-request timestamps.

### A fine-tuning job takes 40 minutes. Why can't the caller just wait on the HTTP response, and what's the standard alternative?
Load balancers and proxies time out well before that (typically 30–120s), a dropped connection loses the result outright, and a held connection pins resources doing nothing. The standard answer is a **webhook**: the caller registers a callback URL, the initial request returns `202 Accepted` with a `job_id` and closes, and the service POSTs the result to that URL on completion — pushed, not polled.

### What's the security concern unique to webhooks, and how is it solved?
The receiver is now exposing an endpoint anyone on the internet can POST to, and a raw POST carries no proof it came from you rather than from someone who guessed the URL and forged a favorable payload. Solved with **HMAC signature verification**: sender and receiver share a secret at registration, the sender hashes the exact raw body (plus a timestamp) with it and sends the digest in a header, and the receiver recomputes and compares. Three details do the actual work: compare with `hmac.compare_digest`, not `==` (a short-circuiting comparison leaks, via timing, how many leading bytes were correct); hash the raw received bytes, never a re-serialized dict (key order and whitespace change the digest); and include a timestamp with a freshness window, or a captured valid delivery stays replayable forever.
