# Production ML Practice — Serving, Monitoring, and Drift

`mlops-practice.md` covers getting a model versioned and registered. This doc picks up after that: how the model actually serves real traffic, how you'd notice if it started quietly getting worse, and what to do about it.

## Cluster 1 — Choosing How a Model Serves Traffic

### Does a model actually need to answer instantly?

That's the first question, before anything else. Two options.

**Batch inference** runs the model on a large chunk of data on a schedule — nightly, hourly, whatever fits. The predictions get stored for later use. This is the right call for anything that doesn't need an answer this second: tomorrow's demand forecast, this week's churn-risk scores, a monthly credit-risk re-score.

**Real-time (online) inference** runs the model on demand, one request at a time. The caller waits for the response — a fraud check at checkout, a chatbot reply. This needs the model loaded and ready, or a fast-starting serverless setup. Every millisecond of latency here is felt directly by the user.

Picking batch when real-time isn't actually needed is one of the cheapest wins in production ML. Batch is simpler to build, cheaper to run, and easier to debug — you can inspect the whole batch output before anyone sees it.

### What does the real-time request path actually look like?

Say real-time serving is the right call. Here's what a request goes through:
```
request -> load balancer -> API layer (auth, validation, rate limiting)
        -> feature lookup (fetch any features not in the request itself)
        -> model inference (the actual forward pass)
        -> response
```

The "model inference" box is usually the smallest part of the actual code. Most of the engineering effort goes into everything around it:
- Validating the input didn't arrive malformed.
- Fetching any extra features the model needs — a user's historical average, say — fast enough to stay within the latency budget.
- Logging the request/prediction pair, so it can feed monitoring later.

Tools like **Triton Inference Server** or **vLLM** (for LLMs specifically) handle the inference box itself, with batching and GPU scheduling built in. See `core-technical-depth.md` and `bnsf-technical-visual.html` for how that batching actually works under the hood.

### Summary example

A checkout fraud check needs real-time serving. The decision blocks the user's transaction, so batch isn't an option.

The request flows through the full stack: auth, then a feature lookup for the user's historical spending average, then inference. The model's own forward pass is the smallest part of that whole path.

Every request/prediction pair gets logged on the way out. That log is what feeds the monitoring covered next.

---

## Cluster 2 — Drift, Detection, and Safe Rollout

### Why would accuracy degrade if nobody touched the code or the model?

A model is a snapshot of patterns in the data it was trained on. The real world keeps moving. The model doesn't — not unless it's retrained.

There are two distinct ways this shows up:

**Data drift (covariate shift)** — the *input* distribution changes. Customer demographics shift. A sensor gets recalibrated. A new product category appears. The true input-to-output relationship hasn't actually changed, just the inputs the model is now seeing.

**Concept drift** — the actual relationship between inputs and the target changes. Fraud tactics evolve to look different from the patterns the model learned. A pandemic changes what "normal" purchasing behavior even looks like.

Both cause the same symptom: accuracy quietly drops. But the fix is different. Data drift might just need more representative retraining data. Concept drift is worse — the old labels themselves no longer reflect the new reality, so more data alone doesn't fix it. The target relationship itself has to be relearned.

### How do you detect drift before a stakeholder notices?

Both kinds of drift look identical from the outside — accuracy just quietly drops. Here's how you catch either one before someone else does.

**Monitor the input distribution directly.** Track feature statistics over time — mean, standard deviation, or a distribution-distance measure like population stability index (PSI) or KL-divergence. Both of these boil down "how different does incoming data look from the training data" into one number you can alert on. This catches data drift specifically.

**Monitor the prediction distribution.** If a model that used to flag 2% of transactions as fraud suddenly flags 15%, something changed — in the data, or in the world — whether or not you know the ground truth yet.

**Monitor against ground truth once it's available.** For fraud, the confirmed-fraud label often arrives weeks later. When it does, compute real accuracy, precision, and recall on that lagged data, and compare it to training-time performance. This is what actually confirms concept drift — not just a symptom of it.

**Shadow deployment.** Run a new candidate model alongside the current production model on live traffic. Log both sets of predictions, but only ever serve the current model's output to users. This lets you compare the two offline, before the new model ever makes a real decision.

### Drift is detected and a new model is ready. Why not switch 100% of traffic to it right away?

**Canary deployment** answers this. Route a small slice of real traffic — say 5% — to the new model version, while everything else keeps going to the current one. Watch the key metrics on that slice. Only ramp up the percentage if it holds up.

This limits the blast radius of a bad model version to a small fraction of real traffic, instead of all of it. And it gives you a live comparison, instead of a leap of faith.

**Visual + memory hook — the traffic split ramping over time, watched at every step:**
```
Day 1     ████████████████████████░  95% v1 (current)  ░ 5% v2 (canary)
Day 2     ████████████████░░░░░░░░  80% v1             ░░░░░░░░ 20% v2   ← metrics held, ramp up
Day 3     ████░░░░░░░░░░░░░░░░░░░░  20% v1             ░░░░░░░░░░░░░░░░░░░░ 80% v2
Day 4     ░░░░░░░░░░░░░░░░░░░░░░░░  0% v1 (retired)    ████████████████████ 100% v2

                    ▲ at EVERY step: if v2's metrics look worse, ramp stops or reverses —
                      this is a dial, not a switch
```
Remember it as a dial you turn gradually, not a switch you flip once. "Full v1 to full v2" never happens in one step. It happens as a series of small, individually-reversible steps, and every step is gated on the previous step's metrics actually holding up.

A canary — the bird, in a coal mine — dies first, and small. That's the whole metaphor: 5% of traffic is the small, contained exposure that tells you something's wrong before it's exposed to everyone.

### Monitoring and canary rollouts are both in place. What still slips through, silently?

Four failure modes, and none of them trip an obvious alarm.

**Silent feature pipeline breakage.** An upstream data source changes a column's format, or starts sending nulls. The model doesn't error out — it just quietly gets fed garbage, and produces worse predictions with no crash to notice.

**Training/serving skew.** A feature computed at training time — say, "average order value over the last 30 days," computed in a batch job — gets computed slightly differently at serving time: live, with different rounding, or a different time window. The model performs great offline and worse in production, because it's technically seeing different features than it was trained on.

**Feedback loops.** A recommendation model's own past recommendations shape what users click. Those clicks become next round's training data, reinforcing whatever the model already favored, and drowning out anything it initially under-recommended. This is a slow, self-reinforcing drift the model can't see from its own metrics.

**Latency creep.** A model that was fast in testing slows down as real traffic volume or batch sizes grow. Eventually it breaches a latency SLA — a service-level agreement, the response-time ceiling you've promised callers — that nobody was watching until users complained.

### Eventually something forces a rollback. What has to be true beforehand for that to be fast?

Rolling back should mean one thing: point the registry or serving config at the previous registered model version, and redeploy. A fast, well-rehearsed action — not an emergency retraining job under pressure.

This is exactly why the model registry from `mlops-practice.md` matters operationally, not just organizationally. Rollback speed is bounded by one thing: whether "the previous version" is a clearly labeled, ready-to-serve artifact, or something someone has to go dig up.

### Summary example

A fraud model's prediction rate silently climbs from 2% flagged to 15% flagged. The prediction-distribution monitor catches this first.

Tracing it back: fraud tactics evolved, so this is concept drift, not data drift.

The fix is a retrained model, rolled out as a canary — 5%, then ramping up over several days. The same monitoring that caught the original drift also watches the new version for training/serving skew.

If the canary's metrics look worse at any step, rollback is simple: point the registry back at the still-running previous version. No emergency retrain under pressure.

---

## Cluster 3 — Securing and Serving the API Layer

Cluster 1's request path had one box labeled `API layer (auth, validation, rate limiting)`, then moved straight past it to feature lookup. This cluster opens that box.

It matters for AI-engineer roles specifically, because the API layer around a model is usually yours to ship. There's rarely a separate backend team wrapping your model for you.

It's also directly claimable ground: NaviDoc (`my-projects-portfolio.md`) already serves through FastAPI, and its access-control module — which filters retrieval candidates by the requesting user's permissions during the similarity search — is exactly the "who is asking, and what are they allowed to see" question this cluster is about, applied one layer deeper than the endpoint.

### Isn't "don't expose the port publicly" enough? Why does an inference endpoint need real auth?

No. "Don't expose the port" is a network boundary, not an identity boundary. It answers *can this packet reach me*, but never *who is asking, and what are they allowed to do*.

Three concrete reasons an inference endpoint specifically needs real auth:

**Cost control.** Unlike a CRUD endpoint, a single inference request can burn real GPU seconds. An unauthenticated endpoint means anyone who finds the URL can run expensive inference on your bill. That's a direct financial exposure, not a hypothetical one — and it scales with how good, and how expensive, your model is.

**Abuse and rate-limit evasion.** Rate limits have to be keyed to somebody. With no identity, the only key available is the source IP, and that's trivially rotated — cloud VMs, proxies. Without auth, rate limiting is barely enforceable at all. Auth is what makes a per-caller quota meaningful in the first place.

**Compliance.** The moment the endpoint touches real user or business data — clinical notes, transactions, PII — auditable per-caller identity stops being good practice. It becomes a requirement. HIPAA, SOC 2, and GDPR all assume you can say who accessed what, and when. "The port was firewalled" is not an access log.

The through-line: the network layer decides *reachability*. The API layer decides *identity and permission*. Only the second one can answer "who ran up this GPU bill," or "who read that record."

### API key, JWT, or OAuth2 — what actually distinguishes them?

This trio trips people up in interviews constantly, mostly because OAuth2 isn't even the same *kind* of thing as the other two.

**API key.** A long random string the caller sends on every request: `X-API-Key: sk_live_a3f...`. The server looks it up in a table to find out who it belongs to.

Good for: service-to-service calls, internal traffic, quick partner integrations — anywhere the caller is a machine, not a person.

The costs: an API key is opaque. It carries no information on its own — no user id, no expiry, no permissions — so every single request costs a database lookup. It has no built-in expiry, so a leaked key stays valid until someone notices and revokes it. And revocation is coarse: one key usually means all-or-nothing access, so "let this caller keep reading but stop it from writing" means minting and redistributing an entirely new key.

**JWT (JSON Web Token).** A signed, self-contained token. Three base64url-encoded segments joined by dots: `header.payload.signature`.

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyXzEyMyIsImV4cCI6MTczNTY4OTYwMCwic2NvcGUiOiJpbmZlcmVuY2U6cmVhZCJ9.mzAknCbx0eYoQjDW9xMBoPVoeFSzLFaePeCk_dWJ5UU
└──────────── header ────────────┘ └───────────────────────── payload ─────────────────────────┘ └───────────── signature ─────────────┘
       {"alg":"HS256",                  {"sub": "user_123",                                          HMAC-SHA256 over
        "typ":"JWT"}                     "exp": 1735689600,                                          header + "." + payload,
                                         "scope": "inference:read"}                                  using the server's secret
```

That payload is a real, decodable one. Base64url is *encoding*, not encryption — anyone holding the token can read the claims. The signature protects *integrity*, not secrecy. Never put anything confidential in a payload.

Three claims worth knowing by name:
- `sub` (**subject**) — who the token is about. `user_123` is the caller's identity. This is what your logs, quotas, and billing key off.
- `exp` (**expiry**) — a Unix timestamp after which the token must be rejected. This one is `2025-01-01T00:00:00Z`, so it's already expired. A token that leaks after its `exp` is worthless — which is exactly why short-lived tokens beat static keys.
- `scope` — what this caller is permitted to do. `inference:read` here. Permissions travel inside the token itself.

Here's why JWTs dominate stateless API auth at scale: the server can verify the token wasn't tampered with by recomputing the signature with its own secret (or verifying with its public key). No database lookup at all. Edit the payload — bump `exp`, add `scope: admin` — and the signature no longer matches. The attacker can't forge a new one without the key. So identity, expiry, and permissions all arrive with the request already verified.

The tradeoff is the mirror image of the API key's: a JWT is hard to revoke *early*, because nothing gets consulted at request time to ask "is this still valid?" That's why access tokens are kept short-lived — minutes — and paired with a longer-lived refresh token that *does* hit a revocable store.

**OAuth2.** Not a token format at all. It's a *delegation/authorization framework* — a protocol for how a user grants a third-party application limited access to their resources, without handing over their password.

"Log in with Google" is the everyday example. You authorize an app to read your Google profile. The app never sees your Google password, gets only the scopes you approved, and you can revoke it later from your Google account without changing that password.

The distinction people get wrong: OAuth2 is the flow. A JWT is often what that flow hands back, as the actual access token. They're not alternatives. They operate at different layers. "We use OAuth2" describes *how the token was obtained*. "We use JWTs" describes *what the token is*.

One more thing worth saying out loud in an interview: OAuth2 is about *authorization* — delegated access — not authentication, proving who you are. The layer that adds identity on top is **OIDC (OpenID Connect)**. That's what actually makes "log in with Google" a real *login*, and OIDC's ID token is, specifically, a JWT.

**Visual + memory hook — same request, three different things in the header:**
```
API key   →  X-API-Key: sk_live_a3f...     opaque; server must LOOK IT UP        "a coat-check ticket"
JWT       →  Authorization: Bearer eyJ...  self-describing; server VERIFIES it   "a signed passport"
OAuth2    →  (not a header — it's how you GOT the Bearer token above)            "the visa process"
```
A coat-check ticket means nothing without the cloakroom's ledger. A passport is readable and tamper-evident on its own. OAuth2 isn't a document at all — it's the process that issued one.

### With a JWT chosen, what actually happens at request time?

Four steps happen, in this order, before any GPU work is touched:

1. Pull the token out of the `Authorization: Bearer <token>` header.
2. **Verify the signature** with the server's secret (HS256) or public key (RS256). This is what proves the payload wasn't edited.
3. **Check `exp`** hasn't passed. An expired token gets rejected even if the signature is perfect.
4. **Check `scope`** actually permits *this* operation. A valid token for `inference:read` must not be able to trigger a fine-tune.

Steps 2 and 3 answer "who are you" — a failure there is a **401 Unauthorized**. Step 4 answers "you're known, but not allowed to do this" — that's a **403 Forbidden**. Returning 401 where you meant 403 is a small tell that gets noticed in review.

In FastAPI, this is a `Depends()` dependency. That means auth gets declared per-route instead of remembered per-handler — forgetting it is a visible omission in the function signature, not an invisible one buried in the body.

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

Trace a request through that endpoint, and here's what comes back: `200` for a valid, unexpired `inference:read` token. `401 token expired` once `exp` passes. `401 invalid token` for a tampered, malformed, or wrongly-signed token, or a missing header. `403 insufficient scope` for a perfectly valid token whose scope is, say, `jobs:write` instead.

Two details are easy to get wrong, and worth knowing why they're there:

**`algorithms=[...]` is not optional.** Leave out the allow-list, and you've built the classic JWT vulnerability: an attacker re-signs the token declaring `alg: none` — or downgrades RS256 to HS256 and signs with the public key as if it were the shared secret — and the library obligingly "verifies" it. Pinning the algorithm server-side closes that off.

**`exp` is checked by the library, not by you.** `jwt.decode()` validates `exp` automatically and raises `ExpiredSignatureError`. Hand-rolling `if claims["exp"] < time.time()` is both redundant and easy to get wrong — a missing claim, clock skew, that kind of thing.

### A valid token gets through. Why does rate limiting still matter?

Authentication answers *who*, never *how much*. A perfectly valid token can still be attached to a client that is:

- **Compromised.** The token was stolen and is being replayed by someone else. It stays valid until `exp`, per the revocation tradeoff above.
- **Buggy.** A retry loop with no backoff, hammering your endpoint 500 times a second in complete good faith. In practice, this is the more common cause of an outage.
- **Abusive within its rights.** A legitimate customer on a cheap tier discovers your GPU endpoint is the fastest way to run their own batch job.

Two mechanisms worth being able to contrast:

**Token bucket.** Each caller has a bucket that refills at a steady rate — say 10 tokens per second, capacity 100. A request spends one token. An empty bucket means rejection. Because the bucket can sit full, it *allows bursts* up to capacity while still capping the long-run average. It's cheap: one counter and one timestamp per caller.

**Sliding window.** Count the requests actually made in the trailing N seconds, and reject past the limit. This is stricter and smoother — no burst allowance — but more expensive, since it needs per-request timestamps, not just a counter.

Token bucket is usually the right default for inference APIs. Real clients are bursty, and the burst allowance absorbs that without punishing them, while the refill rate still bounds sustained GPU cost.

**Visual + memory hook — where each check sits, and what it costs to reach it:**
```
request
  │
  ├─▶ rate limit (per IP / per API key)   ~microseconds, one counter   ← CHEAPEST, so FIRST
  ├─▶ verify JWT signature + exp + scope  ~sub-millisecond crypto
  ├─▶ validate + feature lookup           ~milliseconds, hits a store
  └─▶ MODEL INFERENCE                     ~GPU seconds, $$$            ← MOST EXPENSIVE, so LAST
```
Order the gate by what it costs to get past it — reject at the cheapest layer that can say no. A volumetric flood should be dropped by a counter increment, not after you've already done crypto, hit the feature store, and warmed a GPU.

The practical wrinkle: the *tightest* per-user quota needs the identity that only auth provides. So real systems do both — a coarse pre-auth limit keyed on IP or raw API key, to survive floods, and a fine per-`sub` quota after the token is verified.

### What happens when the operation genuinely takes minutes or hours, not milliseconds?

Everything above assumes a synchronous request/response. That breaks down once a job actually takes a long time.

Holding an HTTP connection open for a 40-minute fine-tuning job isn't an option. Load balancers and proxies time out — typically 30 to 120 seconds. A dropped connection loses the result entirely. And a held connection pins server resources doing nothing the whole time. Polling (`GET /jobs/123` every 5 seconds) technically works, but it's wasteful and laggy — most polls just return "still running."

**Webhooks** are the standard answer. The caller registers a callback URL up front. Your service POSTs the result to that URL when the job finishes. The synchronous call returns immediately with a job id (`202 Accepted`), the connection closes, and the result gets *pushed* later. It inverts who calls whom — your service becomes the client, and the caller becomes the server.

That inversion creates a security problem specific to webhooks: the receiver now has an endpoint that anyone on the internet can POST to. Nothing about a raw POST proves it came from you, rather than from someone who guessed the URL and forged a `{"status": "succeeded", "credits_refunded": 5000}` body.

The fix is **HMAC signature verification**. At registration, both sides hold a shared secret. The sender hashes the exact request body with that secret and puts the result in a header. The receiver recomputes the same hash over the body it received, and compares. Matching signatures prove two things: *authenticity* (only a holder of the secret could have produced it) and *integrity* (any byte changed in transit produces a different hash).

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

Three things make or break this in practice:

**Use `hmac.compare_digest`, not `==`.** A normal string comparison short-circuits at the first differing byte, so its *timing* leaks how many leading bytes were right. Given enough attempts, that's enough to reconstruct a valid signature byte by byte. `compare_digest` takes the same time regardless of where the mismatch is.

**Sign and verify the raw bytes, not a re-serialized dict.** `json.dumps(json.loads(body))` can reorder keys or change whitespace, which changes the hash — and breaks verification for a payload that was never actually tampered with. On the receiving FastAPI side, that means reading `await request.body()`, not the parsed Pydantic model.

**Include a timestamp in what you sign, and reject old ones.** A bare signature is still perfectly replayable. An attacker who captures one valid signed delivery can POST that same body and signature forever. Signing `f"{timestamp}.{body}"` and rejecting deliveries older than a few minutes closes the replay window. It's the same category of problem as `exp` on a JWT, just one layer down.

Two more things matter here, even without code. The receiver must reply fast — `200` immediately, do the real work after — or your sender will time out and retry. And the receiver has to be **idempotent**: webhook delivery is at-least-once, so the same `job_id` will occasionally arrive twice, and the receiver has to treat the duplicate as a no-op instead of double-charging someone.

### Summary example

An AI engineer ships a fine-tuning-as-a-service endpoint. Here's how the whole cluster comes together.

1. Auth is non-negotiable. Each job burns real GPU hours on the company's bill, and the training data is customer-owned.
2. Internal services calling it get a static API key. Customer-facing traffic gets short-lived JWTs, obtained through an OAuth2 flow — so a customer's own dashboard can trigger jobs without ever handling the customer's password.
3. Each request arrives as `Authorization: Bearer eyJ...`. The FastAPI `Depends()` dependency verifies the signature, rejects anything past `exp` with a 401, and rejects a caller holding only `inference:read` from hitting `POST /v1/fine-tune` with a 403.
4. Rate limiting runs *before* that crypto, keyed on API key or IP — so a retry storm from one buggy client gets dropped by a counter, not by a warmed GPU. A tighter per-`sub` token-bucket quota applies once identity is known.
5. The job takes 40 minutes, so the call returns `202 Accepted` with a `job_id` instead of blocking.
6. When training completes, the service POSTs the result to the customer's registered callback URL, with an HMAC-SHA256 signature over the raw body plus a timestamp. The customer verifies it with `hmac.compare_digest` before trusting a single field.

Every one of those requests is logged against its `sub`. That's what makes the GPU bill attributable, and the access log auditable.

### Where people trip up

**A JWT-authenticated endpoint accepts a token that's obviously been edited.** This means the `algorithms=` allow-list was left out of `jwt.decode()`. Without it, the library honors whatever `alg` the token itself declares — including `alg: none` — and "verifies" a completely unsigned payload. Pin `algorithms=["HS256"]` (or `["RS256"]`) server-side. It's a one-line fix for what would otherwise be a total auth bypass.

**Revoking a compromised JWT "doesn't take effect."** That's expected — statelessness is the whole point of the design, since nothing gets consulted at request time. Don't fix this by adding a lookup to every request; that throws away the reason you chose JWTs in the first place. Instead, use short `exp` lifetimes (minutes) plus a revocable refresh token. The blast radius of a stolen access token is then bounded by the clock, not by a database.

**Rate limiting sits after auth, and a flood still takes the service down.** Every junk request is still paying for signature verification and a feature lookup before being told no. Put a coarse limit keyed on IP or raw API key *in front of* auth, and keep the fine per-`sub` quota behind it. The cheap check has to be first.

**Webhook signature verification fails on payloads that were never tampered with.** The receiver re-serialized the JSON before hashing it. Key order and whitespace change the bytes, and therefore the digest. Hash the raw request body exactly as received (`await request.body()` in FastAPI), never `json.dumps(parsed)`.

**An attacker can replay a captured webhook delivery indefinitely.** Only the body was signed — no timestamp. Sign `timestamp + body`, and reject deliveries older than a few minutes. A valid signature alone only proves the payload is *authentic*. It never proves it's *current*.

**A caller reports being charged twice for one completed job.** The webhook receiver assumed exactly-once delivery. Webhook senders retry on timeout or a non-2xx response, so delivery is at-least-once by design. The receiver has to dedupe on `job_id` (or an idempotency key), and return `200` fast while doing the real work asynchronously.

**Someone says "we use OAuth2 instead of JWTs."** Worth gently correcting — that's a category error. OAuth2 is the framework for *obtaining* a token via delegated authorization. A JWT is a *format* that token often takes. They compose; they don't compete. And OAuth2 alone is authorization, not authentication — OIDC is the layer that adds identity.

---

## Cluster 4 — When the LLM Call Itself Fails: Timeouts, Fallbacks, and Degrading Gracefully

Cluster 3 assumes your own service is the thing that can fail. Once your service's job is "call an LLM API and return the result," you've added a dependency you don't control — one with its own latency spikes, rate limits, and outages. "The request failed" stops being an edge case and becomes a Tuesday. The failure modes here are specific to LLM calls — long tail latency, token-based rate limits, a single provider going down — and the fixes are a different toolkit than a normal internal-service retry.

### An LLM call sometimes takes 60+ seconds instead of the usual 2. What do you do?

Set an explicit client-side timeout, well below your own service's deadline. If your API has to respond in 10 seconds, the LLM call gets maybe 6 or 7 seconds — not whatever the SDK defaults to.

Without this, one slow provider call ties up a request thread until *your* framework-level timeout fires. That's usually much later, and much less informative, than a clean, fast failure you control yourself.

### A timeout or a 429 comes back. Do you just retry right away?

No. Immediate retries into a rate-limited or overloaded endpoint make the problem worse, not better. A burst of clients all retrying at once creates a synchronized "thundering herd" that can keep an already-struggling endpoint down.

The fix is **exponential backoff with jitter**: wait `base * 2^attempt`, plus a small random offset, before each retry. Failed requests spread out over time instead of re-arriving in lockstep. Respect a `Retry-After` header if the provider sends one — it's telling you exactly how long to wait, not making a suggestion.

### Retries are exhausted and the primary provider is still down. What's next?

A **fallback model or provider**, configured in advance — not improvised at incident time. That can mean falling back from a large model to a smaller, faster one from the same provider (degraded quality, but still answering), or to a second provider entirely, if you've built against a provider-agnostic interface.

The fallback doesn't have to be as good as the primary. It has to be good enough to avoid returning nothing.

### Even the fallback fails. What does the user actually see?

Never a raw stack trace, and never a spinning indicator that eventually times out silently.

**Graceful degradation** means: return a clear, honest message ("this feature is temporarily unavailable, please try again shortly"), fall back to a cached previous answer if one exists and is still reasonable, or drop to a non-LLM path if one exists — keyword search instead of a semantic answer, say. The product keeps functioning in a reduced form, instead of appearing broken.

### Summary example

An AI engineer's chat feature calls Claude with an 8-second client timeout, inside a service that has a 12-second SLA.

1. On a timeout or a 429, it retries twice with exponential backoff and jitter — roughly 0.5 seconds, then 2 seconds, respecting any `Retry-After` header.
2. If both retries fail, it falls back to a smaller, faster model configured for exactly this situation. That answers with lower latency and slightly lower quality, rather than not answering at all.
3. If that also fails — a full provider outage — the endpoint returns a plain-language "temporarily unavailable" response instead of hanging until the client gives up. It logs the failure with enough context — provider, model, attempt count, latency — to page on-call if the rate crosses a threshold.

The user either gets a good answer, an acceptable answer, or a clear "not right now." Never a spinner that dies silently.

### Where people trip up

**Retries make an outage worse, not better.** They fired immediately and in lockstep. Every client hitting the same failing endpoint at the same instant recreates the load spike that caused the failure in the first place. Exponential backoff with jitter spreads retries out so they don't all land together.

**A "resilient" service still hangs for 60+ seconds under a slow provider.** The client-side timeout was never set. The SDK's default — or no timeout at all — leaves you waiting on someone else's infrastructure, instead of failing fast on your own terms.

**A fallback model was "configured" but never actually got traffic during the one outage that mattered.** It was never tested. A fallback path that only runs during a real incident is a fallback path you're testing for the first time in production. Exercise it deliberately — chaos testing, a feature flag that forces the fallback — before you actually need it.

**Users see a raw error or an infinite spinner during a provider outage.** This is a degradation design gap, not a bug in any one line of code. Decide in advance what "acceptable but not perfect" looks like — a cached answer, a smaller model, a plain-language unavailability message — rather than letting the failure mode be whatever falls out of unhandled exceptions.

---

## Practice Q&A (Self-Test)

**Q1. A fraud-detection model needs an answer in under 200ms at checkout. Batch or real-time serving?**
A: Real-time. The decision has to happen synchronously, inside the user's checkout flow. A nightly batch job couldn't possibly influence a decision that needs to happen in the current request.

**Q2. Model accuracy drops even though nothing about the model or its code changed. What are the two categories of explanation, and how do you tell them apart operationally?**
A: Data drift — the input distribution shifted — versus concept drift — the input-to-output relationship itself changed. Here's how to tell them apart: if predictions on fresh inputs still match ground truth well under the old relationship, but the inputs themselves look statistically different from training, that's data drift. If even a retrain on similarly-distributed recent inputs still underperforms, because the true labels have shifted, that's concept drift.

**Q3. Offline evaluation on held-out data looked great, but the model performs noticeably worse in production. Feature values themselves look reasonable. What's a likely cause?**
A: Training/serving skew. The same named feature gets computed differently at training time — a batch job with a clean 30-day window, say — versus serving time, a live computation with a slightly different window or rounding. The model is technically fed different feature values than it was trained on, even though nothing looks obviously wrong.

**Q4. Why deploy a new model version as a canary instead of switching 100% of traffic to it immediately?**
A: A canary limits how much real traffic is exposed to an unproven model version. If it's actually worse, the damage stays contained to a small slice, and you still get a live, real-traffic comparison against the current version — instead of finding out it's bad only after every user is already affected.

**Q5. Why does model-registry discipline (from `mlops-practice.md`) matter specifically for rollback speed?**
A: Rollback is only fast if "the previous production model" is an unambiguous, already-packaged, ready-to-serve artifact you can point traffic back to immediately. Without a registry tracking exact versions and their deployment history, "roll back" turns into first figuring out which file was actually running before — exactly the wrong time to be doing archaeology.

**Q6. The inference endpoint is already behind a firewall and not on the public internet. Why still put auth on it?**
A: A firewall decides reachability. Auth decides identity and permission, and only the second one survives contact with the real risks: someone — internal, or via a compromised host — running expensive GPU inference on your bill with no way to attribute the cost; rate limits being unenforceable because there's no per-caller key to count against; and, for anything touching user or business data, a compliance requirement to produce a per-caller access log. "The port was firewalled" is not an access log.

**Q7. What is a JWT actually made of, and why does its structure make it popular for stateless API auth?**
A: Three base64url segments joined by dots — `header.payload.signature`. The payload carries claims like `{"sub": "user_123", "exp": 1735689600, "scope": "inference:read"}`: who the caller is, when the token stops being valid, and what it's allowed to do. The signature is computed over header plus payload with the server's key, so the server can confirm nothing was tampered with just by recomputing it — no database lookup per request, which is exactly why it scales. One note: base64url is encoding, not encryption. Anyone holding the token can read the claims, so nothing confidential goes in the payload.

**Q8. Is OAuth2 an alternative to JWTs? (Common trip-up.)**
A: No — they're different layers, not competing options. OAuth2 is a delegation/authorization framework: a protocol by which a user grants a third-party app limited, revocable access to their resources without sharing their password ("log in with Google"). A JWT is a token format, and it's very often what an OAuth2 flow hands back as the access token. "We use OAuth2" describes how the token was obtained. "We use JWTs" describes what the token is. One more distinction: OAuth2 by itself is authorization, not authentication. OIDC is the layer built on top that adds identity, and its ID token is a JWT.

**Q9. A request arrives with a valid, correctly-signed, unexpired JWT — but for an operation the caller shouldn't be able to trigger. What does the API layer return, and why does the status code matter?**
A: `403 Forbidden`, from a `scope` claim check — not `401 Unauthorized`. 401 means "I don't know who you are" — a bad or expired token. 403 means "I know exactly who you are, and you're not allowed to do this." Signature and `exp` failures are 401s. Scope failures are 403s. Collapsing the two is a small tell that gets noticed in review, and it also breaks client-side error handling — a 401 tells a client to go refresh its token, which won't help at all if the real problem was insufficient scope.

**Q10. One line in `jwt.decode()` is the difference between real verification and a total auth bypass. Which, and what's the attack?**
A: The `algorithms=["HS256"]` allow-list. Omit it, and the library honors whatever algorithm is declared inside the token itself. An attacker sets `alg: none` and submits an unsigned payload that gets happily "verified" — or downgrades RS256 to HS256 and signs with your public key as though it were a shared secret. Pinning the accepted algorithm server-side closes both attacks.

**Q11. Traffic is fully authenticated. Why is rate limiting still needed, and where in the request path should it sit?**
A: Auth answers who, never how much. A valid token can be attached to a stolen client, a buggy retry loop with no backoff, or a legitimate customer abusing a cheap tier — and a JWT stays valid until `exp` even after it's known to be compromised. Place it by cost: a coarse limit keyed on IP or raw API key goes before signature verification and inference, so volumetric abuse gets rejected by a counter increment instead of after crypto, a feature lookup, and GPU time. A tighter per-`sub` quota then runs after auth, since that's the first point a real per-caller identity exists. Token bucket — refills at a fixed rate, allows bursts up to capacity, one counter per caller — is the usual default for bursty inference clients. Sliding window is stricter and smoother, but needs per-request timestamps.

**Q12. A fine-tuning job takes 40 minutes. Why can't the caller just wait on the HTTP response, and what's the standard alternative?**
A: Load balancers and proxies time out well before that — typically 30 to 120 seconds. A dropped connection loses the result outright, and a held connection pins resources doing nothing. The standard answer is a webhook: the caller registers a callback URL, the initial request returns `202 Accepted` with a `job_id` and closes, and the service POSTs the result to that URL on completion — pushed, not polled.

**Q13. What's the security concern unique to webhooks, and how is it solved?**
A: The receiver is now exposing an endpoint anyone on the internet can POST to, and a raw POST carries no proof it came from you rather than from someone who guessed the URL and forged a favorable payload. It's solved with HMAC signature verification: sender and receiver share a secret at registration, the sender hashes the exact raw body — plus a timestamp — with it and sends the digest in a header, and the receiver recomputes and compares. Three details do the actual work: compare with `hmac.compare_digest`, not `==`, since a short-circuiting comparison leaks, via timing, how many leading bytes were correct; hash the raw received bytes, never a re-serialized dict, since key order and whitespace change the digest; and include a timestamp with a freshness window, or a captured valid delivery stays replayable forever.
