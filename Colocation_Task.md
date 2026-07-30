# AI Engineer Screening Task: Multi-Model Co-location & Isolation

**Role:** AI Engineer, In Tandem AI
**Format:** Build-anywhere take-home on a **free GPU** (Google Colab or Kaggle T4, 16 GB). No paid GPU, no credits, no instance access. Self-reported results; we re-run the top submissions on the same free tier.
**Time guidance:** Aim for **3–4 hours.** Submit what you have.

---

## 1. What the task is

Serving **one** model fast is the easy part. The real job: our production box runs **multiple models on the same physical GPUs at once** — a latency-sensitive product feature competing with batch/internal workloads for the same hardware — and the engineering challenge is keeping them from wrecking each other.

This task is the miniature of that, shrunk to run on one free 16 GB T4. You will **co-locate three workloads on a single GPU**, **measure how much they interfere with the latency-sensitive one**, then **choose and apply an isolation strategy that holds a stated latency SLA under contention — and report how much useful work the GPU sustains while holding it.**

There is no single correct answer. We are measuring whether you can co-locate real workloads, measure interference honestly, and make a deliberate, defensible sharing decision under a hard constraint — not whether you hit one particular number.

> **Why a free T4 is the point.** Packing three services into 16 GB *is* the small-scale version of packing our fleet onto B-class cards. Three serving stacks on one GPU means **none can grab the default ~90% of HBM** — so you have to right-size the split before anything boots. If you hit OOM, that is the constraint doing its job: right-size it and document it. A submission that reports "all ran, no OOM" without ever tuning the memory split either didn't really co-locate or didn't run it.
>
> But note carefully: **fitting three services in memory is the setup, not the exercise.** Once they fit, they still contend for the card, and A's p99 is what pays for it. Memory is the entry fee; holding the SLA under that contention is the task (see §3.3) — diagnosing *which* resource is actually binding is part of what you are being asked to work out.

---

## 2. The fixed inputs (use these — do not go model-shopping)

These are chosen to fit **all three** models plus their KV-cache inside one 16 GB T4. Use them as given so submissions are comparable and your 3–4 hours go into the isolation work, not HuggingFace roulette.

| Slot | Model | Role |
|---|---|---|
| **Workload A — latency-sensitive** | `unsloth/Llama-3.2-1B-Instruct` | Interactive: single requests, must hold a **p99 SLA** (below) |
| **Workload B — noisy neighbor (batch)** | `Qwen/Qwen2.5-0.5B-Instruct` | Throughput/batch job that hammers the GPU continuously |
| **Workload C — embeddings** | `BAAI/bge-small-en-v1.5` | A third, very different workload serving a retrieval/embedding load |

> **All three are ungated** (no HuggingFace license gate / token needed) so the repo clones and runs clean for both you and us. Do **not** substitute the gated `meta-llama/...` mirror — it breaks the one-command clean run.

**All three must be genuinely co-resident on the one 16 GB T4 at the same time.** This is deliberate: two small models leave slack a generic memory split absorbs by accident; three workloads make the 16 GB genuinely tight, so the split has to be *chosen*, not guessed. Right-sizing three services into 16 GB without OOM, while A still holds its SLA, is the exercise.

- **Base environment:** the **stock free Colab/Kaggle T4 runtime** (Ubuntu + CUDA 12.x + NVIDIA driver — Google/Kaggle fix these; you do not author your own CUDA stack). On top of it, install the **pinned [`requirements.txt`](./requirements.txt) shipped in this repo** (which pins the exact vLLM version) — your `run.sh` does `pip install -r requirements.txt`. Pin exactly these versions; do not float to `latest`. If you must adjust a pin to get it running on your T4, say so in your writeup and commit the version you actually used.
- **Indicative budget:** weights ≈ 3.2 GB total (A ~2.5, B ~1, C ~0.13), leaving ~13 GB to split three ways across each service's KV-cache and activations. `vllm`'s `--gpu-memory-utilization` is a fraction of **total** GPU memory, so all three servers' fractions must sum to **< 1.0** (with headroom), or a later server OOMs at startup. With three workloads this is genuinely tight — sizing the split is the exercise.
- **Token distribution (pinned, state it back):** Workload A — input 256 tokens, output 128 tokens. Workload B — input 512 tokens, output 256 tokens. Workload C (embeddings) — input 256 tokens (no generation). Exclude warm-up requests from all reported numbers.
- **The SLA (the hard constraint):** **Workload A's p99 end-to-end latency under full contention (B + C both loaded) must stay within 2× of its p99 measured alone.** Pick your own absolute target if you prefer, but state it and justify it; the 2×-of-isolated-baseline rule is the default we grade against. An isolation strategy that doesn't bring A back under this line — no matter how clever — has not solved the problem. Report whether you hold it and by how much.
- **⚠ The baseline must be WARM (this is a hard requirement, and it is where submissions go wrong).** The SLA is a *ratio*, so its denominator — A's p99 alone — decides the verdict. vLLM's first requests in a fresh process pay one-off costs (torch.compile, Triton/CUDA-graph autotuning, cache warm-up) that can be **5–10× the steady-state latency**. At `--num-prompts 200`, p99 is roughly the 2nd-worst sample, so **a single cold request owns your p99** and silently inflates your SLA budget by 2–3×. A contaminated baseline makes a config that isolates *nothing* appear to pass.
  **The protocol, required:** before every measured run, warm the server with **≥ 50 requests at the same input/output shape and the same `--max-concurrency` as the measured run**, and discard them. Concurrency matters: warming at 4 does not warm the kernels a run at 8 will select. The determinism flags below make the warm-up shape-identical.
  **Self-check before you submit:** on your `A_alone.json`, compute `p99_e2el_ms / p95_e2el_ms`. **A warm run sits near 1.0–1.2. Above 1.5 means your baseline is contaminated** — re-run it, don't report it.

  **The ratio test catches the bad cases; it does not certify the good ones.** We have measured a cold baseline that passed the 1.5 check and *still* inflated the SLA budget by a third. A ratio under 1.5 is therefore not evidence that you warmed up: **warm up unconditionally** and say so, rather than using the ratio to decide whether you needed to. Likewise, if A comes out *faster* under contention than alone (ratio < 1.0), that is physically impossible — a measurement artifact, not a win, and reporting it as a margin is a serious error. `score.py` warns on both.
- **Goodput, not peak:** report **goodput** = the sustained request rate (req/s) on Workload A while **holding the p99 SLA**, with B and C running. Peak throughput with a blown SLA does not count.
- **The single ranking number (computed by `score.py`, shipped in this repo):**
  ```
  composite = A_goodput_at_SLA  ×  (B+C throughput you retained under isolation)
  ```
  Holding A's SLA is trivial if you just starve B and C — so the score multiplies A's goodput by **how much B+C throughput you preserved**. The only way to score high is to hold A's SLA *and* keep the batch + embedding workloads productive — exactly the production objective (most useful work per GPU under a latency SLA). **If A blows the SLA, A's goodput counts as 0.** Run `python3 score.py --out-dir out` yourself; we recompute it on our re-run. This number does the ranking — not a subjective read of your writeup.
- **T4 note:** Turing (T4) has no FP8 and limited kernel support — fp16/bf16 is fine for these tiny models; don't spend time chasing FP8/quantization here, this task is about isolation, not single-model speed. **Kaggle gives a guaranteed T4 (or 2× T4) and is the more reliable free source**; Colab's free GPU is whatever's available and may not be a T4 when you sit down. Use Kaggle if Colab won't give you a T4.

**Do not swap or drop any of A, B, C** — the three fixed models keep every submission's numbers comparable, and all three co-resident is the hard requirement. The exercise is **three genuinely co-resident workloads on one 16 GB GPU**, with A holding its SLA under B+C contention.

---

## 3. How to approach it

1. **Stand all three up, co-resident.** Get Workloads A, B, and C all serving on the **one** GPU at the same time. Capture an `nvidia-smi` dump showing all three resident in memory simultaneously. (This is the proof you actually co-located — a submission without it cannot score the co-residency part.)
2. **Establish A's isolated baseline + measure the interference.** Benchmark Workload A's latency in two conditions, identical inputs:
   - **(1) A alone** — B and C idle/stopped. This sets your SLA reference (the p99 the SLA is 2× of).
   - **(2) A under full contention** — B running its continuous batch load **and** C serving its embedding load.
   Report **p50 / p95 / p99 latency** and **TTFT vs inter-token latency separately** for both conditions. The **p99 delta is the interference**; whether condition (2) blows the SLA is the problem you now have to fix.
3. **Choose and apply an isolation strategy to hold the SLA.** Pick one (or combine) and justify it:
   - separate processes / containers,
   - MPS,
   - request prioritization / scheduling / admission control (e.g. throttling B so A's SLA survives),
   - anything else you can defend with a measurement.

   > **The memory split does not count as your isolation mechanism.** Dividing `--gpu-memory-utilization` three ways is what lets three servers **boot** at all (three at the ~0.9 default would request 270% of the card) — it is a *co-residency* tool, not a *protection* tool. It is required, and it is not the answer. **Gate: your submission must show a specific mechanism, applied, that moves A's contended p99 back inside the SLA — a measured before/after where the only change is that mechanism.** If the only difference between your "before" and "after" is the memory fraction, you have not done step 3, whatever the composite says.

   > **A flag set to its own default is not a mechanism either.** We read the *effective values* your config ships, not the presence of the flags — a knob passed at the value the engine would have used anyway constrains nothing. For any knob that is part of your answer, show the value you chose and the measured effect of changing it.

   Apply it, then **re-run A under full contention and show A's p99 back within the SLA.** Then **measure goodput**: the sustained req/s on A while holding the SLA with B and C running. Document what you tuned, what you traded away (how much B/C throughput you gave up to save A), and every OOM you hit while right-sizing three services into 16 GB.

   > **Not every lever on that list will work, and finding out which is the exercise.** A negative result, measured and explained, is a legitimate and valuable finding — report it. But it is not the same as holding the SLA, so say plainly which you achieved: "I held it with mechanism X", or "I could not hold it; here is the dose-response proving why the levers I tried cannot, and here is what I would reach for next."
4. **Write it up** (see deliverables). Short, tied to your own numbers.

**Suggested time budget (3–4h):** ~50 min get all three co-resident without OOM + nvidia-smi proof · ~40 min A-alone baseline + A-under-full-contention · ~70–90 min the isolation sweep to hold the SLA + goodput measurement (this is the graded core — spend the most here) · ~30 min writeup + make `run.sh` reproduce clean. If you run short, a rigorous baseline-vs-contention measurement plus an honest "here's the isolation approach I'd take to hold the SLA and why" beats a rushed, unverified sweep.

---

## 4. Commands (use these so numbers are comparable)

**Co-residency proof — capture while all three servers are live and A is under load:**
```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 1 | head -20
nvidia-smi    # full dump showing ALL THREE serving processes resident
```
In this vLLM version the three processes appear in `nvidia-smi` as **`VLLM::EngineCore`**, not as `python3` — so grep for that (or just don't grep; paste the raw dump). A valid three-server dump looks like three `VLLM::EngineCore` rows with non-trivial `GPU Memory Usage`, captured while the card shows **>0% utilization**.
Run A on port 8000, B on 8001, C on 8002 — three `vllm serve` processes, distinct ports, one GPU.

**Workload A — latency benchmark (run once per condition: A-alone, A-under-contention-before-isolation, A-under-contention-after-isolation):**
```bash
vllm bench serve \
  --model unsloth/Llama-3.2-1B-Instruct \
  --base-url http://localhost:8000 \
  --dataset-name random \
  --random-input-len 256 --random-output-len 128 \
  --random-range-ratio 0 \
  --ignore-eos \
  --seed 1234 \
  --num-prompts 200 --max-concurrency 8 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result --result-filename out/A_contended.json
```
**The three flags that keep every submission comparable — don't drop them:**
- `--random-range-ratio 0` → every request is *exactly* 256/128 tokens, not a distribution around it.
- `--ignore-eos` → A always generates the full 128 tokens, so the GPU load is **identical regardless of prompt content** (without this, generation stops at EOS and your load becomes content-/seed-dependent — the exact variability we're removing).
- `--seed 1234` → same sampled tokens every run, so your before/after and our re-run line up.

Use the same flags for the A-alone (`--result-filename out/A_alone.json`) and post-isolation runs, changing only the filename. **Workload B** uses the same generative benchmark with its own lengths (512/256) and `--save-result out/B_*.json`.

**Warm-up before every measured run (required — see §2's SLA note).** Fire a discarded run at the *same shape and same concurrency*, then measure:
```bash
# WARM-UP — same flags, same --max-concurrency, result thrown away
vllm bench serve --model unsloth/Llama-3.2-1B-Instruct --base-url http://localhost:8000 \
  --dataset-name random --random-input-len 256 --random-output-len 128 \
  --random-range-ratio 0 --ignore-eos --seed 1234 \
  --num-prompts 50 --max-concurrency 8 > /dev/null
# then the measured run, with --save-result
```
Do this for A, B and C, alone and contended. It costs about a minute per server and it is the difference between a real baseline and a number that invalidates your whole submission.

**Workload C is an EMBEDDING model — benchmark it differently.** `BAAI/bge-small-en-v1.5` does no text generation, so TTFT/ITL/output-tokens don't apply. Its only job here is to be a **third memory+compute tenant creating contention**, and its "retention" is measured in **embedding requests/sec**, not tokens. In the pinned `vllm==0.19.1` it is served as a pooling model:

```bash
vllm serve BAAI/bge-small-en-v1.5 --port 8002 \
  --runner pooling --convert embed \
  --dtype float16 --gpu-memory-utilization <your fraction>
```
Drive it with `vllm bench serve --backend openai-embeddings --model BAAI/bge-small-en-v1.5 --base-url http://localhost:8002 ... --save-result out/C_*.json` (or, if your build's embedding bench differs, any client that sustains a steady embed load and lets you record requests/sec — state what you used). `score.py` reads C's `request_throughput` for retention, so request/sec is what matters.

> **Use `--num-prompts 5000` or higher for C.** 200 embedding requests complete in well under a second on a T4 — far too short to be a real measurement, and C carries **half** the retention factor in the composite. Size C's runs so each lasts **≥ 30 s**, and so C's load outlasts A's entire measured window.

**Known T4/version gotchas** (these are environment facts, not part of the test — don't burn your 3–4 hours rediscovering them; we verified every one of these on a free Colab T4 with the pinned `requirements.txt`):

- bf16 is unsupported on Turing — pass `--dtype float16` explicitly.
- `--max-model-len` must be set well below A's declared 131,072-position context, or the KV-cache allocation alone exceeds your fraction. 2048 is ample for the shapes below.
- `--disable-log-requests` **was removed** in this vLLM version, and `vllm serve` rejects unknown arguments outright — so passing it means the server never binds its port, which surfaces from a backgrounded launcher as a mystifying "connection refused". Request logging is off by default now; drop the flag.
- If FlashInfer's sm_75 JIT fails on your runtime, `--attention-backend TRITON_ATTN` is a sanctioned workaround.
- `vllm serve --help` is paginated into config *groups* in this version and does not list individual flags. Use `vllm serve --help=all` (likewise `vllm bench serve --help=all`) when you want to confirm a flag exists.
- **Budget for boot time.** A single server takes **1.5–5 minutes** to become ready (weights download on first touch, then CUDA graph capture); all three co-resident took us **~4.5 minutes** of pure startup, and A alone took 260s on a cold cache. Poll `/health` until it answers instead of using a fixed `sleep`, and set any readiness timeout to **at least 420s** or you will kill servers that were about to come up.

Note any of these you used in your writeup.

**Generating the contention correctly matters.** Both B and C must be *actively loading the GPU for the entire duration of A's measured run*, or your "interference" is understated. Don't fire one short batch — keep B (batch generation) and C (embedding requests) under continuous load (background loops, or `--num-prompts` large enough to outlast A's run). Start B and C, confirm via `nvidia-smi` that both are actually working, *then* benchmark A. State exactly how you sustained the contention.

**Goodput measurement.** To find A's goodput-under-SLA, sweep A's offered rate (e.g. raise `--max-concurrency` 1→2→4→8→16) with B+C loaded, and report the **highest sustained req/s at which A still holds the p99 SLA**. The curve (and where A's p99 crosses the SLA line) is the evidence.

---

## 5. Delivery format (hard reproducibility gate)

Your submission must be a **git repo** that we can run end-to-end with **one command**, on a fresh free T4, with no manual fixups. Reproducibility is a hard gate: if it does not come up on our clean run, the technical sections cannot score.

The repo must contain a single entrypoint, `run.sh`, that from a clean runtime:
1. installs the pinned `requirements.txt` (`pip install -r requirements.txt`; no `latest` tags, pinned model revisions),
2. launches **all three** workloads (A, B, C) co-resident with your memory split,
3. prints the `nvidia-smi` co-residency dump showing all three resident,
4. runs the benchmarks and writes their logs to `./out/`, then
5. runs `python3 score.py --out-dir out` and prints your composite score.

**Your `run.sh` must produce these exact files in `out/`** (the `vllm bench serve` runs from §4 with `--save-result --result-filename out/<name>.json`, keeping the determinism flags) so the shipped `score.py` can rank you and we can recompute identically:

| File | What it is |
|---|---|
| `out/A_alone.json` | A benchmarked alone — sets the SLA baseline (its p99 e2e) |
| `out/A_contended.json` | A under full B+C contention, **after** your isolation |
| `out/B_alone.json` | B alone — its uncontended throughput |
| `out/B_contended.json` | B under your isolation (while A holds its SLA) |
| `out/C_alone.json` / `out/C_contended.json` | C (embeddings) alone / under isolation — **required**; the composite's B+C retention factor averages B and C, so missing C understates your score |

Also write `out/nvidia_smi.txt` (the co-residency dump) and `out/goodput_sweep.txt` (the rate sweep showing where A's p99 crosses the SLA).

**Two consistency rules the scorer and our re-run both check:**
1. **Capture `nvidia_smi.txt` while A's benchmark is actually running**, not before or after. A dump showing three resident processes at **0% GPU utilization** proves memory co-residency but not contention — and it's the instrument that would have caught a real defect in a previous submission (a tenant that had silently drained before A's window). Sample it in a loop during the run, and note that a benchmark client takes a little while to ramp before it issues its first request.
2. **The scored `A_contended.json` must be the same operating point as the matching sweep entry.** If your composite comes from a run at `--max-concurrency 8` and your sweep's c8 point reports a materially different req/s, one of them is wrong and we will ask which. Reconcile it before submitting, or explain the discrepancy in your writeup.

**Ship a warm baseline.** Before you submit, run §2's self-check on your own files: `p99_e2el_ms / p95_e2el_ms` on `A_alone.json` should be ~1.0–1.2. `score.py` prints this ratio and warns above 1.5 — but as §2 notes, passing it does not prove you warmed up.

**One-cell bootstrap (required).** Provide a single Colab/Kaggle cell at the top of your `README.md` that clones the repo and runs it, e.g.:
```python
!git clone --depth 1 https://github.com/<you>/<repo>.git && cd <repo> && bash run.sh
```
It must be **idempotent and self-contained** — pinned versions, no interactive prompts, survives a fresh ephemeral runtime. "Works on my Colab" submissions that don't re-run on ours fail the gate.

**Recommended: make it runnable via [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli).** We verify the top slice by provisioning a clean T4 and running your repo headlessly, e.g. `colab run --gpu t4 run.sh`. Authoring your `run.sh` so it executes cleanly under `colab run` (ephemeral provision → execute → teardown) is the surest way to pass our re-verification — and it's exactly how we re-run you, so test it that way yourself.

## 6. What to submit

1. **The repo** (link), with `run.sh` + the one-cell bootstrap above. The serving config = the exact `vllm serve` invocations for **all three** workloads, all flags, the memory split, and any launcher/scheduler code, all pinned. We must bring the three-workload setup up from clean with the single command.
2. **The `nvidia-smi` dump** showing all three models co-resident under load (raw, pasted).
3. **Benchmark logs (the `out/*.json` files above, raw — not hand-typed tables)** + the `score.py` composite-score output your `run.sh` printed.
4. **`out/tuning_trace.md` + `out/sweep/` — your tuning trail (required).** The 2–4 intermediate attempts you made before the final config, each a one-liner in `tuning_trace.md`: the split you tried → what happened (OOM? A's p99? B/C retention?) → what you changed and why. E.g.:
   > `0.45/0.45/0.13 → C OOM'd at startup (sum>0.9 + 3 CUDA ctxs). 0.40/0.35/0.13 → fit, but A p99 2.2× warm (over SLA) because B saturated SMs. Tried <lever X> at three settings → A p99 2.1× (dose-response in sweep/attempt_02-04): not the binding resource. Applied <mechanism Y> → A p99 1.4×, B retained 0.7. Final: 0.40/0.33/0.13 + Y.`

   **Back each non-OOM attempt with its raw measurement**, not just the prose: save the `vllm bench serve --save-result` output as `out/sweep/attempt_01.json`, `attempt_02.json`, … (an OOM attempt can be a one-line note instead — there's no benchmark to save). A real tuner produces these as a byproduct of working; a one-shot config has only the endpoint and no sweep behind it. We sanity-check that the trace, the sweep files, the nvidia-smi memory math, and your final composite all tell the **same** story — and we reconcile it on our re-run. This is part of the score.
5. **A short writeup (~1 page):**
   - Your isolation mechanism and **why** you chose it for this scenario.
   - **Did you hold A's p99 SLA under full contention, and by what margin?** A's p99 alone vs under-contention vs after-isolation, and the **goodput** (sustained req/s holding the SLA).
   - What you traded to hold it (how much B/C throughput you sacrificed; memory headroom).
   - Where the bottleneck was (compute-bound? memory-bandwidth-bound? KV-cache pressure?) and how you know.
   - **Production transfer:** how you would isolate a latency-sensitive product feature from a batch/agentic workload **on our 8× NVLinked B300 box.** Which models get dedicated cards vs share one; MIG vs MPS vs memory fractioning and why; your policy when the cluster is saturated and a new request arrives.

---

## 7. How it is evaluated

This is a filter and an interview seed, not a final ranking. It separates three groups:

- **Could not get all three models co-resident on one GPU** (unresolved OOM — the most common one-shot failure, since a naive split doesn't fit three into 16 GB), or "isolation" claimed with no measurement. → out
- **Co-located but never held the SLA** — interference measured but A's p99 under full contention stays blown, or the isolation step is thin/untuned, or no goodput number. → out
- **All three genuinely co-resident; interference measured honestly against a WARM baseline; a deliberately chosen isolation mechanism — not just the memory split — that brings A's p99 back within the SLA under full contention; goodput-under-SLA reported; the tradeoff (B/C throughput sacrificed) and OOM/right-sizing documented; production writeup that transfers to multi-card.** → advances

**Measurement honesty is graded, and it outweighs the number.** We recompute your composite, and we also check the instrument: baseline warm-up state, whether every tenant was genuinely loaded throughout A's measured window, whether the scored run agrees with your sweep, and whether any reported ratio is physically possible. **A high composite built on a contaminated baseline scores below an honest lower number** — we have seen submissions where the headline figure did not survive inspection, and the reviewer's question is always the same: *did the candidate notice?* Flagging a defect in your own results, or in this task's protocol, is a strong positive signal. Reporting an impossible result as a comfortable margin is the single worst thing a submission can do here, because on a real box these calls set SLAs.

**Ranking among the passers = the `score.py` composite** (A goodput-at-SLA × B+C retention), recomputed on our re-run from your `out/*.json`. It is a single comparable number across all submissions, and it is built so the only way to raise it is to hold A's SLA *while* keeping B+C productive — gaming it requires doing the job. The tuning trace and live defense corroborate; the composite does the ranking.

**We grade the method, not the absolute T4 numbers.** We re-run the top slice on the same free T4 (zero GPU cost to us) to confirm the setup comes up, all three are resident, and the SLA/goodput reproduce. The numbers sort and sanity-check; the writeup and a live conversation carry the hiring signal.

**Why this is hard to one-shot.** A single AI prompt will produce a plausible three-server config, but on a real T4 a guessed memory split OOMs the third server, and a generic config blows A's p99 under genuine B+C contention — it has no measured SLA, no goodput curve, and a writeup that can't explain why its split holds. Clearing all three of *fits-without-OOM*, *holds-the-SLA*, and *reports-real-goodput* is what a generic recipe cannot do. We don't try to detect copying — the constraint makes the generic path fail outright.

---

## 8. Ground rules

- **AI assistance is encouraged.** Note briefly how you used it — that's a plus. But a one-shot-generated config that was never run is obvious and scores poorly: it has no real `nvidia-smi` dump, no interference delta, and a writeup that can't explain its own numbers.
- **Don't fabricate.** Every number must come from a run we can reproduce on a free T4. Round-number results, curves with no knee, a goodput number with no sweep behind it, or "no OOM at default utilization across three servers" are tells.
- **Scope down honestly.** A smaller scope done rigorously beats a wide scope of unverified claims. If you got all three co-resident and measured A-alone vs full-contention but ran out of time on the isolation sweep, submit that and say so — honest partial work with good reasoning beats a polished artifact you can't defend. (If even three-way co-residency without OOM defeated you, tell us what you tried and where it broke — that's still signal.)

We're testing the one thing a single-model task can't: keeping competing workloads from wrecking each other on shared hardware, under a hard SLA. Good luck.
