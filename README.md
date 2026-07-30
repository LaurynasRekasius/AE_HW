# AE_HW — In Tandem AI Engineer Take-Home

This repo holds the take-home task for the **AI Engineer** role at In Tandem AI.

## The task

**[`Colocation_Task.md`](./Colocation_Task.md)** — Multi-Model Co-location & Isolation.
Co-locate three workloads on one free 16 GB GPU (Colab/Kaggle T4), measure how they
interfere with a latency-sensitive workload, and apply an isolation strategy that holds
a stated p99 SLA under contention. Build-anywhere, ~3–4 hours, runs on free GPU.

Read the task doc in full — it has the fixed inputs, commands, the SLA, deliverables,
and how it's evaluated.

## How to submit — your own PRIVATE repo

Create your **own private git repo** (don't fork this one — we want submissions
independent, and private so candidates don't see each other's work). Give read access
to the GitHub user(s) we name in your invite email, and send us the link when done.

Your repo must contain:

- `run.sh` — single entrypoint that brings the three-workload setup up from a clean
  runtime, runs the benchmarks, and prints your composite score (see task §5).
- `requirements.txt` — **copy the one in this repo** (pins the vLLM version); `run.sh`
  installs it on top of the stock Colab/Kaggle T4 runtime. Don't float to `latest`.
- `score.py` — **copy the one from this repo unchanged** (we recompute with the same
  scorer on our re-run; don't modify it).
- A one-cell Colab/Kaggle bootstrap in your README (`git clone … && bash run.sh`).
- Your serving config, the `out/*.json` benchmark logs, `out/nvidia_smi.txt`,
  `out/tuning_trace.md` + the `out/sweep/attempt_*.json` files backing it, and a
  ~1-page writeup.

## How you're ranked — the task scores itself

We don't rank you on a subjective read of your writeup. **[`score.py`](./score.py)**
computes one comparable number from your logs:

```
composite = A_goodput_at_SLA  ×  (B+C throughput you retained under isolation)
```

Holding the latency SLA on workload A is trivial if you just starve B and C — so the
score multiplies A's goodput by how much B+C throughput you **kept alive**. The only
way to score high is to hold A's SLA *and* keep the batch + embedding workloads
productive (and if A blows the SLA, its goodput counts as **0**). That's the real
production objective. Run `python3 score.py --out-dir out` yourself; we recompute it
on our re-run, which is the source of truth.

Reproducibility is a hard gate: we re-run the top submissions on a free T4. If it
doesn't come up cleanly from your one command, the technical sections can't score.

### Two things that sink submissions — read these before you start

1. **Your baseline must be warm.** The SLA is a *ratio*, so A's p99-measured-alone is its
   denominator. vLLM's first requests pay one-off compile/autotune costs that can be 5–10×
   steady state, and at 200 prompts a single cold request owns your p99 — inflating your SLA
   budget 2–3× and making a config that isolates *nothing* look like it passes. Warm every
   server with ≥50 discarded requests at the same shape **and same concurrency**, then measure.
   `score.py` prints a `p99/p95` warmth check and warns if it exceeds 1.5. Task doc §2.
2. **The memory split is not your isolation mechanism.** Dividing `--gpu-memory-utilization`
   three ways is what lets three servers *boot*; it is required and it is not the answer. Gate:
   show a measured before/after where the only change is the mechanism you chose. Task doc §3.3.

`score.py` also audits your measurements (baseline warmth, physically impossible ratios,
too-short windows) and prints warnings. They don't change your score — but we re-check them on
our re-run, so shipping an unflagged warning is a serious negative signal. Fix the measurement,
not the scorer.

## Ground rules (short version)

AI assistance is encouraged — note how you used it. Don't fabricate numbers; every
result must come from a run we can reproduce. Scope down honestly if you run short.
Full rules are in the task doc.
