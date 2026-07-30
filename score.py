#!/usr/bin/env python3
"""
score.py — the standardized scorer for the co-location task.

This is what RANKS submissions (not oral defense). Everyone runs the same scorer
against the same standardized log files, so the number is directly comparable
across candidates — and we recompute it on our own re-run as the source of truth.

THE OBJECTIVE (why this number is hard to game)
-----------------------------------------------
The production goal is: extract the most useful work from one GPU WHILE honoring a
latency SLA on the interactive workload. Holding A's SLA is trivial if you starve B
and C — so a score that only rewarded "A held its SLA" would be gameable by killing
the other workloads. This scorer multiplies A's goodput by how much B+C throughput
you PRESERVED, so the only way to score high is to hold A's SLA *and* keep the batch
and embedding workloads productive. That is exactly the engineering skill we hire for.

    composite = A_goodput_at_SLA  ×  (B+C throughput retained under your isolation)

    - A_goodput_at_SLA: A's sustained req/s, counted as 0 if A's p99 under full
      contention exceeds the SLA (default 2× its isolated baseline). Blow the SLA → 0.
    - retention: (B+C throughput WITH your isolation) / (B+C throughput running alone),
      clamped to [0,1]. Starve B/C to save A → retention→0 → low score.

Continuous, reproducible, and gaming it == doing the job. The tuning trace
(out/tuning_trace.md) and the live defense remain — but the NUMBER does the ranking.

EXPECTED INPUTS (your run.sh must produce these via `vllm bench serve --save-result`)
------------------------------------------------------------------------------------
  out/A_alone.json        A benchmarked alone  (sets the SLA baseline = its p99 e2e)
  out/A_contended.json    A under full B+C contention, AFTER your isolation
  out/B_alone.json        B benchmarked alone  (B's max throughput, no contention)
  out/B_contended.json    B under your isolation (while A holds its SLA)
  out/C_alone.json        C (embeddings) alone   — retention measured in requests/sec
  out/C_contended.json    C under your isolation   (embeddings have no output tokens)

A/B are generative (token throughput); C is an embedding model (request throughput).
Each file is the JSON that `vllm bench serve --save-result --result-filename <f>`
writes. We read p99 end-to-end latency and request/token throughput from it. If C's
files are absent the retention factor falls back to B only (understating your score).

MEASUREMENT VALIDITY (read this — it can invalidate your score)
--------------------------------------------------------------
The SLA is a RATIO, so its denominator (A's p99 measured alone) decides the verdict.
vLLM's first requests in a fresh process pay one-off compile/autotune costs that can be
5-10x steady-state. At --num-prompts 200, p99 is ~the 2nd-worst sample, so ONE cold
request can own your p99 and inflate your SLA budget 2-3x. A contaminated baseline makes
a config that isolates NOTHING look like it passes.

This scorer therefore also audits the instrument and prints warnings:

  W1  A_alone p99/p95 > 1.5     -> baseline contaminated by cold-start; re-run it warm
                                   (one-sided: a ratio UNDER 1.5 does not prove you warmed
                                   up. A measured cold baseline on a free T4 came in at 1.21
                                   and still inflated the SLA budget 1.34x. Warm up always.)
  W2  contended/alone p99 < 1.0 -> physically impossible (A "faster" under load);
                                   a measurement artifact, not a margin
  W3  any measured window < 5s  -> too short to carry a scored factor

Warnings do NOT change the composite — we grade the artifact you submitted. But we check
them on our re-run, and an unflagged W1/W2 in a submission is a serious negative signal.
Fix your measurement, don't fix the scorer.

USAGE
-----
  python3 score.py --out-dir out                 # default SLA = 2.0x baseline
  python3 score.py --out-dir out --sla-mult 2.0  # explicit
"""
import argparse, json, os, sys


def load(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ! could not parse {path}: {e}", file=sys.stderr)
        return None


def p99_e2e_ms(d):
    """vllm bench serve --save-result keys vary slightly by version; try the common ones."""
    for k in ("p99_e2el_ms", "p99_e2e_ms", "p99_latency_ms", "p99_ms"):
        if d and k in d and d[k] is not None:
            return float(d[k])
    # some versions nest under 'percentiles' or use 'e2el'
    for k in ("e2el_p99_ms", "median_e2el_ms"):
        if d and k in d and d[k] is not None:
            return float(d[k])
    return None


def _get(d, *keys):
    for k in keys:
        if d and k in d and d[k] is not None:
            return float(d[k])
    return None


def validity_warnings(A_alone, A_cont, out_dir, sla_mult):
    """Audit the instrument, not the result. Returns a list of warning strings."""
    warns = []

    # W1 — cold-start contamination in the SLA denominator.
    p99, p95 = _get(A_alone, "p99_e2el_ms", "p99_e2e_ms"), _get(A_alone, "p95_e2el_ms", "p95_e2e_ms")
    if p99 and p95:
        ratio = p99 / p95
        if ratio > 1.5:
            msg = (f"W1 CONTAMINATED BASELINE: A_alone p99/p95 = {ratio:.2f} "
                   f"(p99 {p99:.0f} ms vs p95 {p95:.0f} ms). A warm run sits near 1.0-1.2.\n"
                   f"     One cold-start request is owning your p99, which inflates the SLA "
                   f"limit to {sla_mult:g}x an outlier\n"
                   f"     instead of {sla_mult:g}x your real latency. Warm the server with >=50 "
                   f"requests at the SAME shape and\n"
                   f"     SAME --max-concurrency, discard them, then re-measure. See task sheet "
                   f"section 2 (the SLA note).")
            ttft99, ttft50 = _get(A_alone, "p99_ttft_ms"), _get(A_alone, "p50_ttft_ms", "mean_ttft_ms")
            if ttft99 and ttft50 and ttft50 > 0:
                msg += f"\n     Corroborating: A_alone p99 TTFT {ttft99:.0f} ms vs p50 {ttft50:.0f} ms (x{ttft99/ttft50:.1f})."
            warns.append(msg)

    # W2 — physically impossible interference.
    cont = _get(A_cont, "p99_e2el_ms", "p99_e2e_ms")
    if p99 and cont and cont / p99 < 1.0:
        warns.append(
            f"W2 IMPOSSIBLE RATIO: A's p99 is {cont/p99:.2f}x under contention, i.e. FASTER with "
            f"two neighbours\n"
            f"     competing for the same SMs than alone on an idle GPU. That cannot happen. It "
            f"means the two runs\n"
            f"     are not comparable (almost always W1 above). Do NOT report this as headroom - "
            f"compare like-for-like\n"
            f"     (p50 vs p50, p95 vs p95) to see your true interference, then fix the baseline.")

    # W3 — windows too short to carry a scored factor.
    for name in ("A_alone", "A_contended", "B_alone", "B_contended", "C_alone", "C_contended"):
        d = load(f"{out_dir}/{name}.json")
        dur = _get(d, "duration")
        if dur is not None and dur < 5.0:
            n = d.get("completed") or d.get("num_prompts")
            warns.append(
                f"W3 SHORT WINDOW: {name}.json ran {dur:.2f}s ({n} requests) - too short to be a "
                f"measurement.\n"
                f"     One scheduling hiccup moves this ratio materially, and C carries HALF the "
                f"retention factor.\n"
                f"     Raise --num-prompts so every scored run lasts >=30s.")
    return warns


def req_throughput(d):
    for k in ("request_throughput", "requests_per_second", "req_per_sec"):
        if d and k in d and d[k] is not None:
            return float(d[k])
    return None


def tok_throughput(d):
    for k in ("output_throughput", "total_token_throughput", "output_tokens_per_second"):
        if d and k in d and d[k] is not None:
            return float(d[k])
    return None


def main():
    ap = argparse.ArgumentParser(description="Composite scorer for the co-location task.")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--sla-mult", type=float, default=2.0,
                    help="A's p99 under contention must be <= sla-mult × its isolated baseline (default 2.0).")
    a = ap.parse_args()
    o = a.out_dir

    A_alone = load(f"{o}/A_alone.json")
    A_cont = load(f"{o}/A_contended.json")
    B_alone = load(f"{o}/B_alone.json")
    B_cont = load(f"{o}/B_contended.json")
    C_alone = load(f"{o}/C_alone.json")
    C_cont = load(f"{o}/C_contended.json")

    if not (A_alone and A_cont and B_alone and B_cont):
        sys.exit("MISSING REQUIRED LOGS: need A_alone, A_contended, B_alone, B_contended in out/. "
                 "Score cannot be computed — this fails the reproducibility gate.")

    base = p99_e2e_ms(A_alone)
    cont = p99_e2e_ms(A_cont)
    if base is None or cont is None:
        sys.exit("Could not read A's p99 e2e latency from the logs. Check --save-result output.")

    sla_limit = a.sla_mult * base
    sla_ok = cont <= sla_limit

    a_goodput = req_throughput(A_cont) or 0.0
    if not sla_ok:
        a_goodput_scored = 0.0
    else:
        a_goodput_scored = a_goodput

    # B+C retention: throughput under isolation / throughput alone (token-based; req fallback).
    def retention(alone, contd):
        if not (alone and contd):
            return None
        a_tp = tok_throughput(alone) or req_throughput(alone)
        c_tp = tok_throughput(contd) or req_throughput(contd)
        if not a_tp or a_tp <= 0 or c_tp is None:
            return None
        return max(0.0, min(1.0, c_tp / a_tp))

    b_ret = retention(B_alone, B_cont)
    c_ret = retention(C_alone, C_cont)
    rets = [r for r in (b_ret, c_ret) if r is not None]
    bc_retention = sum(rets) / len(rets) if rets else 0.0

    composite = round(a_goodput_scored * bc_retention, 4)

    print("=" * 60)
    print("CO-LOCATION TASK — COMPOSITE SCORE")
    print("=" * 60)
    print(f"A p99 e2e — isolated baseline : {base:.1f} ms")
    print(f"A p99 e2e — under B+C         : {cont:.1f} ms")
    print(f"SLA limit ({a.sla_mult:g}× baseline)      : {sla_limit:.1f} ms  -> {'HELD ✅' if sla_ok else 'BLOWN ❌ (goodput counts as 0)'}")
    print(f"A goodput under contention    : {a_goodput:.3f} req/s "
          f"({'counted' if sla_ok else 'ZEROED — SLA blown'})")
    print(f"B throughput retained         : {f'{b_ret:.2f}' if b_ret is not None else 'n/a'}")
    print(f"C throughput retained         : {f'{c_ret:.2f}' if c_ret is not None else 'n/a'}")
    print(f"B+C retention factor          : {bc_retention:.3f}")
    print("-" * 60)
    print(f"COMPOSITE SCORE = {composite}")
    print("  (= A goodput-at-SLA × B+C retention; higher is better; 0 if SLA blown)")
    print("=" * 60)

    # ---- measurement-validity audit (does NOT change the composite) ----
    warns = validity_warnings(A_alone, A_cont, o, a.sla_mult)
    p99a = _get(A_alone, "p99_e2el_ms", "p99_e2e_ms")
    p95a = _get(A_alone, "p95_e2el_ms", "p95_e2e_ms")
    if p99a and p95a:
        print(f"baseline warmth check: A_alone p99/p95 = {p99a/p95a:.2f} "
              f"({'OK — warm' if p99a/p95a <= 1.5 else 'FAIL — cold-start contaminated'}) "
              f"[target ~1.0–1.2]")
    if warns:
        print()
        print("!" * 60)
        print(f"MEASUREMENT VALIDITY: {len(warns)} WARNING(S) — the composite above may not")
        print("mean what it appears to mean. Fix the measurement before you submit.")
        print("!" * 60)
        for w in warns:
            print(f"  {w}")
        print("!" * 60)
        print("These do not change your score — we grade what you submit. But we re-run and")
        print("re-check them, so shipping an unflagged warning is a serious negative signal.")
        print("If you disagree with a warning, say so in your writeup and show your reasoning.")
        print("!" * 60)
    else:
        print("measurement-validity audit: no warnings ✅")

    # machine-readable line for collation across candidates
    print(json.dumps({"composite": composite, "sla_ok": sla_ok,
                      "a_p99_baseline_ms": base, "a_p99_contended_ms": cont,
                      "a_goodput_req_s": a_goodput, "bc_retention": round(bc_retention, 4),
                      "warnings": [w.split(":")[0] for w in warns],
                      "baseline_p99_p95": round(p99a / p95a, 3) if (p99a and p95a) else None}))


if __name__ == "__main__":
    main()
