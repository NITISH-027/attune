# ATTUNE

A solo LLM-personalisation prototype that checks whether a saved preference fits the current query before using it.

**Live demo:** https://13.61.105.132.sslip.io

The demo is hosted on Amazon EC2 (Ubuntu 24.04, t3.micro, Stockholm / eu-north-1). Caddy provides public HTTPS and proxies to FastAPI on localhost. The URL depends on the current EC2 public IP and is not a permanent service guarantee.

## What works

- Explicit preference memory with scope and source evidence.
- Applied/Suppressed decisions and an inspector in the chat UI.
- Editable memory and query-level overrides.
- Halloween-themed interface for Hack-to-Ween 2026.

## Current architecture: v2

FastAPI serves the API and HTML/JavaScript interface. MiniLM embeddings help retrieve candidate memories. Gemini extracts stated preferences, judges applicability to the current query and generates answers. The trained PyTorch v1 gate is retained as a frozen baseline; it is not the central decision-maker in the current v2 memory workflow.

## Evaluation: separate experiments, not one headline score

These are project-reported results from saved evaluation files. They have not been independently reproduced for this deployment.

| Version / test | Reported result | Source |
| --- | --- | --- |
| v1, own 200-query test | 93.0% accuracy; 9.2% negative-transfer rate | `eval/final_scorecard.json`, comparative systems |
| v1, self-authored 150-query preference test (labelled PrefEval in the repository) | 86.0% accuracy; 38.2% negative-transfer rate | `eval/final_scorecard.json`, external_benchmarks |
| v1, CUPID test | 16.3% recall | `eval/final_scorecard.json`, external_benchmarks |
| v2, small CUPID-labelled experiment | 83.3% recall; 50.0% negative-trigger rate | `eval/v2_experiment_results.json` and dashboard telemetry |

The 150-query test is not the official PrefEval benchmark. Lack of exact query overlap does not prove zero leakage: related templates can still overlap. V1 and v2 results use different experiments and should not be treated as an apples-to-apples improvement claim. The current dashboard includes historical baseline telemetry and project-reported metrics, not a guarantee of correctness.

## Limits

This is a prototype, not a production multi-user service. Broad preference reliability, strong memory access control, abuse prevention and scale have not been established. Use synthetic demo preferences only. Do not enter secrets, medical information or other sensitive personal data. The Gemini API can be unavailable or quota-limited. Restarting the host may require restarting the app process.

## Run locally

Install Git LFS before cloning so the checkpoint downloads rather than remaining a pointer:

```bash
git lfs install
git clone https://github.com/NITISH-027/attune.git
cd attune
git lfs pull
python -m venv .venv
# Activate .venv for your operating system.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Store your own Gemini key in a local `.env` file as `GEMINI_API_KEY=...`. Never commit that file or paste keys into issues, screenshots or messages.

```bash
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. A successful `/api/status` response proves the server is running, not that Gemini requests succeed; test a real chat separately.

## Hack-to-Ween disclosure

ATTUNE is an existing solo prototype built in October 2026, extended with a Halloween UI and deployed on AWS during Hack-to-Ween 2026. Built with AI coding assistance. Author: Nitish Prabagaran. Team: nitishprabagaran07 (solo).
