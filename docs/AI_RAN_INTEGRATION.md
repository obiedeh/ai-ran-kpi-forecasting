# AI-for-RAN integration — how this rApp fits, and where the boundary is

This repo implements the **Non-RT RIC KPI-forecasting rApp pattern**, aligned with the **AI-for-RAN** pillar of the [AI-RAN Alliance](https://ai-ran.org/) taxonomy (the other two pillars are AI-and-RAN and AI-on-RAN — both out of scope here).

It is **pattern, not deployment.** This document explains exactly how the pattern would slot into a real Non-RT RIC, and what is intentionally not exercised.

---

## Where this fits in the AI-RAN stack

```
                Service Management & Orchestration (SMO)
              ┌────────────────────────────────────────────┐
              │   Non-RT RIC ─── this rApp's deployment    │
              │                  target                    │
              │   ┌────────────────────────────────────┐   │
              │   │   kpi-forecasting-rapp             │   │
              │   │                                    │   │
              │   │   R1-Data ─► forecast ─► A1-P     │   │
              │   │      ▲                       │     │   │
              │   │      │                       │     │   │
              │   └──────┼───────────────────────┼─────┘   │
              └──────────┼───────────────────────┼─────────┘
                         │ KPM measurements      │ A1 policy
                         │ (R1 data services)    │ candidates
                         │                       ▼
                ┌────────┴──────────┐    ┌───────────────────┐
                │     O1 / R1       │    │   Near-RT RIC     │
                │  (KPM, FCAPS)     │    │  (xApps enforce   │
                └────────┬──────────┘    │   policies via E2)│
                         │               └────────┬──────────┘
                         │                        │ E2 control
                         ▼                        ▼
                ┌─────────────────────────────────────────────┐
                │             O-CU / O-DU / O-RU              │
                └─────────────────────────────────────────────┘
```

The rApp **produces** A1 policy candidates; the Near-RT RIC **enforces** them. Our scope ends at the policy candidate. The Near-RT RIC's xApps and the E2-Control plane that actually changes RAN behaviour are downstream consumers, out of scope here.

---

## Mapping to AI-RAN Alliance taxonomy

The [AI-RAN Alliance](https://ai-ran.org/) (NVIDIA, NTT DOCOMO, Microsoft Azure for Operators, SoftBank, Nokia, Ericsson, Samsung, founded Feb 2024) splits AI work in the RAN into three pillars:

| Pillar | Definition | This repo? |
|---|---|---|
| **AI-and-RAN** | AI and RAN workloads share GPU infrastructure | No |
| **AI-on-RAN** | AI services delivered over the RAN | No |
| **AI-for-RAN** | AI used to *improve* the RAN itself | **Yes — operational layer** |

The companion repo [`wireless-link-intelligence-system`](https://github.com/obiedeh/wireless-link-intelligence-system) targets AI-for-RAN at the **PHY layer** (channel estimation, modulation, INT8 ONNX deployment). This repo targets AI-for-RAN at the **operational layer** (KPI forecasting → A1 policy candidates). Together they cover both tiers of the AI-for-RAN pillar.

---

## What's exercised, what's not

| Component | Status | Where |
|---|---|---|
| KPM-shaped CSV input parsing | ✅ Exercised | `data.py` + `schemas/kpm_input_v1.json` |
| Three-model forecasting (Ridge / GBR / MLP) | ✅ Exercised + measured | `models.py` + `reports/model_comparison/` |
| Forecast → A1 policy candidate conversion | ✅ Exercised + schema-validated | `a1_policy.py` + `schemas/a1_policy_v1.json` |
| End-to-end R1 → forecast → A1 dataflow | ✅ Exercised on synthetic data | `scripts/simulate_r1_dataflow.py` |
| rApp manifest (declaring identity + contracts) | ✅ Real shape | `rapp_manifest.yaml` |
| Scenario evidence packs (congestion / backhaul / outage) | ✅ Real, with dashboards | `reports/scenarios/` |
| Live R1 data plane (KPM wire protocol over E2/O1) | ❌ Not exercised | Wire transport, ASN.1 encoding |
| Live A1-P policy plane | ❌ Not exercised | Wire transport |
| O-RAN SMO integration | ❌ Not exercised | Vendor-specific orchestrator hooks |
| Helm packaging + dms_cli registration | ❌ Not exercised | rApp lifecycle |
| Integration with a Non-RT RIC product | ❌ Not exercised | FlexRIC / OSC RIC / Nokia MantaRay / Ericsson IAP / Mavenir |

The "not exercised" rows are **intentional**. The deliverable is the *pattern* — enough that a senior AI-for-RAN engineer can confirm the manifest, schemas, ML pipeline, and policy-output shape are all real, without expecting a deployable Helm chart.

---

## How this would slot into a real Non-RT RIC

Three reference Non-RT RIC implementations a reader might evaluate this against:

### 1. **FlexRIC** (Eurecom, open-source)
- Academic and research deployments
- Used in most AI-PHY / AI-RAN research papers
- Integration: package this Python module as an rApp container; register with FlexRIC's R1 services for KPM subscription; expose a REST endpoint on A1-P producer side
- Estimated integration effort beyond what's here: ~2 weeks for a competent engineer (mostly Helm + dms_cli + serialization)

### 2. **O-RAN Software Community (OSC) RIC** (Linux Foundation)
- The reference implementation from the O-RAN Alliance
- ONAP-aligned; uses dms_cli for rApp lifecycle management
- Integration: similar to FlexRIC, plus OSC-specific descriptor files
- Estimated integration effort beyond what's here: ~3 weeks

### 3. **Vendor RICs** (Nokia MantaRay, Ericsson IAP, Mavenir, Rakuten Symphony)
- Production-grade; each vendor has its own rApp SDK and marketplace
- Integration: vendor-specific. Each requires re-packaging the manifest + ML artifacts in the vendor's SDK
- Estimated integration effort beyond what's here: ~4–6 weeks per vendor

In all three cases the **ML pipeline**, the **forecasting models**, and the **A1 policy schema** transfer unchanged. What changes is the wire-protocol packaging — which is the part this repo intentionally doesn't ship.

---

## Why "AI-RAN" framing, not just "O-RAN"

O-RAN Alliance (founded 2018) defined the architecture — RIC, E2, A1, O1, R1. AI-RAN Alliance (founded 2024) defines the application — AI-for-RAN, AI-on-RAN, AI-and-RAN. The two are complementary, not competing:

- **O-RAN** = the deployment surface (where this rApp would run)
- **AI-RAN** = the application category (what this rApp does)

In 2026, AI-RAN has more momentum in industry programmes (especially with NVIDIA's involvement), while O-RAN provides the canonical interfaces. This rApp is therefore framed first as **AI-for-RAN at the operational layer** (the application story) and second as a **Non-RT RIC rApp** (the deployment story).

---

## References

- AI-RAN Alliance: https://ai-ran.org/
- O-RAN Alliance: https://www.o-ran.org/
- O-RAN WG2 Non-RT-RIC Architecture: O-RAN.WG2.Non-RT-RIC-ARCH (current release)
- O-RAN E2SM-KPM Service Model: O-RAN.WG3.E2SM-KPM-v02.00.05
- A1-P interface specification: O-RAN.WG2.A1AP (current release)
- Companion L1-layer repo: [`wireless-link-intelligence-system`](https://github.com/obiedeh/wireless-link-intelligence-system)
