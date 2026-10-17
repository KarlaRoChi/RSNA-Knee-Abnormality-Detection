# ⚖️ RSNA Knee Abnormality Detection — Rules & Prize Eligibility

**Official rules:** <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules>
**Last checked against the official pages:** 2026-10-17
*This is a summary for internal use. The official rules page is authoritative.*

---

## 1. Key terms

| Item | Value |
|---|---|
| Sponsor | Radiological Society of North America (RSNA) |
| Prize pool | $77,000 total: main LB $9k / $7k / $6.5k / $6k / $5.5k / $5k ×5 (10 places), plus an Efficiency track paying $7k / $6k / $5k |
| Winner license | **CC BY-NC 4.0** (winning submission and its source code) |
| Data license | Competition use plus commercial and academic research, under the **RSNA MIRA license** (<http://rsna.org/mira-license>) |
| Team size | Max 5. Mergers allowed if the combined submission count fits the limit |
| Submissions | 5 per day, 2 final selections |
| Code requirements | Notebook only. CPU or GPU ≤ 9 h. **Internet disabled.** Free, public external data and pre-trained models allowed. Output `submission.csv` |

## 2. Deadlines (23:59 UTC)

| Date | Milestone |
|---|---|
| 2026-10-15 | Entry + team merger deadline (**passed**) |
| **2026-10-22** | **Final submission deadline** |
| 2026-11-05 | Winners' Requirement Deadline: training code, video and method description due |

## 3. What a submission needs to be prize-eligible

1. **Hidden-test rerun works offline.** Internet must be off, and all weights must be mounted from Kaggle datasets or models.
2. **External data and models are public and free.** This also covers "reasonably accessible, minimal cost" items. Each dataset you attach must stay public until the end of the competition.
3. **Open-source code only** under an OSI license that allows commercial use (Foundational Rules §6.c).
4. **No hand-labelling or human prediction of test records.** Labels derived from the *training* reports are fine.
5. **No private sharing** of code or data outside your team.
6. **Winner obligations, which are stricter than in most competitions:**
   - Deliver **training code, inference code and model weights**, plus a description of the compute environment: a `requirements.txt` and either the Kaggle image or a Dockerfile.
   - Publish the **model weights as a public Kaggle dataset** linked to the inference notebook.
   - Post a link to the open-sourced code and weights on the forum.
   - Record a **short video** of the approach.
   - Publish the final model for open distribution (as a Kaggle Model).
   - Sign the prize documents and tax forms, and take part in a call or panel with the host if asked.
7. **Efficiency track.** Only the submissions you select for the main LB are considered, and they must beat `sample_submission.csv` on the private LB. The score combines AUC with runtime (9 h = 32,400 s), lower is better. The exact formula is in `OVERVIEW.md`.

## 4. Audit of our notebooks (2026-10-17)

| Kernel | Internet | External inputs | Eligibility |
|---|---|---|---|
| `rochikarla/rsna-knee-abnormality-detection-baseline` (`kaggle_rsna_knee_baseline.ipynb`) | Off | Competition data only | ✅ Eligible. Our own code end to end |
| `rochikarla/rsna-knee-radiomics-ensemble` | Off | Competition data only | ✅ Eligible |
| `rochikarla/rsna-knee-abnormality-detection-vision-2-5d` | Off | Competition data only (no pre-trained weights) | ✅ Eligible |
| `rochikarla/rsna-knee-public-blend` (`kernel_public_blend/`) | Off | 5 public datasets from other competitors (table below) | ⚠️ **At risk.** See §5 |

### External datasets used by `kernel_public_blend`

| Dataset | License on Kaggle | Public? |
|---|---|---|
| `goodpjw2008/rsna-knee-2-5d-convnext-reader` | Apache 2.0 | ✅ |
| `heliosli/rsna-knee-compact-r7cnxfsss` | "Other" (unspecified) | ✅ |
| `rostigor/rsna-knee-r6-fe055406` | "Other" (unspecified) | ✅ |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | "Other" (unspecified) | ✅ |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | "Other" (unspecified) | ✅ |
| `nartaa/rsna-knee-oai-original384-s42-pair` (fallback path in code only) | — | ❌ **403: private or deleted** |

## 5. Risks and actions

- ⚠️ **The public blend is a fork with no model changes, built from other competitors' weights.** As a winner you must deliver the **training code** and weights. We do not own the training code for the nartaa, heliosli or rostigor checkpoints, and four of the five weight datasets have an unspecified licence. Before selecting it as a final submission, either:
  1. confirm that each source author has published training code under a compatible licence, and ask the host in the forum whether a pure public-weights blend qualifies, or
  2. select one of our own pipelines (baseline, radiomics or vision 2.5D) as the second final submission so that at least one pick is fully eligible.
- ⚠️ **Labels for the public weights were produced by an LLM reading the training reports.** This is allowed because it is derived from training data, not test data. The winners' write-up would need to name the LLM and show that it was free or low-cost to use.
- ⚠️ **Remove the fallback path to `nartaa/rsna-knee-oai-original384-s42-pair`.** The dataset is not public. Today the code only falls back to it, but an inaccessible input must never be what actually gets used.
- ⚠️ The "OAI" in the weight names suggests the Osteoarthritis Initiative was used as external pre-training data. OAI requires a free registration and a data-use agreement. Confirm that this still counts as "reasonably accessible to all".
- ⏰ **Final deadline is 2026-10-22.** Select the 2 final submissions before then.
