# FedAtt-XAI — implementation

Code for *Attention-Weighted Aggregation and Client Personalization for Explainable
Federated Breast Cancer Image Classification*. Every stage of Algorithm 1 is
implemented, together with all baselines, ablations, the centralized reference,
the explanation-evaluation protocol, the DLG stress test and the statistics.

## 1. Repository map

| Manuscript element | File |
|---|---|
| Dataset indexing, label mapping (§3.1.3, §4.1.1) | `fedattxai/data/datasets.py` |
| Grouped Dirichlet partition, 70/10/20 group split, leakage assertion (Alg. 1 lines 1–9) | `fedattxai/data/partition.py` |
| Macenko / DICOM preprocessing fitted on client train only (line 10, §3.3) | `fedattxai/data/preprocessing.py` |
| Per-client caches, augmentation | `fedattxai/data/loaders.py` |
| BACH near-duplicate audit (Table 6) | `fedattxai/data/audit.py`, `scripts/audit_bach.py` |
| ViT-Small/16 IN-21k, attention capture, ResNet-50 ablation | `fedattxai/models/backbones.py` |
| Local AdamW training, FedProx, MOON, FedAS client steps (lines 14–19) | `fedattxai/fl/local_training.py` |
| FedAvg, layer-wise FedAtt (Eq. 6–7), FedAS, FedBN-LN | `fedattxai/fl/aggregation.py` |
| Server loop, validation checkpoint (line 23), evaluation | `fedattxai/fl/server.py` |
| Adaptive personalisation τ / Ep (lines 26–34, Eq. 8) | `fedattxai/fl/personalization.py` |
| Centralized reference (§4.6.1) | `fedattxai/fl/centralized.py` |
| Metrics, Youden J, pooled confusion matrix, ECE, Brier (§4.5.1) | `fedattxai/evaluation/metrics.py` |
| BCa bootstrap, rank-biserial (§4.7) | `fedattxai/evaluation/statistics.py` |
| Raw attention, rollout, ViT-Grad-CAM, GradientSHAP (§3.6) | `fedattxai/xai/attributions.py` |
| Deletion/insertion, stability, IoU, pointing game, cross-client agreement, failure modes (§4.5.2) | `fedattxai/xai/evaluation.py` |
| DLG attack (§4.9) | `fedattxai/privacy/dlg.py` |

## 2. Installation (Colab A100)

```bash
pip install -r requirements.txt
```

## 3. Expected data layout

```
/content/data/BreaKHis_v1/**/SOB_[B|M]_*.png
/content/data/ICIAR2018_BACH_Challenge/Photos/{Normal,Benign,InSitu,Invasive}/*.tif
/content/data/INbreast/AllDICOMs/*.dcm   AllXML/*.xml   INbreast.xls
/content/data/CBIS-DDSM/*case_description*.csv   CBIS-DDSM/<TCIA series folders>/**.dcm
/content/data/BreCaHAD/images/*.tif   groundTruth/*.json
```
Paths are set in `configs/*.yaml`.

## 4. Running

```bash
# one run
python scripts/make_partitions.py --config configs/breakhis.yaml
python scripts/train_federated.py --config configs/breakhis.yaml --method fedatt_xai --seed 0
python scripts/run_explanations.py --config configs/breakhis.yaml --seed 0
# everything
bash scripts/run_all.sh
```
Any config value can be overridden: `--set partition.beta=0.1 fl.rounds=100`.
Methods: `fedavg fedprox fedbn moon fedas fedatt fedatt_xai fedas_xai`.

Outputs per run (`results/<task>/beta<β>_N<N>/<method>/seed<s>/`): resolved config,
environment export, per-round log, checkpoint and SHA-256, raw per-client predictions,
`metrics.json` (global and personalised, pooled and per client), cost.
`scripts/make_tables.py` rebuilds the tables from these files.

## 5. Implementation decisions that the manuscript should state

These choices are not fully determined by the text; they are fixed here and should
be reflected in the Methods (or changed in the config to match what was run).

1. **Dirichlet allocation.** For each class, client proportions ~ Dir(β·1_N) and whole
   groups are assigned greedily to the client furthest below its target (standard
   Hsu et al. label-skew protocol at group level). Algorithm 1 line 3 writes the draw
   per client; the two formulations should be aligned in the text.
2. **Learning-rate schedule.** Cosine decay across communication rounds; the AdamW
   state is re-initialised at every round on every client.
3. **FedAtt scale T.** Tuned by random search (log-uniform 1e-2…1e2); layer groups are
   `patch_embed`, `cls_token`, `pos_embed`, `blocks.0…11`, `norm`; the head is averaged
   by sample count.
4. **FedAS.** The manuscript uses PFLlib (commit 44c2d67). This package contains a
   self-contained re-implementation (head-only parameter alignment + Fisher-trace
   weighted aggregation of the shared encoder) so that all methods share one training
   loop. For exact correspondence with the reported FedAS numbers, run PFLlib's
   `clientas`/`serveras` under the same partitions.
5. **MOON** uses the ViT pre-logit CLS representation (no extra projection head).
6. **Mammography normalisation.** Only [0,1] scaling (`mammo_channel_norm: false`), as
   written; set to `true` if channel statistics were also applied.
7. **BreCaHAD masks.** Point annotations rendered as disks (radius 12 px, configurable);
   preprocessing and SHAP background taken from BreaKHis client 0.
8. **Personalisation.** A single run up to max(grid) epochs with per-epoch validation
   loss is equivalent to training each Ep candidate separately (constant η_p, seeded
   data order).
9. **Failure mode 1.** A top-20 % mask covers 20 % of the image by construction, so
   "single component covering < 10 % of the image" cannot be satisfied as written; the
   definition in §4.10 needs revising (e.g. largest component < 10 %).
10. **Standard deviation** uses ddof = 1 across the five seeds.

## 6. Scope

Data decentralisation only — no differential privacy, secure aggregation or
encryption. Explanations are evaluated computationally, not clinically.
