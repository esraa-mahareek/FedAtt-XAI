#!/usr/bin/env bash
# Full experimental pipeline of the manuscript. Run from the repository root.
# Expect several GPU-days on one A100; every step is resumable per seed.
set -euo pipefail
TASKS="breakhis bach inbreast cbis_ddsm"
BASELINES="fedavg fedprox fedbn moon fedas fedatt"
SEEDS="0 1 2 3 4"

# 0) partitions + BACH audit ---------------------------------------------------
for t in $TASKS; do
  python scripts/make_partitions.py --config configs/$t.yaml
  python scripts/make_partitions.py --config configs/$t.yaml --set partition.beta=0.1
done
python scripts/audit_bach.py

# 1) hyper-parameter search (validation only, 20 trials) -----------------------
#    Copy the best values from <results>/hparams/<task>/<method>.json into the
#    task yaml (methods.* / personalization.tau) before step 2.
for t in $TASKS; do
  for m in fedprox moon fedatt fedatt_xai fedas; do
    python scripts/hparam_search.py --config configs/$t.yaml --method $m
  done
done

# 2) main comparison, β = 0.5 and β = 0.1 (Tables 9–14) ------------------------
for beta in 0.5 0.1; do
  for t in $TASKS; do for s in $SEEDS; do
    for m in $BASELINES fedatt_xai; do
      python scripts/train_federated.py --config configs/$t.yaml --method $m --seed $s --set partition.beta=$beta
    done
  done; done
done

# 3) centralized reference (Table 15) ------------------------------------------
for t in $TASKS; do for s in $SEEDS; do
  python scripts/train_centralized.py --config configs/$t.yaml --seed $s
done; done

# 4) ablations (Tables 17–21) --------------------------------------------------
for t in $TASKS; do for s in $SEEDS; do
  python scripts/train_federated.py --config configs/$t.yaml --method fedatt_xai --seed $s \
         --tag resnet50 --set model.backbone=resnet50
  python scripts/train_federated.py --config configs/$t.yaml --method fedas_xai --seed $s
  python scripts/tau_sweep.py --config configs/$t.yaml --seed $s
done; done
# number of clients (Table 20)
declare -A NS=( [breakhis]="4 16" [cbis_ddsm]="4 16" [bach]="2 6" [inbreast]="3 8" )
for t in $TASKS; do for n in ${NS[$t]}; do for s in $SEEDS; do
  for m in fedavg fedatt fedas fedatt_xai; do
    python scripts/make_partitions.py --config configs/$t.yaml --seeds $s --set partition.num_clients=$n
    python scripts/train_federated.py --config configs/$t.yaml --method $m --seed $s --set partition.num_clients=$n
  done
done; done; done

# 5) DLG stress test (Table 22) -------------------------------------------------
for t in $TASKS; do for s in $SEEDS; do for m in $BASELINES fedatt_xai; do
  python scripts/run_dlg.py --config configs/$t.yaml --method $m --seed $s
done; done; done

# 6) explanations (Tables 23–29) ------------------------------------------------
for t in $TASKS brecahad; do for s in $SEEDS; do
  python scripts/run_explanations.py --config configs/$t.yaml --seed $s
done; done

# 7) tables --------------------------------------------------------------------
python scripts/make_tables.py --results /content/results --beta 0.5
