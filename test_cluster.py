"""Quick cluster-feature test."""
import time, warnings
warnings.filterwarnings("ignore")
from benchmark.cv_harness import build_extended_sales, gated_sacred_eval, FOLDS_8
from policies import EnsemblePolicy

fs, fis, master = build_extended_sales()
WW = dict(w_ma=0.2875, w_lgb_share=0.6971, multiplier=1.0731, safety_units=0.5962)

for k in (4, 8, 12):
    print(f"\n===== Cluster k={k} =====", flush=True)
    t0 = time.time()

    def make(kk=k):
        return EnsemblePolicy(
            coverage_weeks=2, w_ma=WW["w_ma"], multiplier=WW["multiplier"],
            safety_units=WW["safety_units"], w_lgb_share=WW["w_lgb_share"],
            master=master, censoring_strategy="mean_impute",
            per_series_scaling=True, demand_cluster_k=kk,
        )

    out = gated_sacred_eval(
        name=f"Ensemble [champion_cluster_{k}]",
        policy_factory=make,
        full_sales=fs, full_in_stock=fis, master=master,
        val_tolerance=0.10, force_sacred=False,
        n_workers=1, coverage_weeks=2, folds=FOLDS_8,
    )
    print(f"elapsed: {time.time()-t0:.1f}s", flush=True)
