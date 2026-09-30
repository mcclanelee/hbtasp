"""Validate and analyze the completed V4 final factorial."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/checkpoints/v4_final_factorial"
DATA=OUT/"cell_results.csv"
KEYS=["period_ms","lines","seed"]
METRICS=["mandatory_dmr","historical_completed_only_dice",
         "historical_coverage_adjusted_dice","historical_weighted_coverage_utility",
         "historical_mandatory_effective_dice","mean_complete_image_dice",
         "pixel_defect_recall","image_complete_miss_rate"]

def paired(d,a,b,metric):
    x=d[d.configuration==a].set_index(KEYS)[metric].sort_index()
    y=d[d.configuration==b].set_index(KEYS)[metric].sort_index()
    delta=x-y
    delta=delta.mask(np.isclose(delta,0,atol=1e-12),0.0)
    n=len(delta);sem=delta.std(ddof=1)/np.sqrt(n)
    ci=((delta.mean(),delta.mean()) if sem==0 else
        stats.t.interval(.95,n-1,loc=delta.mean(),scale=sem))
    test=stats.ttest_rel(x,y) if sem>0 else None
    return {"contrast":f"{a} - {b}","metric":metric,"n":n,
            "mean_difference":delta.mean(),"ci95_low":ci[0],"ci95_high":ci[1],
            "paired_t_p":test.pvalue if test is not None else 1.0,"wins":int((delta>0).sum()),
            "ties":int((delta==0).sum()),"losses":int((delta<0).sum())}

def main():
    global OUT, DATA
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.out if args.out.is_absolute() else ROOT/args.out
    DATA=OUT/"cell_results.csv"
    d=pd.read_csv(DATA)
    counts=d.groupby("configuration").size()
    if len(d)!=800 or not (counts==200).all(): raise RuntimeError(f"incomplete: {counts}")
    edf=d.configuration.str.startswith("EDF")
    edf_account=d.completed+d.expired_waiting+d.admission_infeasible.fillna(0)
    h_account=(d.completed+d.expired_waiting+d.mandatory_infeasible.fillna(0)
               +d.optional_skipped.fillna(0)+d.dispatch_infeasible.fillna(0))
    residual=np.where(edf,d.total_regions-edf_account,d.total_regions-h_account)
    if np.abs(residual).max()!=0: raise RuntimeError("terminal accounting failure")
    if d.thermal_violations.sum()!=0: raise RuntimeError("normal thermal violation")
    summary=d.groupby("configuration")[METRICS].agg(["mean","std"]).reset_index()
    summary.to_csv(OUT/"configuration_summary.csv",index=False)
    contrasts=[]
    pairs=[("HBTASP-Dynamic","EDF-Dynamic-Reservation"),
           ("HBTASP-Dynamic","HBTASP-FixedL3"),
           ("HBTASP-FixedL3","EDF-FixedL3")]
    for a,b in pairs:
        for metric in METRICS: contrasts.append(paired(d,a,b,metric))
    c=pd.DataFrame(contrasts);c.to_csv(OUT/"paired_contrasts.csv",index=False)
    indexed=d.set_index(KEYS)
    interactions=[]
    for metric in METRICS:
        h_dyn=indexed[indexed.configuration=="HBTASP-Dynamic"][metric]
        h_fix=indexed[indexed.configuration=="HBTASP-FixedL3"][metric]
        e_dyn=indexed[indexed.configuration=="EDF-Dynamic-Reservation"][metric]
        e_fix=indexed[indexed.configuration=="EDF-FixedL3"][metric]
        delta=(h_dyn-h_fix)-(e_dyn-e_fix)
        sem=delta.std(ddof=1)/np.sqrt(len(delta))
        ci=((delta.mean(),delta.mean()) if sem==0 else
            stats.t.interval(.95,len(delta)-1,loc=delta.mean(),scale=sem))
        interactions.append({"metric":metric,"difference_in_differences":delta.mean(),
                             "ci95_low":ci[0],"ci95_high":ci[1]})
    pd.DataFrame(interactions).to_csv(OUT/"factorial_interactions.csv",index=False)
    means=d.groupby("configuration")[METRICS].mean()
    core=c[(c.contrast=="HBTASP-Dynamic - EDF-Dynamic-Reservation")]
    lines=[]
    lines.append("# Final 2x2 factorial analysis\n")
    lines.append("Status: scheduling, historical Average-Dice utility, and zero-credit confusion-replay evidence.\n")
    lines.append("All 800 cells are present (200/configuration), terminal accounting residual is zero, and modeled thermal violations are zero.\n")
    lines.append("## Same-model primary means\n")
    lines.append("| Configuration | Mandatory DMR | Completed-only Dice | Coverage-adjusted Dice | Weighted coverage utility | Mandatory effective Dice |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name,row in means.iterrows():
        lines.append(f"| {name} | {row.mandatory_dmr:.4f} | {row.historical_completed_only_dice:.4f} | {row.historical_coverage_adjusted_dice:.4f} | {row.historical_weighted_coverage_utility:.4f} | {row.historical_mandatory_effective_dice:.4f} |")
    lines.append("\n## HBTASP-Dynamic minus EDF-Dynamic-Reservation\n")
    lines.append("| Metric | Difference | 95% CI | p | Wins/200 |")
    lines.append("|---|---:|---:|---:|---:|")
    for _,r in core.iterrows():
        lines.append(f"| {r.metric} | {r.mean_difference:.5f} | [{r.ci95_low:.5f}, {r.ci95_high:.5f}] | {r.paired_t_p:.3g} | {int(r.wins)} |")
    lines.append("\nWeighted utility is the manuscript-objective metric. Coverage-adjusted Dice is unweighted and is retained for R2.8 transparency. No universal dominance claim is made.\n")
    (OUT/"FINAL_ANALYSIS.md").write_text("\n".join(lines),encoding="utf-8")
    cp=json.loads((OUT/"checkpoint.json").read_text(encoding="utf-8"));cp.update({
        "status":"complete_validated","validated_cells":len(d),
        "max_terminal_accounting_residual":float(np.abs(residual).max()),
        "total_thermal_violations":int(d.thermal_violations.sum())})
    (OUT/"checkpoint.json").write_text(json.dumps(cp,indent=2),encoding="utf-8")
    print(means[METRICS[:5]].round(6).to_string());print("\n",core.to_string(index=False))

if __name__=="__main__":main()
