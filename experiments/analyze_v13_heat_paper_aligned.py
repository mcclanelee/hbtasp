from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
try:
 from experiments.publication_style import apply_publication_style, BLUE, ORANGE, GREEN
except ImportError:
 from publication_style import apply_publication_style, BLUE, ORANGE, GREEN
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/checkpoints/v13_heat_paper_aligned'
d=pd.read_csv(OUT/'cell_results.csv');a=d[d.task_set_accepted.astype(str).str.lower().eq('true')].copy()
summary=d.groupby('level',as_index=False).agg(task_set_acceptance=('task_set_accepted','mean'),service_failure=('mandatory_service_failure_rate','mean'),
 complete_image_dice=('mean_complete_image_dice','mean'),recall=('pixel_defect_recall','mean'),image_miss=('image_complete_miss_rate','mean'))
thermal=a.groupby('level',as_index=False).agg(accepted_cell_atp=('average_temperature_c','mean'),accepted_cell_iit=('iit_celsius_seconds','mean'),
 accepted_cell_peak=('peak_temperature_c','max'),accepted_cells=('task_set_accepted','size'))
summary=summary.merge(thermal,on='level',how='left');summary.to_csv(OUT/'level_summary.csv',index=False)
period=d.groupby(['level','period_ms'],as_index=False).agg(task_set_acceptance=('task_set_accepted','mean'),service_failure=('mandatory_service_failure_rate','mean'),
 complete_image_dice=('mean_complete_image_dice','mean'),recall=('pixel_defect_recall','mean'),image_miss=('image_complete_miss_rate','mean'))
period.to_csv(OUT/'period_level_summary.csv',index=False)
audit={'cells':int(len(d)),'accepted_cells':int(d.task_set_accepted.sum()),'terminal_residual_max':float(d.mandatory_terminal_residual.abs().max()),
 'rejected_cells_with_finite_thermal':int((~d.task_set_accepted & d.average_temperature_c.notna()).sum()),
 'accepted_admitted_dmr_max':float(a.mandatory_admitted_dmr.max())}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(summary.to_string(index=False));print(json.dumps(audit,indent=2))

apply_publication_style();fig,ax=plt.subplots(2,2,figsize=(8.8,6.2),constrained_layout=True)
levels=summary.level
ax[0,0].plot(levels,summary.task_set_acceptance,'o-',color=BLUE)
ax[0,1].plot(levels,summary.complete_image_dice,'o-',color=ORANGE,label='Complete-image Dice')
ax[0,1].plot(levels,summary.recall,'s--',color=GREEN,label='Pixel recall')
ax[1,0].plot(levels,summary.accepted_cell_atp,'o-',color=BLUE)
ax[1,1].plot(levels,summary.accepted_cell_iit,'o-',color=ORANGE)
for a0,y in zip(ax.flat,['Task-set acceptance','Deadline-aware quality','ATP in accepted cells (°C)','IIT in accepted cells (°C·s)']):
 a0.set_xlabel('Fixed CS-DNN level');a0.set_ylabel(y);a0.set_xticks(levels);a0.grid(True,alpha=.25)
ax[0,0].set_ylim(-.03,1.03);ax[0,1].set_ylim(-.03,1.03);ax[0,1].legend(frameon=False)
fig.savefig(OUT/'v13_heat_paper_aligned_boundary.pdf',bbox_inches='tight');fig.savefig(OUT/'v13_heat_paper_aligned_boundary.png',dpi=300,bbox_inches='tight')
