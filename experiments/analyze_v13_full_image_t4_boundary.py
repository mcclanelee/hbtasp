"""Aggregate and plot the full-image T4 p99.5 real-time boundary."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

try:
    from experiments.publication_style import apply_publication_style, BLUE, ORANGE, GREEN
except ImportError:
    from publication_style import apply_publication_style, BLUE, ORANGE, GREEN

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/checkpoints/v13_full_image_t4_boundary'
D=pd.read_csv(OUT/'cell_results.csv')
H=pd.read_csv(ROOT/'experiments/checkpoints/v11_unified_overall/cell_results.csv')
H=H[H.configuration.eq('Full-HBTASP')]
S=pd.read_csv(ROOT/'experiments/checkpoints/v14_strict_all_regions_matched/cell_results.csv')

summary=D.groupby(['configuration','period_ms'],as_index=False).agg(
    service_failure=('mandatory_service_failure_rate','mean'),
    completion=('on_time_completion_rate','mean'),
    image_miss=('deadline_aware_image_complete_miss_rate','mean'),
    dice=('deadline_aware_complete_image_dice','mean'),
    recall=('deadline_aware_pixel_recall','mean'),
    atp=('average_temperature_c','mean'),iit=('iit_celsius_seconds','mean'))
summary.to_csv(OUT/'period_summary.csv',index=False)
overall=D.groupby('configuration',as_index=False).mean(numeric_only=True)
overall.to_csv(OUT/'overall_summary.csv',index=False)
strict=S.groupby('period_ms',as_index=False).agg(
    image_service_failure=('full_image_on_time_acceptance',lambda x: 1-x.mean()),
    image_miss=('image_complete_miss_rate','mean'),
    dice=('mean_complete_image_dice','mean'),recall=('pixel_defect_recall','mean'),
    admitted_dmr=('admitted_deadline_violation_ratio','mean'),
    peak=('peak_modeled_temperature','max'))
strict.to_csv(OUT/'strict_hbtasp_period_summary.csv',index=False)

apply_publication_style()
fig,ax=plt.subplots(2,2,figsize=(9.0,6.5),constrained_layout=True)
colors={'DeepLabV3-MobileNetV3-Full':BLUE,'YOLOv8n-Full':GREEN}
labels={'DeepLabV3-MobileNetV3-Full':'DeepLab full image','YOLOv8n-Full':'YOLOv8n full image'}
for c,g in summary.groupby('configuration'):
    ax[0,0].plot(g.period_ms,g.service_failure,'o-',color=colors[c],label=labels[c])
    ax[0,1].plot(g.period_ms,g.image_miss,'o-',color=colors[c],label=labels[c])
h=H.groupby('period_ms',as_index=False).mean(numeric_only=True)
ax[0,0].plot(h.period_ms,h.mandatory_service_failure_rate,'o-',color=ORANGE,label='HBTASP partial service')
ax[0,1].plot(h.period_ms,h.image_complete_miss_rate,'o-',color=ORANGE,label='HBTASP partial service')
ax[0,0].plot(strict.period_ms,strict.image_service_failure,'s--',color='#7A5195',label='HBTASP all-region')
ax[0,1].plot(strict.period_ms,strict.image_miss,'s--',color='#7A5195',label='HBTASP all-region')
deep=summary[summary.configuration.eq('DeepLabV3-MobileNetV3-Full')]
ax[1,0].plot(deep.period_ms,deep.dice,'o-',color=BLUE,label='DeepLab full image')
ax[1,0].plot(h.period_ms,h.mean_complete_image_dice,'o-',color=ORANGE,label='HBTASP partial service')
ax[1,0].plot(strict.period_ms,strict.dice,'s--',color='#7A5195',label='HBTASP all-region')
ax[1,1].plot(deep.period_ms,deep.recall,'o-',color=BLUE,label='DeepLab full image')
ax[1,1].plot(h.period_ms,h.pixel_defect_recall,'o-',color=ORANGE,label='HBTASP partial service')
ax[1,1].plot(strict.period_ms,strict.recall,'s--',color='#7A5195',label='HBTASP all-region')
for a,y in zip(ax.flat,['Required-unit service failure','Image complete-miss rate','Deadline-aware image Dice','Deadline-aware pixel recall']):
    a.set_xlabel('Period (ms)');a.set_ylabel(y);a.set_ylim(-.03,1.03);a.grid(True,alpha=.25)
handles,legend=ax[0,0].get_legend_handles_labels()
fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.5,1.045),ncol=4,frameon=False)
fig.savefig(OUT/'v14_full_image_t4_realtime_boundary.pdf',bbox_inches='tight')
fig.savefig(OUT/'v14_full_image_t4_realtime_boundary.png',dpi=300,bbox_inches='tight')

audit={
 'cells':int(len(D)), 'terminal_residual_max':float(D.mandatory_terminal_residual.abs().max()),
 'deep_images':int(D[D.configuration.str.startswith('DeepLab')].released_images.sum()),
 'yolo_images':int(D[D.configuration.str.startswith('YOLO')].released_images.sum()),
 'nominal_iit_max':float(D.iit_celsius_seconds.max()),
 'strict_cells':int(len(S)),
 'strict_terminal_residual_max':float((
     S.total_regions-S.pre_execution_rejections-S.admitted_deadline_violations-
     (S.completed-S.deadline_misses)).abs().max()),
 'strict_admitted_dmr_max':float(S.admitted_deadline_violation_ratio.max()),
 'strict_image_service_failure_mean':float(1-S.full_image_on_time_acceptance.mean()),
}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(summary.to_string(index=False));print(json.dumps(audit,indent=2))
