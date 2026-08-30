"""Unified Overall figure with the paper-aligned HEAT adaptation."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
try:
 from experiments.publication_style import apply_publication_style, BLUE, GREEN, RED, ORANGE
except ImportError:
 from publication_style import apply_publication_style, BLUE, GREEN, RED, ORANGE

ROOT=Path(__file__).resolve().parents[1]
OLD=pd.read_csv(ROOT/'experiments/checkpoints/v11_unified_overall/cell_results.csv')
OLD=OLD[~OLD.configuration.eq('HEAT-L3')].copy()
H=pd.read_csv(ROOT/'experiments/checkpoints/v13_heat_paper_aligned/cell_results.csv')
H=H[H.level.eq(3)].copy();H['configuration']='HEAT-adapted-L3'
H=H.rename(columns={'mean_complete_image_dice':'mean_complete_image_dice'})
D=pd.concat([OLD,H],ignore_index=True,sort=False)
HEAT100_PATH=ROOT/'experiments/checkpoints/v15_heat_100ms_feasible_domain/summary.csv'
HEAT100=pd.read_csv(HEAT100_PATH) if HEAT100_PATH.exists() else pd.DataFrame()
OUT=ROOT/'experiments/checkpoints/v14_unified_overall_corrected_heat';OUT.mkdir(parents=True,exist_ok=True)
D.to_csv(OUT/'cell_results.csv',index=False)
configs=['Static-V','ESATD-L3','HEAT-adapted-L3','Full-HBTASP'];labels=['Static-V','ESATD--L3','HEAT--L3 (task-set rejection)','HBTASP'];colors=[BLUE,GREEN,RED,ORANGE];marks=['s','^','D','o']
def agg(metric):
 s=D.groupby(['configuration','period_ms','seed'])[metric].mean().reset_index();m=s.groupby(['configuration','period_ms'])[metric].mean().unstack(0)
 h=s.groupby(['configuration','period_ms'])[metric].apply(lambda x:0 if len(x)<2 or stats.sem(x,nan_policy='omit')==0 else stats.t.ppf(.975,x.notna().sum()-1)*stats.sem(x,nan_policy='omit')).unstack(0);return m,h
apply_publication_style();fig,axes=plt.subplots(2,2,figsize=(11.2,8.0));panels=[('mandatory_service_failure_rate','Method-specific endpoint (%)',100),('average_temperature_c',r'Average modeled temperature ($^\circ$C)',1),('iit_celsius_seconds',r'IIT ($^\circ$C$\cdot$s)',1),('mean_complete_image_dice','Complete-image Dice',1)];periods=[100,150,200,250,300];handles=[]
for pi,(ax,(metric,y,scale)) in enumerate(zip(axes.flat,panels)):
 m,h=agg(metric)
 for c,l,col,mk in zip(configs,labels,colors,marks):
  yv=(scale*m[c].reindex(periods)).astype(float).to_numpy()
  ev=(scale*h[c].reindex(periods)).astype(float).to_numpy()
  art=ax.errorbar(periods,yv,yerr=ev,color=col,marker=mk,lw=2.15,ms=6.5,
                  capsize=3.5,capthick=1.2,elinewidth=1.2,label=l)
  if pi==0:handles.append(art)
 ax.set_xlabel('Period / deadline (ms)');ax.set_ylabel(y);ax.grid(True,alpha=.25);ax.text(-.11,1.03,f'({chr(97+pi)})',transform=ax.transAxes,fontweight='bold')
 if pi in (0,2):ax.set_yscale('symlog',linthresh=.1)
 # At 100 ms, every HEAT-L3 task set is rejected by the capacity test.
 # Its thermal quantities are undefined rather than zero, so show that state
 # explicitly instead of silently starting the thermal curves at 150 ms.
 if pi in (1,2):
  ymin = 30.2 if pi == 1 else 0.06
  ax.plot([100],[ymin],marker='x',ms=8,mew=1.8,color=RED,linestyle='none',zorder=6)
  ax.annotate('rejected\n(no trajectory)',xy=(100,ymin),xytext=(8,8),
              textcoords='offset points',fontsize=10,color=RED,ha='left',va='bottom')
  # A separate 100-ms feasibility extension uses 1--3 lines.  Its open marker
  # supplies a real thermal observation without mixing that lighter-load grid
  # into the formal 4/6/8/10-line Overall mean.
  if not HEAT100.empty:
   col = 'accepted_cell_atp' if pi == 1 else 'accepted_cell_iit'
   feasible = HEAT100[HEAT100.task_set_acceptance.gt(0)][col].dropna()
   if len(feasible):
    value = float(feasible.mean())
    ax.plot([100],[value],marker='D',ms=6.5,mew=1.5,mfc='white',mec=RED,
            color=RED,linestyle='none',zorder=7)
    ax.annotate('1--3-line\nfeasible audit',xy=(100,value),xytext=(8,4),
                textcoords='offset points',fontsize=9.8,color=RED,ha='left',va='bottom')
fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,.995));fig.subplots_adjust(top=.91,bottom=.09,left=.1,right=.98,hspace=.34,wspace=.28)
fig.savefig(OUT/'v14_unified_overall_corrected_heat.pdf',bbox_inches='tight');fig.savefig(OUT/'v14_unified_overall_corrected_heat.png',dpi=300,bbox_inches='tight')
