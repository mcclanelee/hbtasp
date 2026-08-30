"""Paper-aligned non-preemptive HEAT adaptation for fixed CS-DNN paths."""
from __future__ import annotations
import csv, hashlib, json, math, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from experiments.initial_manuscript_event_replay import build_periodic_releases
from experiments.initial_manuscript_hbtasp import GPU_WCET, TAMB, VOLTAGES, end_temperature
from experiments.initial_r2_8_metrics import load_confusion, score_trace
from experiments.mandatory_terminal_registry import MandatoryTerminalRegistry
from experiments.run_v9_thermal_augmented_factorial import reconstruct_thermal

ROOT=Path(__file__).resolve().parents[1]
POOL_PATH=ROOT/'experiments/checkpoints/v8_calibrated_final_factorial/calibrated_histogram_test_pool.json'
CONFUSION_PATH=ROOT/'mask_replay_final_test_shared/mask_confusion_by_level.csv'
OUT=ROOT/'experiments/checkpoints/v13_heat_paper_aligned'; RESULT=OUT/'cell_results.csv'
PERIODS=(100,150,200,250,300); LINES=(4,6,8,10); LEVELS=(1,2,3,4,5)
SEEDS=(101,202,303,404,505,606,707,808,909,1010); EPOCHS=1000
_POOL=None;_CONFUSION=None

def sha256(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def init_worker():
 global _POOL,_CONFUSION
 _POOL=json.loads(POOL_PATH.read_text(encoding='utf-8'));_CONFUSION=load_confusion(CONFUSION_PATH)

def level_duration(core,level_idx,v_idx):
 return GPU_WCET[core][level_idx]*0.8/VOLTAGES[v_idx]

def choose_voltage(n_tasks,core,level_idx,period):
 # Paper Eq. (4), with no migrating inference.  Shares are measured at Fmax.
 cmax=level_duration(core,level_idx,len(VOLTAGES)-1)
 fopt=n_tasks*cmax/period
 normalized=[v/VOLTAGES[-1] for v in VOLTAGES]
 return next((i for i,f in enumerate(normalized) if f+1e-12>=fopt),len(VOLTAGES)-1)

def run_cell(level,period_ms,lines,seed):
 li=level-1;period=period_ms/1000.;releases=build_periodic_releases(_POOL,period_ms,lines,EPOCHS,seed)
 registry=MandatoryTerminalRegistry.from_release_batches(releases.values())
 n=lines*4;cmax=[level_duration(j,li,4) for j in range(2)]
 cap=[int(math.floor((period+1e-12)/c)) for c in cmax]
 # Algorithm 3: all identical ratios prefer the faster first core.
 n0=min(n,cap[0]);n1=n-n0;accepted=n1<=cap[1]
 trace=[]
 if not accepted:
  for jobs in releases.values():
   for job in jobs:
    if job.mandatory: registry.transition(job.uid,'PRE_REJECT')
    trace.append({'event':'mandatory_infeasible' if job.mandatory else 'dispatch_infeasible',
      'time':job.release,'uid':job.uid,'image_uid':job.image_uid,'source_image_id':job.source_image_id,
      'region_index':job.region_index,'mandatory':job.mandatory,'weight':job.weight,'terminal_state':'PRE_REJECT' if job.mandatory else None})
  terminal=registry.finalize();mask,_=score_trace(trace,_POOL,_CONFUSION)
  return {'configuration':f'HEAT-adapted-L{level}','level':level,'period_ms':period_ms,'lines':lines,'seed':seed,'epochs':EPOCHS,
    'task_set_accepted':False,'cluster_capacity_gpu0':cap[0],'cluster_capacity_gpu1':cap[1],
    'assigned_per_slice_gpu0':n0,'assigned_per_slice_gpu1':n1,**terminal,**mask,
    'average_temperature_c':float('nan'),'iit_celsius_seconds':float('nan'),'peak_temperature_c':float('nan'),'observation_horizon_s':float('nan')}
 volts=[choose_voltage(n0,0,li,period),choose_voltage(n1,1,li,period)]
 free=[0.,0.];temps=[float(TAMB),float(TAMB)];tt=[0.,0.]
 for jobs in releases.values():
  assigned=(jobs[:n0],jobs[n0:])
  for core,core_jobs in enumerate(assigned):
   for job in core_jobs:
    start=max(job.release,free[core])
    if start>tt[core]:temps[core]=end_temperature(temps[core],VOLTAGES[0],start-tt[core]);tt[core]=start
    dur=level_duration(core,li,volts[core]);finish=start+dur;st=temps[core];et=end_temperature(st,VOLTAGES[volts[core]],dur)
    trace.append({'event':'dispatch','time':start,'processor':core,'uid':job.uid,'image_uid':job.image_uid,'source_image_id':job.source_image_id,
      'region_index':job.region_index,'mandatory':job.mandatory,'weight':job.weight,'level':level,'voltage':VOLTAGES[volts[core]],'temperature':st})
    late=finish>job.deadline+1e-12
    if job.mandatory: registry.transition(job.uid,'LATE_COMPLETE' if late else 'ON_TIME')
    trace.append({'event':'complete','time':finish,'processor':core,'uid':job.uid,'image_uid':job.image_uid,'source_image_id':job.source_image_id,
      'region_index':job.region_index,'mandatory':job.mandatory,'weight':job.weight,'level':level,'voltage':VOLTAGES[volts[core]],'temperature':et,
      'deadline_miss':late,'terminal_state':('LATE_COMPLETE' if late else 'ON_TIME') if job.mandatory else None})
    free[core]=finish;temps[core]=et;tt[core]=finish
 terminal=registry.finalize();mask,_=score_trace(trace,_POOL,_CONFUSION);thermal=reconstruct_thermal(trace)
 return {'configuration':f'HEAT-adapted-L{level}','level':level,'period_ms':period_ms,'lines':lines,'seed':seed,'epochs':EPOCHS,
  'task_set_accepted':True,'cluster_capacity_gpu0':cap[0],'cluster_capacity_gpu1':cap[1],
  'assigned_per_slice_gpu0':n0,'assigned_per_slice_gpu1':n1,'selected_voltage_gpu0':VOLTAGES[volts[0]],'selected_voltage_gpu1':VOLTAGES[volts[1]],
  **terminal,**mask,**thermal}

def write(rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 tmp=RESULT.with_suffix('.tmp')
 with tmp.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 tmp.replace(RESULT)

def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=pd.read_csv(RESULT).to_dict('records') if RESULT.exists() else []
 done={(int(r['level']),int(r['period_ms']),int(r['lines']),int(r['seed'])) for r in rows};keys=[(l,p,n,s) for l in LEVELS for p in PERIODS for n in LINES for s in SEEDS]
 start=time.time()
 with ProcessPoolExecutor(max_workers=4,initializer=init_worker) as ex:
  fs={ex.submit(run_cell,*k):k for k in keys if k not in done}
  for fu in as_completed(fs):
   rows.append(fu.result());rows.sort(key=lambda r:(int(r['level']),int(r['period_ms']),int(r['lines']),int(r['seed'])));write(rows)
   (OUT/'checkpoint.json').write_text(json.dumps({'status':'complete' if len(rows)==len(keys) else 'running','completed_cells':len(rows),'total_cells':len(keys),
    'updated_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds_this_run':time.time()-start,
    'protocol':{'paper_contract':'HEAT_PAPER_CONTRACT_V1.md','periods_ms':PERIODS,'lines':LINES,'levels':LEVELS,'seeds':SEEDS,'epochs':EPOCHS,
    'inference':'non-preemptive; migration disabled','infeasible_task_set':'reject entire cell','thermal_reporting':'accepted cells only'}},indent=2),encoding='utf-8')
   if len(rows)%50==0:print(f'[{len(rows)}/{len(keys)}]',flush=True)
 if rows:
  cp=json.loads((OUT/'checkpoint.json').read_text(encoding='utf-8'))
  cp['status']='complete' if len(rows)==len(keys) else 'running'
  cp['sha256']={'pool':sha256(POOL_PATH),'confusion':sha256(CONFUSION_PATH),
   'runner':sha256(Path(__file__)),'paper_contract':sha256(ROOT/'experiments/HEAT_PAPER_CONTRACT_V1.md'),
   'event_kernel':sha256(ROOT/'experiments/initial_manuscript_event_replay.py'),
   'hbtasp_primitives':sha256(ROOT/'experiments/initial_manuscript_hbtasp.py'),'results':sha256(RESULT)}
  (OUT/'checkpoint.json').write_text(json.dumps(cp,indent=2),encoding='utf-8')
if __name__=='__main__':main()
