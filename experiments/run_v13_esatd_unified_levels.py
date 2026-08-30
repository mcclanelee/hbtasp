from __future__ import annotations
import csv,hashlib,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from experiments.overall_baseline_adapters import run_esatd_fixed
from experiments.initial_r2_8_metrics import load_confusion,score_trace
from experiments.run_v9_thermal_augmented_factorial import reconstruct_thermal
ROOT=Path(__file__).resolve().parents[1];POOL_PATH=ROOT/'experiments/checkpoints/v8_calibrated_final_factorial/calibrated_histogram_test_pool.json';CONF=ROOT/'mask_replay_final_test_shared/mask_confusion_by_level.csv';OUT=ROOT/'experiments/checkpoints/v13_esatd_unified_levels';RESULT=OUT/'cell_results.csv'
LEVELS=(1,2,3,4,5);PERIODS=(100,150,200,250,300);LINES=(4,6,8,10);SEEDS=(101,202,303,404,505,606,707,808,909,1010);EPOCHS=1000;_P=None;_C=None
def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def init():
 global _P,_C
 _P=json.loads(POOL_PATH.read_text(encoding='utf-8'));_C=load_confusion(CONF)
def cell(k):
 l,p,n,s=k;summary,trace=run_esatd_fixed(_P,p,n,EPOCHS,s,l);mask,_=score_trace(trace,_P,_C);thermal=reconstruct_thermal(trace);return {'configuration':f'ESATD-L{l}','level':l,'period_ms':p,'lines':n,'seed':s,'epochs':EPOCHS,**summary,**mask,**thermal}
def write(rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 t=RESULT.with_suffix('.tmp')
 with t.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 t.replace(RESULT)
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=pd.read_csv(RESULT).to_dict('records') if RESULT.exists() else [];done={(int(r['level']),int(r['period_ms']),int(r['lines']),int(r['seed'])) for r in rows};keys=[(l,p,n,s) for l in LEVELS for p in PERIODS for n in LINES for s in SEEDS];start=time.time()
 with ProcessPoolExecutor(max_workers=4,initializer=init) as ex:
  fs={ex.submit(cell,k):k for k in keys if k not in done}
  for fu in as_completed(fs):
   rows.append(fu.result());rows.sort(key=lambda r:(int(r['level']),int(r['period_ms']),int(r['lines']),int(r['seed'])));write(rows);(OUT/'checkpoint.json').write_text(json.dumps({'status':'complete' if len(rows)==len(keys) else 'running','completed_cells':len(rows),'total_cells':len(keys),'updated_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds_this_run':time.time()-start},indent=2),encoding='utf-8')
   if len(rows)%50==0:print(f'[{len(rows)}/{len(keys)}]',flush=True)
 if rows:
  cp={'status':'complete' if len(rows)==len(keys) else 'running','completed_cells':len(rows),'total_cells':len(keys),
   'updated_utc':datetime.now(timezone.utc).isoformat(),
   'protocol':{'levels':LEVELS,'periods_ms':PERIODS,'lines':LINES,'seeds':SEEDS,'epochs':EPOCHS,'release':'periodic'},
   'sha256':{'pool':sha256(POOL_PATH),'confusion':sha256(CONF),'runner':sha256(Path(__file__)),
    'adapter':sha256(ROOT/'experiments/overall_baseline_adapters.py'),'event_kernel':sha256(ROOT/'experiments/initial_manuscript_event_replay.py'),'results':sha256(RESULT)}}
  (OUT/'checkpoint.json').write_text(json.dumps(cp,indent=2),encoding='utf-8')
if __name__=='__main__':main()
