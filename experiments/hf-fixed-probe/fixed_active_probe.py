#!/usr/bin/env python3
from __future__ import annotations
import contextlib, hashlib, json, math, os, pathlib, queue, re, shutil, signal, statistics, subprocess, sys, threading, time
from typing import Any
import torch

MODEL=pathlib.Path('/model'); JOB=pathlib.Path('/job'); RESULTS=pathlib.Path('/results')
RUN_ID=os.environ.get('RUN_ID','hf-a100-eggroll-fixed-probe-step1777105-20260810')
OUT=RESULTS/RUN_ID; LOCAL=pathlib.Path('/tmp/agillm43_fixed_probe'); LM=LOCAL/'model'
SOURCE_MOUNT=JOB/'agillm43_singlefile_intelligence_v24_eggroll_fixed_tokens.py'
TRAIN_MOUNT=JOB/'fixed_train_tokens_800.json'; LAUNCHER_MOUNT=MODEL/'launch_eggroll_experiment.sh'
SOURCE=LM/SOURCE_MOUNT.name; TRAIN=LOCAL/TRAIN_MOUNT.name; LAUNCHER=LM/LAUNCHER_MOUNT.name
CKPT_DIR=LM/'base_checkpoint'; CKPT_NAME='pretrain_step01777105_from01739752_20260809T2256Z.pt'
CKPT_SHA='72ee3cf3ae4e3dedb22f16d8c2d21c80c66e0bdcd2001f606f030886cc98e2af'
SOURCE_SHA='e3c89682ed04e8d5419cf30e814577b50def3346d3546a4eb27f37acd78f19cf'
TRAIN_SHA='381f82fe92dacaf0621d881bdd951e7f5c76d32d403dbe41382cd2d45cd4bbb3'
TOKEN_SHA='bbde62ac4424d40585ae8df66c9369d3606e3dc134b4c75e56ca0493f1b97b42'
COMMITS=int(os.environ.get('COMMITS','14')); BATCH=int(os.environ.get('BATCH_SIZE','4')); TIMEOUT=int(os.environ.get('BRANCH_TIMEOUT_S','360'))
OUT.mkdir(parents=True,exist_ok=True); LOCAL.mkdir(parents=True,exist_ok=True)

def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False,default=str)+'\n'); os.replace(tmp,path)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def finite(xs):
 out=[]
 for x in xs:
  try:x=float(x)
  except:continue
  if math.isfinite(x):out.append(x)
 return out
def mean(xs):
 x=finite(xs); return statistics.fmean(x) if x else None
def median(xs):
 x=finite(xs); return statistics.median(x) if x else None

def prep():
 if LM.exists(): shutil.rmtree(LM)
 LM.mkdir(parents=True); shutil.copy2(SOURCE_MOUNT,SOURCE); shutil.copy2(TRAIN_MOUNT,TRAIN); shutil.copy2(LAUNCHER_MOUNT,LAUNCHER)
 t=time.monotonic(); subprocess.run(['cp','-a',str(MODEL/'base_checkpoint'),str(LM)],check=True)
 r={'created_at':now(),'copy_s':time.monotonic()-t,'source_sha256':sha(SOURCE),'train_sha256':sha(TRAIN),'checkpoint_sha256':sha(CKPT_DIR/CKPT_NAME)}
 r['passed']=r['source_sha256']==SOURCE_SHA and r['train_sha256']==TRAIN_SHA and r['checkpoint_sha256']==CKPT_SHA
 if not r['passed']: raise RuntimeError('input integrity failure '+repr(r))
 dump(OUT/'copy_receipt.json',r); return r

def ckpt_meta():
 x=torch.load(CKPT_DIR/CKPT_NAME,map_location='cpu',weights_only=False); s=x.get('skeleton',x); d=s.get('continuation_dblock_resume_state',{})
 return {'step':int(s.get('step',0)),'seen_tok':int(s.get('seen_tok',0)),'committed_step':int(d.get('committed_step',d.get('step',0)) or 0)}

def command():
 import shlex
 s=re.sub(r'\\\n\s*',' ',LAUNCHER.read_text()); m=re.search(r'\bexec\s+(python3\b.*)',s,re.S)
 if not m: raise RuntimeError('launcher command missing')
 c=shlex.split(m.group(1)); i=c.index('train'); c[0]=sys.executable; c[i-1]=str(SOURCE); return c
def remove_value(c,f):
 while f in c:
  i=c.index(f); del c[i:i+2]
def setv(c,f,v): remove_value(c,f); c.extend([f,str(v)])
def flag(c,f,on):
 while f in c:c.remove(f)
 if on:c.append(f)

def make(name,active,meta):
 c=command(); sd=LOCAL/'runs'/name; shutil.rmtree(sd,ignore_errors=True); sd.mkdir(parents=True)
 setv(c,'--resume',CKPT_DIR/CKPT_NAME); setv(c,'--save_dir',sd); setv(c,'--steps',COMMITS); setv(c,'--batch_size',BATCH)
 setv(c,'--source','fixed-token-contract'); setv(c,'--fixed_tokens_file',TRAIN); setv(c,'--fixed_tokens_sha256',TRAIN_SHA)
 flag(c,'--fixed_tokens_cycle',False); flag(c,'--no-fixed_tokens_cycle',True)
 setv(c,'--save_every_sec',999999); setv(c,'--delta_every_steps',0); setv(c,'--delta_every_sec',0); setv(c,'--max_ckpts',1)
 setv(c,'--disk_free_floor_gb',4); setv(c,'--cuda_max_reserved_mib',74000); setv(c,'--cuda_min_free_mib',2048)
 setv(c,'--dblock_stop_after_commits',meta['committed_step']+100000); setv(c,'--dblock_log_every',1)
 event=OUT/(name+'.eggroll.jsonl'); event.unlink(missing_ok=True); setv(c,'--eggroll_log_jsonl',event); flag(c,'--eggroll_dry_run',False)
 if active:
  setv(c,'--eggroll_every_steps',1); setv(c,'--eggroll_warmup_steps',0); setv(c,'--eggroll_population',16); setv(c,'--eggroll_population_chunk',16)
  setv(c,'--eggroll_rank',1); setv(c,'--eggroll_sigma',0.1); setv(c,'--eggroll_lr',0.0001); setv(c,'--eggroll_tokens',128)
  setv(c,'--eggroll_score_tokens',32); setv(c,'--eggroll_batch',1); setv(c,'--eggroll_guard_crops',4); setv(c,'--eggroll_seed',4317)
  setv(c,'--eggroll_target_router',-1); setv(c,'--eggroll_max_targets',1); setv(c,'--eggroll_accept_tolerance',0.0)
  setv(c,'--eggroll_min_pair_signal',1e-7); setv(c,'--eggroll_max_update_rms_ratio',0.001); flag(c,'--eggroll_fail_fast',True); expected=COMMITS
 else:
  setv(c,'--eggroll_every_steps',0); setv(c,'--eggroll_target_router',-1); flag(c,'--eggroll_fail_fast',False); expected=0
 return c,sd,event,expected

DB=re.compile(r'\[dblock\]\s+step=(\d+)\s+block=(\d+)\s+obj=([a-z]+).*?\bloss=([-+0-9.eE]+).*?\braw_sum=([-+0-9.eE]+).*?peak_alloc=([-+0-9.eE]+)GB\s+peak_reserved=([-+0-9.eE]+)GB')
AN=re.compile(r'\[(dblock-anchor|dblock-sat-anchor|dblock-nat-anchor)\]\s+step=(\d+)\s+raw_ce=([-+0-9.eE]+).*?weighted=([-+0-9.eE]+).*?crop_sha256=([0-9a-f]{16,64})')
ER=re.compile(r'\[eggroll\]\s+event=(\d+)\s+step=(\d+)\s+status=([^\s]+)')
FIX=re.compile(r'\[fixed-tokens\]\s+(\{.*\})'); DIAG='[dblock-local-diag] '

def run(name,active,c,event,expected):
 log=OUT/(name+'.log'); env=os.environ.copy(); env.update({'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':'0','AGILLM43_EXPERIMENTAL_DETERMINISTIC':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','NVIDIA_TF32_OVERRIDE':'0','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':'0'})
 p=subprocess.Popen(c,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,start_new_session=True); q=queue.Queue()
 def reader():
  try:
   for line in p.stdout:q.put(line)
  finally:q.put(None)
 threading.Thread(target=reader,daemon=True).start(); t=time.monotonic(); first=last=None; steps=set(); events=set(); done=False; killed=False; timed=False
 with log.open('w',buffering=1) as f:
  while True:
   elapsed=time.monotonic()-t
   if elapsed>TIMEOUT and p.poll() is None:
    timed=True; killed=True; os.killpg(p.pid,signal.SIGKILL)
   try:x=q.get(timeout=.5)
   except queue.Empty:x='__EMPTY__'
   if x is None:done=True
   elif x!='__EMPTY__':
    clean=x.rstrip('\n'); f.write(f'{elapsed:.6f}\t{clean}\n')
    for line in clean.replace('\r','\n').splitlines():
     m=DB.search(line)
     if m:
      steps.add(int(m.group(1))); first=elapsed if first is None else first; last=elapsed
     m=ER.search(line)
     if m:events.add(int(m.group(1)))
     if m or DB.search(line) or '[fixed-tokens]' in line or '[experimental-determinism]' in line: print(f'[{name} +{elapsed:7.2f}s] {line}',flush=True)
    if len(steps)>=COMMITS and len(events)>=expected and p.poll() is None:
     killed=True; print(f'[{name}] collected commits={len(steps)} eggroll={len(events)}; hard kill before checkpoint save',flush=True); os.killpg(p.pid,signal.SIGKILL)
   if p.poll() is not None and done:break
 rc=p.wait(); r={'name':name,'active':active,'started_at':now(),'runtime_s':time.monotonic()-t,'returncode':rc,'intentional_kill':killed,'timed_out':timed,'commits':len(steps),'events':len(events),'expected_events':expected,'train_window_s':None if first is None or last is None else last-first,'command':c}
 dump(OUT/(name+'.process.json'),r)
 if timed or len(steps)!=COMMITS or len(events)!=expected:raise RuntimeError('branch incomplete '+repr(r))
 return r

def parse(path):
 d=[]; a=[]; g=[]; fixed=[]; deterministic=False
 for physical in path.read_text(errors='replace').splitlines():
  try:et,raw=physical.split('\t',1); elapsed=float(et)
  except:raw=physical; elapsed=float('nan')
  for line in raw.replace('\r','\n').splitlines():
   m=DB.search(line)
   if m:d.append({'step':int(m.group(1)),'block':int(m.group(2)),'objective':m.group(3),'loss':float(m.group(4)),'raw_sum':float(m.group(5)),'elapsed_s':elapsed})
   m=AN.search(line)
   if m:a.append({'kind':m.group(1),'step':int(m.group(2)),'raw_ce':float(m.group(3)),'weighted':float(m.group(4)),'crop_sha256':m.group(5),'elapsed_s':elapsed})
   if DIAG in line:
    with contextlib.suppress(Exception):
     x=json.loads(line.split(DIAG,1)[1]); x['elapsed_s']=elapsed; g.append(x)
   m=FIX.search(line)
   if m:
    with contextlib.suppress(Exception):fixed.append(json.loads(m.group(1)))
   if '[experimental-determinism] enabled' in line:deterministic=True
 return {'dblocks':sorted(d,key=lambda x:x['step']),'anchors':sorted(a,key=lambda x:(x['step'],x['kind'])),'diags':sorted(g,key=lambda x:int(x.get('step',-1))),'fixed':fixed,'deterministic':deterministic}

def events(path):
 out=[]
 if path.exists():
  for line in path.read_text(errors='replace').splitlines():
   with contextlib.suppress(Exception):
    x=json.loads(line)
    if isinstance(x,dict):out.append(x)
 return out

def stream_cmp(x,y):
 x={int(r['step']):r for r in x['diags'] if 'step' in r}; y={int(r['step']):r for r in y['diags'] if 'step' in r}; common=sorted(set(x)&set(y)); fields=('batch_sha256','block','objective','input_sha256_by_objective','target_sha256_by_objective'); bad=[]
 for s in common:
  diff={k:[x[s].get(k),y[s].get(k)] for k in fields if x[s].get(k)!=y[s].get(k)}
  if diff:bad.append({'step':s,'differences':diff})
 return {'left_rows':len(x),'right_rows':len(y),'matched_steps':len(common),'mismatches':bad,'passed':len(common)==COMMITS and len(x)==COMMITS and len(y)==COMMITS and not bad}

def numeric(x,y):
 x={r['step']:r for r in x['dblocks']}; y={r['step']:r for r in y['dblocks']}; common=sorted(set(x)&set(y)); dl=[y[s]['loss']-x[s]['loss'] for s in common]
 xa={(r['step'],r['kind'],r['crop_sha256']):r for r in x.values()} if False else {}
 return {'matched_steps':len(common),'mean_delta_right_minus_left':mean(dl),'median_delta_right_minus_left':median(dl),'mean_abs_delta':mean(abs(z) for z in dl),'max_abs_delta':max((abs(z) for z in dl),default=None),'deltas':[{'step':s,'delta':z} for s,z in zip(common,dl)]}

def event_summary(rows):
 status={}; sig=[]; search=[]; guard=[]; flips=[]; update=[]
 for r in rows:
  st=str(r.get('status','unknown')); status[st]=status.get(st,0)+1; sig.append(r.get('pair_signal_rms')); flips.append(r.get('route_flip_fraction_mean')); update.append(r.get('update_rms_ratio'))
  b=r.get('baseline_search_ce'); c=r.get('search_after_ce')
  if isinstance(b,(int,float)) and isinstance(c,(int,float)):search.append(float(b)-float(c))
  b=r.get('baseline_guard_ce'); c=r.get('guard_after_ce')
  if isinstance(b,(int,float)) and isinstance(c,(int,float)):guard.append(float(b)-float(c))
  elif isinstance(b,list) and isinstance(c,list):
   guard.extend(float(u)-float(v) for u,v in zip(b,c) if isinstance(u,(int,float)) and isinstance(v,(int,float)))
 accepted=sum(1 for r in rows if r.get('accepted') is True or str(r.get('status','')).lower()=='accepted')
 rejected=sum(1 for r in rows if r.get('accepted') is False or str(r.get('status','')).lower()=='rejected')
 return {'events':len(rows),'statuses':status,'accepted':accepted,'rejected':rejected,'nonzero_signal':sum(1 for x in finite(sig) if abs(x)>0),'pair_signal_mean':mean(sig),'pair_signal_median':median(sig),'search_improvement_mean':mean(search),'guard_improvement_mean':mean(guard),'route_flip_mean':mean(flips),'update_rms_ratio_mean':mean(update),'details':rows}

def main():
 print('FIXED_PROBE_STARTED='+now(),flush=True); copy=prep(); meta=ckpt_meta(); dump(OUT/'checkpoint_metadata.json',meta)
 specs=[('baseline_a',False),('baseline_b',False),('eggroll_allrouters_sigma01',True)]; runs={}; parsed={}; ev={}
 for name,active in specs:
  c,sd,event,expected=make(name,active,meta); print('BRANCH_BEGIN='+json.dumps({'name':name,'active':active,'commits':COMMITS,'batch':BATCH}),flush=True)
  runs[name]=run(name,active,c,event,expected); parsed[name]=parse(OUT/(name+'.log')); ev[name]=events(event); print('BRANCH_END='+name,flush=True)
 ab=stream_cmp(parsed['baseline_a'],parsed['baseline_b']); aa=stream_cmp(parsed['baseline_a'],parsed['eggroll_allrouters_sigma01'])
 nr=numeric(parsed['baseline_a'],parsed['baseline_b']); na=numeric(parsed['baseline_a'],parsed['eggroll_allrouters_sigma01'])
 ma={r['step']:r['loss'] for r in parsed['baseline_a']['dblocks']}; mb={r['step']:r['loss'] for r in parsed['baseline_b']['dblocks']}; me={r['step']:r['loss'] for r in parsed['eggroll_allrouters_sigma01']['dblocks']}; common=sorted(set(ma)&set(mb)&set(me))
 active_delta=[me[s]-(ma[s]+mb[s])/2 for s in common]; base_noise=[abs(ma[s]-mb[s]) for s in common]
 active_better_both=sum(1 for s in common if me[s]<min(ma[s],mb[s])); active_worse_both=sum(1 for s in common if me[s]>max(ma[s],mb[s]))
 windows=[runs['baseline_a']['train_window_s'],runs['baseline_b']['train_window_s']]; bw=mean(windows); aw=runs['eggroll_allrouters_sigma01']['train_window_s']; overhead=None if not bw else aw/bw
 es=event_summary(ev['eggroll_allrouters_sigma01']); receipts_ok=all(parsed[n]['fixed'] and parsed[n]['fixed'][-1].get('file_sha256')==TRAIN_SHA and parsed[n]['fixed'][-1].get('token_int64le_sha256')==TOKEN_SHA and parsed[n]['deterministic'] for n,_ in specs)
 valid=ab['passed'] and aa['passed'] and receipts_ok; noise=mean(base_noise) or 0.0; improvement=-(mean(active_delta) or 0.0)
 if not valid:classification='invalid'
 elif overhead is not None and overhead<=1.0 and improvement>noise:classification='promising_speedup'
 elif es['accepted']>0:classification='mechanistic_signal_not_speedup'
 else:classification='no_speedup_detected'
 summary={'schema':'agillm43.eggroll.fixed_probe.v1','run_id':RUN_ID,'finished_at':now(),'checkpoint':meta,'copy_receipt':copy,'commits_per_branch':COMMITS,'batch_size':BATCH,'tokens_per_branch':COMMITS*BATCH*2048,'contracts':{'source_sha256':SOURCE_SHA,'train_sha256':TRAIN_SHA,'token_sha256':TOKEN_SHA,'checkpoint_sha256':CKPT_SHA},'input_match':{'baseline_repeat':ab,'active_vs_baseline':aa,'receipts_ok':receipts_ok,'valid':valid},'baseline_numerical_noise':nr,'active_vs_baseline_a':na,'paired_step_analysis':{'steps':len(common),'mean_abs_baseline_repeat_loss_noise':mean(base_noise),'max_abs_baseline_repeat_loss_noise':max(base_noise,default=None),'mean_active_loss_delta_vs_baseline_mean':mean(active_delta),'median_active_loss_delta_vs_baseline_mean':median(active_delta),'active_better_than_both_baselines':active_better_both,'active_worse_than_both_baselines':active_worse_both,'rows':[{'step':s,'baseline_a':ma[s],'baseline_b':mb[s],'active':me[s],'active_minus_baseline_mean':me[s]-(ma[s]+mb[s])/2} for s in common]},'runtime':{'baseline_mean_train_window_s':bw,'active_train_window_s':aw,'active_over_baseline_ratio':overhead,'branches':runs},'eggroll':es,'classification':classification}
 dump(OUT/'summary.json',summary); report=f'''# AGILLM 4.3 EGGROLL fixed-token probe\n\nRun `{RUN_ID}` used two EGGROLL-disabled repeats and one active branch from checkpoint step {meta['step']}. Every branch consumed {COMMITS} commits, batch {BATCH}, and {COMMITS*BATCH*2048:,} tokens.\n\n- Exact token/input/schedule contracts passed: **{valid}**\n- Mean baseline repeat loss noise: `{mean(base_noise)}`\n- Mean active loss delta versus baseline mean: `{mean(active_delta)}`\n- Active better than both baselines: `{active_better_both}/{len(common)}` steps\n- Active worse than both baselines: `{active_worse_both}/{len(common)}` steps\n- Active/baseline training-window ratio: `{overhead}`\n- EGGROLL accepted events: `{es['accepted']}/{es['events']}`\n- Mean immediate search improvement: `{es['search_improvement_mean']}`\n- Mean guard improvement: `{es['guard_improvement_mean']}`\n\n**Classification: `{classification}`**\n\nThis is a short matched pretraining probe, not a claim about long-run convergence. It is designed to decide whether active router EGGROLL earns a larger experiment while keeping production v22 untouched.\n'''; (OUT/'REPORT.md').write_text(report)
 print('FIXED_PROBE_SUMMARY='+json.dumps(summary,sort_keys=True),flush=True); print('RESULTS_PATH='+str(OUT),flush=True); return 0

if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as e:
  f={'schema':'agillm43.eggroll.fixed_probe.failure.v1','failed_at':now(),'type':type(e).__name__,'message':str(e)}
  with contextlib.suppress(Exception):dump(OUT/'failure.json',f)
  print('FATAL='+json.dumps(f,sort_keys=True),flush=True); raise
