"""Train the local literary decoder from random weights on Apple MPS."""
from __future__ import annotations
import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
PROJECT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/'src'))
import numpy as np
import torch
from tokenizers import Tokenizer
from ruslit_lm import Config,Decoder
from ruslit_lm.data import EpochBatches,evaluation_batches


def atomic_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    temporary.replace(path)


def to_cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:to_cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(to_cpu(v) for v in value)
    return value


@torch.no_grad()
def evaluate(model,path,batches=32):
    model.eval();losses=[]
    for x,y in evaluation_batches(path,batches=batches):
        _,loss=model(x,y);losses.append(loss.detach())
    result=float(torch.stack(losses).mean().item())
    model.train()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=PROJECT/'data')
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'runs'/'experiment')
    parser.add_argument('--hours',type=float,default=5.5)
    parser.add_argument('--deadline-unix',type=float,help='Optional earlier absolute deadline for a shared experiment budget')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--prepare-data',action='store_true',help='Download, clean, split and tokenize the source texts when the corpus is missing')
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--eval-interval',type=int,default=500)
    parser.add_argument('--eval-batches',type=int,default=32)
    parser.add_argument('--peak-lr',type=float,default=6e-4)
    parser.add_argument('--min-lr',type=float,default=3e-5)
    parser.add_argument('--warmup',type=int,default=400)
    parser.add_argument('--attention',choices=['manual','sdpa'],default='sdpa')
    args=parser.parse_args()
    if not 0<args.hours<=6:parser.error('Training time must be between zero and six hours')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS is unavailable')
    torch.set_num_threads(4);torch.manual_seed(20261109)
    data=args.data_dir.resolve()
    if args.prepare_data and not (data/'statistics.json').exists():
        subprocess.run([sys.executable,str(PROJECT/'data_pipeline'/'prepare.py'),'--output-dir',str(data)],check=True)
    statistics=json.loads((data/'statistics.json').read_text())
    if statistics['status']!='complete':raise RuntimeError('Corpus preparation is incomplete')
    tokenizer_hash=hashlib.sha256((data/'tokenizer.json').read_bytes()).hexdigest()
    run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    checkpoints=run/'checkpoints';checkpoints.mkdir(exist_ok=True)
    results=run/'results';results.mkdir(exist_ok=True)
    if (results/'metrics.jsonl').exists() and not args.resume:
        raise RuntimeError('This run already contains metrics. Resume it or choose a new --run-dir.')
    clock_path=run/'training_clock.json'
    if args.resume:
        clock=json.loads(clock_path.read_text())
    else:
        start=time.time()
        deadline=min(start+args.hours*3600,args.deadline_unix) if args.deadline_unix else start+args.hours*3600
        if deadline<=start:raise ValueError('Training deadline has already passed')
        clock={'started_unix':start,'deadline_unix':deadline,'budget_hours':(deadline-start)/3600}
        atomic_json(clock_path,clock)
    config=Config(attention=args.attention)
    model=Decoder(config).to('mps').train()
    assert model.parameter_count==29_676_544
    decay=[p for p in model.parameters() if p.ndim>=2]
    no_decay=[p for p in model.parameters() if p.ndim<2]
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':.1},
                                {'params':no_decay,'weight_decay':0}],
                               lr=args.peak_lr,betas=(.9,.95),foreach=True)
    batches=EpochBatches(data/'train.bin')
    tokenizer=Tokenizer.from_file(str(data/'tokenizer.json'))
    shutil.copy2(data/'tokenizer.json',run/'tokenizer.json')
    step=0;best_loss=float('inf');best_step=0;stale=0;previous_elapsed=0.0
    planned_steps=30000
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=True)
        if state['tokenizer_sha256']!=tokenizer_hash:raise RuntimeError('Tokenizer differs from resume checkpoint')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        batches.load_state_dict(state['data_state'])
        torch.set_rng_state(state['torch_rng'])
        if 'mps_rng' in state:torch.mps.set_rng_state(state['mps_rng'])
        step=state['step'];best_loss=state['best_validation_loss'];best_step=state['best_step']
        stale=state.get('stale_evaluations',0);previous_elapsed=state['elapsed_seconds']
        planned_steps=state.get('planned_steps',30000)
    started=time.perf_counter();segment_start=started;segment_step=step
    log_start=started;log_step=step;loss_acc=[];stop_requested=False
    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    metadata={'model':dataclasses.asdict(config),'parameters':model.parameter_count,
        'training':vars(args)|{'data_dir':str(data),'run_dir':str(run),'resume':str(args.resume) if args.resume else None},
        'tokenizer_sha256':tokenizer_hash,'corpus_statistics_sha256':hashlib.sha256((data/'statistics.json').read_bytes()).hexdigest(),
        'torch':torch.__version__,'device':'mps','dtype':'float32','seed':20261109,
        'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'pretrained_weights_used':False}
    atomic_json(run/'config.json',metadata)
    def elapsed():return time.time()-clock['started_unix']
    def log(event,**values):
        record={'event':event,'step':step,'training_tokens':step*8192,'epoch':batches.epoch_progress,
                'elapsed_seconds':elapsed(),**values}
        with (results/'metrics.jsonl').open('a') as file:file.write(json.dumps(record,ensure_ascii=False)+'\n')
        atomic_json(results/'latest.json',record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
        return record
    def checkpoint(name,resume=False):
        torch.mps.synchronize()
        state={'model':to_cpu(model.state_dict()),'config':dataclasses.asdict(config),'step':step,
            'best_validation_loss':best_loss,'best_step':best_step,'elapsed_seconds':elapsed(),
            'tokenizer_sha256':tokenizer_hash,'training_tokens':step*8192}
        if resume:
            state.update(optimizer=to_cpu(optimizer.state_dict()),data_state=batches.state_dict(),
                torch_rng=torch.get_rng_state(),planned_steps=planned_steps,stale_evaluations=stale)
            if hasattr(torch.mps,'get_rng_state'):state['mps_rng']=torch.mps.get_rng_state().cpu()
        path=checkpoints/name;temporary=path.with_suffix(path.suffix+'.tmp')
        torch.save(state,temporary);temporary.replace(path)
    def samples(label):
        prompts=['Поздним вечером Иван Петрович вернулся домой и увидел, что',
                 'Она остановилась у окна. За стеклом медленно']
        rows=[]
        for prompt in prompts:
            ids=torch.tensor([tokenizer.encode(prompt).ids],device='mps')
            generated=model.generate(ids,max_new_tokens=96,temperature=.8,top_k=50,top_p=.95,seed=42)
            rows.append({'prompt':prompt,'text':tokenizer.decode(generated[0].tolist()),'seed':42,'temperature':.8})
        model.train()
        with (results/'samples.jsonl').open('a') as file:
            file.write(json.dumps({'label':label,'step':step,'training_tokens':step*8192,'samples':rows},ensure_ascii=False)+'\n')
    log('started',parameters=model.parameter_count,train_tokens=statistics['splits']['train']['tokens'],
        steps_per_epoch=batches.steps_per_epoch,hours_limit=clock['budget_hours'])
    if step==0:
        samples('random_initialization')
        initial=evaluate(model,data/'validation.bin',args.eval_batches)
        log('validation',validation_loss=initial,perplexity=math.exp(initial),learning_rate=0)
    segment_start=time.perf_counter();log_start=segment_start;log_step=step
    stop_reason='time_budget'
    failed=False
    try:
        while time.time()<clock['deadline_unix']-15:
            if stop_requested:stop_reason='requested_stop';break
            if args.max_steps and step>=args.max_steps:stop_reason='max_steps';break
            if step<args.warmup:
                lr=args.peak_lr*(step+1)/args.warmup
            else:
                fraction=min(1,max(0,(step-args.warmup)/max(1,planned_steps-args.warmup)))
                lr=args.min_lr+.5*(args.peak_lr-args.min_lr)*(1+math.cos(math.pi*fraction))
            for group in optimizer.param_groups:group['lr']=lr
            x,y=batches.next()
            optimizer.zero_grad(set_to_none=True)
            _,loss=model(x,y)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=False)
            optimizer.step();step+=1;loss_acc.append(loss.detach())
            if step%4==0:torch.mps.synchronize()
            if step%25==0:
                torch.mps.synchronize();now=time.perf_counter()
                mean_loss=float(torch.stack(loss_acc).mean().item());loss_acc=[]
                gradient_norm=float(norm.item())
                if not math.isfinite(mean_loss) or not math.isfinite(gradient_norm):raise RuntimeError('Non-finite training state')
                log('train',train_loss=mean_loss,learning_rate=lr,gradient_norm=gradient_norm,
                    training_tokens_per_second=(step-log_step)*8192/(now-log_start),
                    mps_driver_bytes=torch.mps.driver_allocated_memory())
                log_start=now;log_step=step
            if step==segment_step+100:
                torch.mps.synchronize()
                observed=(step-segment_step)/(time.perf_counter()-segment_start)
                remaining=max(0,clock['deadline_unix']-time.time())
                planned_steps=step+int(remaining*observed*.9)
                log('schedule_calibrated',planned_steps=planned_steps)
            if step%args.eval_interval==0:
                validation_loss=evaluate(model,data/'validation.bin',args.eval_batches)
                improved=validation_loss<best_loss-.001
                if improved:
                    best_loss=validation_loss;best_step=step;stale=0
                    checkpoint('best.pt')
                else:stale+=1
                log('validation',validation_loss=validation_loss,perplexity=math.exp(validation_loss),
                    best_validation_loss=best_loss,best_step=best_step,improved=improved,learning_rate=lr)
                checkpoint('latest.resume.pt',resume=True)
                if step%2500==0:samples('training_checkpoint')
                log_start=time.perf_counter();log_step=step
                if stale>=12 and step*8192>=30_000_000:
                    stop_reason='validation_plateau';break
    except BaseException as exc:
        failed=True
        log('failed',error=repr(exc));raise
    finally:
        if not failed and math.isfinite(best_loss):checkpoint('latest.resume.pt',resume=True)
    final_loss=evaluate(model,data/'validation.bin',args.eval_batches)
    if final_loss<best_loss:
        best_loss=final_loss;best_step=step;checkpoint('best.pt')
    # Keep resume metadata consistent when the final check improves between
    # regular evaluation steps, including a stop before the first interval.
    checkpoint('latest.resume.pt',resume=True)
    samples('final_weights')
    log('complete',stop_reason=stop_reason,final_validation_loss=final_loss,
        best_validation_loss=best_loss,best_step=best_step)
    atomic_json(results/'training_summary.json',{'status':'complete','stop_reason':stop_reason,
        'steps':step,'training_tokens':step*8192,'epochs':batches.epoch_progress,
        'elapsed_seconds':elapsed(),'best_validation_loss':best_loss,'best_step':best_step,
        'final_validation_loss':final_loss,'parameters':model.parameter_count})


if __name__=='__main__':main()
