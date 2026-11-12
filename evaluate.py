"""Score held-out books and compare the trained decoder with statistical baselines."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from tokenizers import Tokenizer

PROJECT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/'src'))
from ruslit_lm import Config,Decoder


def batches(path,batch_size=16,context=512):
    tokens=np.memmap(path,dtype='<u2',mode='r')
    blocks=(len(tokens)-1)//context
    offsets=np.arange(context+1)
    for first in range(0,blocks,batch_size):
        starts=np.arange(first,min(blocks,first+batch_size))*context
        array=tokens[starts[:,None]+offsets].astype(np.int64)
        yield array[:,:-1].copy(),array[:,1:].copy()


def fit_baselines(path,vocabulary=8192):
    tokens=np.memmap(path,dtype='<u2',mode='r')
    unigrams=np.zeros(vocabulary,dtype=np.uint64)
    bigrams=np.zeros((vocabulary,vocabulary),dtype=np.uint32)
    for first in range(0,len(tokens),1_000_000):
        chunk=np.asarray(tokens[first:first+1_000_000])
        unigrams+=np.bincount(chunk,minlength=vocabulary).astype(np.uint64)
    for first in range(0,len(tokens)-1,1_000_000):
        end=min(len(tokens)-1,first+1_000_000)
        indices=np.asarray(tokens[first:end],dtype=np.int64)*vocabulary+tokens[first+1:end+1]
        np.add.at(bigrams.reshape(-1),indices,1)
    return unigrams,bigrams


@torch.no_grad()
def score(model,path,unigrams,bigrams,device='mps',alpha=.05):
    model.eval();nll=0.;unigram_nll=0.;bigram_nll=0.;positions=0
    vocabulary=len(unigrams);unigram_prob=(unigrams+1)/(unigrams.sum()+vocabulary)
    row_counts=bigrams.sum(axis=1,dtype=np.uint64)
    for x,y in batches(path):
        inputs=torch.from_numpy(x).to(device);targets=torch.from_numpy(y).to(device)
        _,loss=model(inputs,targets)
        n=y.size;nll+=float(loss.item())*n;positions+=n
        unigram_nll-=float(np.log(unigram_prob[y]).sum())
        probabilities=(bigrams[x,y].astype(np.float64)+alpha)/(row_counts[x]+alpha*vocabulary)
        bigram_nll-=float(np.log(probabilities).sum())
    result={'scored_target_tokens':positions}
    for label,value in [('transformer',nll),('unigram',unigram_nll),('bigram',bigram_nll)]:
        loss=value/positions
        result[label]={'cross_entropy_nats_per_token':loss,'perplexity':math.exp(loss)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=PROJECT/'data')
    parser.add_argument('--checkpoint',type=Path,default=PROJECT/'checkpoints'/'best.pt')
    parser.add_argument('--baseline-cache',type=Path)
    parser.add_argument('--output-dir',type=Path,default=PROJECT/'runs'/'evaluation',
                        help='Directory for evaluation.json and unigram_counts.npy')
    args=parser.parse_args();torch.set_num_threads(4)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    expected=hashlib.sha256((args.data_dir/'tokenizer.json').read_bytes()).hexdigest()
    assert expected==state['tokenizer_sha256']
    model=Decoder(Config(**state['config']));model.load_state_dict(state['model']);model.to(device).eval()
    tokenizer=Tokenizer.from_file(str(args.data_dir/'tokenizer.json'))
    started=time.perf_counter()
    if args.baseline_cache and args.baseline_cache.exists():
        cache=np.load(args.baseline_cache);unigrams=cache['unigrams'];bigrams=cache['bigrams']
        if int(unigrams.sum())!=(args.data_dir/'train.bin').stat().st_size//2:
            raise ValueError('Baseline cache has a different training-token count')
    else:unigrams,bigrams=fit_baselines(args.data_dir/'train.bin')
    # Selected on the validation split in the original experiment.
    alpha=.05
    result={'checkpoint':args.checkpoint.name,'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'step':state['step'],'training_tokens_at_checkpoint':state['training_tokens'],'device':device,
        'tokenizer_sha256':expected,'vocabulary':8192,
        'train_bin_sha256':hashlib.sha256((args.data_dir/'train.bin').read_bytes()).hexdigest(),
        'baseline_smoothing':{'unigram_additive':1.0,'bigram_additive':alpha},'splits':{}}
    for split in ('validation','test'):
        result['splits'][split]=score(model,args.data_dir/(split+'.bin'),unigrams,bigrams,device,alpha=alpha)
        print(json.dumps({'split':split,**result['splits'][split]},ensure_ascii=False),flush=True)
    prompts=['Поздним вечером Иван Петрович вернулся домой и увидел, что',
             'Она остановилась у окна. За стеклом медленно',
             '— Позвольте, — сказал старик, — я хочу рассказать вам',
             'На другой день маленький город проснулся',
             'Письмо лежало на столе уже третий день.']
    samples=[]
    for prompt in prompts:
        ids=torch.tensor([tokenizer.encode(prompt).ids],device=device)
        output=model.generate(ids,max_new_tokens=180,temperature=.8,top_k=50,top_p=.95,seed=42)
        text=tokenizer.decode(output[0].tolist())
        samples.append({'prompt':prompt,'text':text,'seed':42,'temperature':.8,'top_k':50,'top_p':.95,
                        'text_manually_edited':False})
        print(json.dumps({'sample':samples[-1]},ensure_ascii=False),flush=True)
    result['samples']=samples;result['evaluation_seconds']=time.perf_counter()-started
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    # These counts are corpus facts, useful for the notebook's frequency illustration.
    np.save(args.output_dir/'unigram_counts.npy',unigrams)


if __name__=='__main__':main()
