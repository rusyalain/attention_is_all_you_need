"""Offline text completion with the trained literary Transformer."""
from __future__ import annotations
import argparse
import hashlib
import math
from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer

PROJECT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/'src'))
from ruslit_lm import Config,Decoder


def load_model(checkpoint=None,device=None):
    device=device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint=Path(checkpoint or PROJECT/'checkpoints'/'best.pt')
    if not checkpoint.is_file():
        raise FileNotFoundError('Model weights are missing. Run git lfs pull in the repository directory.')
    with checkpoint.open('rb') as source:
        if source.read(40).startswith(b'version https://git-lfs.github.com/spec/'):
            raise FileNotFoundError('Only the Git LFS pointer is present. Run git lfs pull to download the model weights.')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    tokenizer_path=PROJECT/'tokenizer.json'
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()!=state['tokenizer_sha256']:
        raise ValueError('Tokenizer does not match the checkpoint')
    model=Decoder(Config(**state['config']))
    model.load_state_dict(state['model'])
    return model.to(device).eval(),Tokenizer.from_file(str(tokenizer_path))


def complete(prompt,*,checkpoint=None,temperature=.8,top_k=50,top_p=.95,max_new_tokens=160,seed=42,device=None):
    if not 0<top_p<=1:raise ValueError('top_p must be in (0,1]')
    if not math.isfinite(temperature) or temperature<0:raise ValueError('temperature must be finite and nonnegative')
    if top_k<0:raise ValueError('top_k must be nonnegative')
    if max_new_tokens<1:raise ValueError('max_new_tokens must be positive')
    model,tokenizer=load_model(checkpoint,device)
    ids=tokenizer.encode(prompt).ids or [tokenizer.token_to_id('<|endoftext|>')]
    token_ids=torch.tensor([ids],device=next(model.parameters()).device,dtype=torch.long)
    result=model.generate(token_ids,max_new_tokens=max_new_tokens,temperature=temperature,
        top_k=top_k,top_p=top_p,seed=seed,eos_id=tokenizer.token_to_id('<|endoftext|>'))
    return tokenizer.decode(result[0].tolist())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt',default='Она остановилась у окна. За стеклом медленно')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--temperature',type=float,default=.8)
    parser.add_argument('--top-k',type=int,default=50)
    parser.add_argument('--top-p',type=float,default=.95)
    parser.add_argument('--tokens',type=int,default=160)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['mps','cpu'])
    args=parser.parse_args()
    print(complete(args.prompt,checkpoint=args.checkpoint,temperature=args.temperature,top_k=args.top_k,
        top_p=args.top_p,max_new_tokens=args.tokens,seed=args.seed,device=args.device))
