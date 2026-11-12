"""Verify causality and manual/SDPA agreement on the chosen actual backend."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import torch
PROJECT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/'src'))
from ruslit_lm import Config,Decoder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=PROJECT/'checkpoints'/'best.pt')
    parser.add_argument('--device',choices=['mps','cpu'],default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--output-json',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(71)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=Decoder(Config(**state['config']));model.load_state_dict(state['model'])
    model=model.to(args.device).eval()
    ids=torch.randint(0,model.config.vocab_size,(2,model.config.context_length),device=args.device)
    modified=ids.clone();modified[:,256:]=(modified[:,256:]+17)%model.config.vocab_size
    with torch.no_grad():
        logits,_=model(ids);changed,_=model(modified)
        causal_error=float((logits[:,:256]-changed[:,:256]).abs().max().item())
        for block in model.blocks:block.attention.method='manual'
        manual,_=model(ids)
        implementation_error=float((logits-manual).abs().max().item())
    result={'passed':causal_error<1e-5 and implementation_error<1e-3,
            'device':args.device,'checkpoint_step':state['step'],
            'input_shape':list(ids.shape),'future_change_starts_at':256,
            'causal_prefix_max_logit_error':causal_error,
            'manual_vs_sdpa_max_logit_error':implementation_error,
            'torch':torch.__version__,'cpu_fallback_enabled':False}
    if args.output_json:
        args.output_json.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    assert result['passed'],result


if __name__=='__main__':main()
