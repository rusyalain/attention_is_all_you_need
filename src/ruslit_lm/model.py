"""Decoder-only Transformer. All weights start from random initialization."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 8192
    context_length: int = 512
    layers: int = 8
    width: int = 512
    heads: int = 8
    dropout: float = 0.1
    attention: str = 'sdpa'

    def __post_init__(self):
        if self.width % self.heads:
            raise ValueError('Model width must be divisible by the number of heads')
        if self.attention not in ('manual','sdpa'):
            raise ValueError('attention must be manual or sdpa')


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.method = config.attention
        self.dropout_probability = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.ones(config.context_length, config.context_length, dtype=torch.bool).tril()
        self.register_buffer('causal_mask', mask, persistent=False)

    def queries_keys_values(self, x):
        batch, length, _ = x.shape
        projections = self.qkv(x).chunk(3, dim=-1)
        return tuple(t.reshape(batch,length,self.heads,self.head_width).transpose(1,2)
                     for t in projections)

    def weights(self, x):
        q,k,_ = self.queries_keys_values(x)
        scores = q @ k.transpose(-2,-1) / math.sqrt(self.head_width)
        # Future positions receive -infinity before softmax, hence exactly zero weight.
        scores = scores.masked_fill(~self.causal_mask[:x.shape[1],:x.shape[1]], float('-inf'))
        return scores.softmax(dim=-1)

    def forward(self, x):
        batch, length, width = x.shape
        q,k,v = self.queries_keys_values(x)
        probability = self.dropout_probability if self.training else 0.0
        if self.method == 'manual':
            scores = q @ k.transpose(-2,-1) / math.sqrt(self.head_width)
            scores = scores.masked_fill(~self.causal_mask[:length,:length], float('-inf'))
            attention = F.dropout(scores.softmax(dim=-1),p=probability,training=self.training)
            result = attention @ v
        else:
            # The notebook derives the identical matrix calculation explicitly.
            result = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=probability)
        result = result.transpose(1,2).contiguous().reshape(batch,length,width)
        return self.dropout(self.projection(result))


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.width)
        self.attention = CausalSelfAttention(config)
        self.norm_mlp = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(nn.Linear(config.width,4*config.width),nn.GELU(),
                                 nn.Linear(4*config.width,config.width),nn.Dropout(config.dropout))

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size,config.width)
        self.positions = nn.Embedding(config.context_length,config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.width)
        self.apply(self._initialize)
        # Scale residual outputs to keep their initial variance stable across blocks.
        for block in self.blocks:
            nn.init.normal_(block.attention.projection.weight,std=.02/math.sqrt(2*config.layers))
            nn.init.normal_(block.mlp[2].weight,std=.02/math.sqrt(2*config.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=.02)
            if isinstance(module,nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, token_ids, targets=None, *, last_logits_only=False):
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError('Input exceeds the configured context length')
        positions = torch.arange(token_ids.shape[1],device=token_ids.device)
        x = self.dropout(self.tokens(token_ids)+self.positions(positions))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        if last_logits_only:
            x = x[:,-1:,:]
        # Weight tying uses the same matrix for input embeddings and output logits.
        logits = F.linear(x,self.tokens.weight)
        loss = None if targets is None else F.cross_entropy(logits.flatten(0,1),targets.flatten())
        return logits,loss

    @torch.no_grad()
    def attention_maps(self, token_ids):
        self.eval()
        positions = torch.arange(token_ids.shape[1],device=token_ids.device)
        x = self.tokens(token_ids)+self.positions(positions)
        result=[]
        for block in self.blocks:
            result.append(block.attention.weights(block.norm_attention(x)).cpu())
            x = block(x)
        return result

    @torch.no_grad()
    def generate(self, token_ids, *, max_new_tokens=160,temperature=.8,top_k=50,
                 top_p=.95,eos_id=0,seed=42):
        self.eval()
        # Seed the CPU sampler; backend rounding can still change a continuation.
        generator = torch.Generator(device='cpu').manual_seed(seed)
        result = token_ids.clone()
        for _ in range(max_new_tokens):
            context = result[:,-self.config.context_length:]
            logits,_ = self(context,last_logits_only=True)
            scores = logits[0,-1].float().cpu()
            if temperature<=0:
                next_id = int(scores.argmax())
            else:
                scores = scores / temperature
                if top_k:
                    threshold = torch.topk(scores,min(top_k,scores.numel())).values[-1]
                    scores[scores<threshold] = float('-inf')
                if top_p<1:
                    ordered,indices = scores.sort(descending=True)
                    remove = ordered.softmax(dim=-1).cumsum(dim=-1)>top_p
                    remove[1:] = remove[:-1].clone();remove[0] = False
                    scores[indices[remove]] = float('-inf')
                next_id = int(torch.multinomial(scores.softmax(dim=-1),1,generator=generator))
            if next_id==eos_id:
                break
            token = torch.tensor([[next_id]],device=result.device,dtype=torch.long)
            result = torch.cat((result,token),dim=1)
        return result
