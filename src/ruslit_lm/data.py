"""Packed uint16 tokens, shuffled blocks, and exact resumable epoch boundaries."""
from pathlib import Path
import numpy as np
import torch


class EpochBatches:
    def __init__(self,path,batch_size=16,context_length=512,seed=20261109):
        self.path=Path(path)
        self.tokens=np.memmap(self.path,dtype='<u2',mode='r')
        self.batch_size=batch_size
        self.context_length=context_length
        self.num_blocks=(len(self.tokens)-1)//context_length
        self.steps_per_epoch=self.num_blocks//batch_size
        if self.steps_per_epoch<1:
            raise ValueError('The split is too short for one training batch')
        self.rng=np.random.default_rng(seed)
        self.order=self.rng.permutation(self.num_blocks)
        self.cursor=0
        self.epoch=0
        self.offsets=np.arange(context_length+1)

    def next(self,device='mps'):
        if self.cursor+self.batch_size>len(self.order):
            self.epoch+=1
            self.order=self.rng.permutation(self.num_blocks)
            self.cursor=0
        starts=self.order[self.cursor:self.cursor+self.batch_size]*self.context_length
        self.cursor+=self.batch_size
        chunk=self.tokens[starts[:,None]+self.offsets].astype(np.int64)
        x=torch.from_numpy(chunk[:,:-1].copy()).to(device)
        y=torch.from_numpy(chunk[:,1:].copy()).to(device)
        return x,y

    @property
    def epoch_progress(self):
        return self.epoch+self.cursor/(self.steps_per_epoch*self.batch_size)

    def state_dict(self):
        return {'order':torch.from_numpy(self.order.copy()),'cursor':self.cursor,
                'epoch':self.epoch,'rng':self.rng.bit_generator.state}

    def load_state_dict(self,state):
        self.order=state['order'].cpu().numpy().copy()
        self.cursor=state['cursor'];self.epoch=state['epoch']
        self.rng.bit_generator.state=state['rng']


def evaluation_batches(path,*,batch_size=16,context_length=512,batches=32,seed=1337,device='mps'):
    tokens=np.memmap(Path(path),dtype='<u2',mode='r')
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(tokens)-context_length-1,size=batch_size*batches)
    offsets=np.arange(context_length+1)
    for index in range(batches):
        chosen=starts[index*batch_size:(index+1)*batch_size]
        chunk=tokens[chosen[:,None]+offsets].astype(np.int64)
        yield torch.from_numpy(chunk[:,:-1].copy()).to(device),torch.from_numpy(chunk[:,1:].copy()).to(device)
