"""Verify the saved corpus against its recorded hashes, counts and work groups."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

PROJECT=Path(__file__).resolve().parent


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as file:
        while block:=file.read(1024*1024):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=PROJECT/'data')
    args=parser.parse_args();data=args.data_dir
    hashes=json.loads((data/'checksums.json').read_text())
    for name,expected in hashes.items():
        file=data/name
        assert file.stat().st_size==expected['bytes'],name
        assert digest(file)==expected['sha256'],name
    statistics=json.loads((data/'statistics.json').read_text())
    rows=[json.loads(line) for line in (data/'manifest.jsonl').read_text().splitlines()]
    assert len(rows)==statistics['documents']
    assert len({r['author_key'] for r in rows})==statistics['authors']
    assert len({r['normalized_text_sha256'] for r in rows})==len(rows)
    groups=collections.defaultdict(set)
    for row in rows:groups[row['work_group']].add(row['split'])
    assert all(len(parts)==1 for parts in groups.values())
    tokenizer=Tokenizer.from_file(str(data/'tokenizer.json'))
    assert tokenizer.get_vocab_size()==8192
    eos=tokenizer.token_to_id('<|endoftext|>');unk=tokenizer.token_to_id('<|unk|>')
    for split,counts in statistics['splits'].items():
        tokens=np.memmap(data/(split+'.bin'),mode='r',dtype='<u2')
        selected=[r for r in rows if r['split']==split]
        assert len(tokens)==counts['tokens']==sum(r['tokens'] for r in selected)
        assert int(tokens.max())<8192
        assert np.count_nonzero(tokens==eos)==counts['documents']
        assert np.count_nonzero(tokens==unk)==0
        offset=0
        for row in selected:
            assert row['token_offset']==offset
            offset+=row['tokens']
    print(json.dumps({'passed':True,'files_checked':len(hashes),'documents':len(rows),
        'authors':statistics['authors'],'tokens':statistics['tokens'],
        'no_work_group_crosses_splits':True,'unknown_tokens':0},ensure_ascii=False))


if __name__=='__main__':main()
