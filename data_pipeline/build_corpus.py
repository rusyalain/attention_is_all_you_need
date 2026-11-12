"""Build, split and tokenize the collected fiction; never train a language model."""
from __future__ import annotations
import collections
import datetime as dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

os.environ.setdefault('RAYON_NUM_THREADS','8')
import numpy as np
import tokenizers
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
from corpus_common import norm,title_key,edition_title_key,clean_text
from genre_rules import exclusion_reason
import load_sources

import os
BASE=Path(os.environ.get('RUSLIT_SOURCE_CACHE',Path(__file__).resolve().parent/'cache')).resolve()
OUTPUT=Path(os.environ.get('RUSLIT_CORPUS_OUTPUT',Path(__file__).resolve().parents[2]/'russian-prose-corpus')).resolve()
OUTPUT.mkdir(parents=True,exist_ok=True)

def event(name,**values):
    print(json.dumps({'event':name,**values},ensure_ascii=False),flush=True)

def text_hash(text):
    return hashlib.sha256(norm(text).encode()).hexdigest()

def fingerprints(text):
    result={}
    for sentence in re.split(r'[.!?…]+[»"\)]*\s+|\n\s*\n',text):
        normalized=norm(sentence)
        if len(normalized)<70:continue
        result[hashlib.blake2b(normalized.encode(),digest_size=12).digest()]=len(normalized)
    return result

def deduplicate(records):
    excluded=[];by_title={}
    for row in sorted(records,key=lambda r:(r['source_priority'],r['source_record'])):
        row['text']=clean_text(row['text'])
        reason=exclusion_reason(row)
        if reason:
            excluded.append({k:v for k,v in row.items() if k!='text'}|{'reason':reason});continue
        excluded_title=(row['author_key']=='горький_м' and row['title_key'] in {
            'мещане','надне','несвоевременныемысли','песняобуревестнике','песняосоколе',
            'человек','городжелтогодьявола','заметкииздневникавоспоминания'})
        excluded_title=excluded_title or (row['author_key']=='аксаков_ст' and row['title_key']=='рассказыивоспоминанияохотникаоразныхохотах')
        dramatic_header=re.search(r'(?mi)^\s*действующие лица\s*[:.]?\s*$|\b(?:комедия|драма|трагедия|водевиль)\s+(?:в|во)\s+(?:\w+\s+){0,3}действ',row['text'][:5000])
        if excluded_title or dramatic_header:
            excluded.append({k:v for k,v in row.items() if k!='text'}|{'reason':'genre_qa_poetry_play_or_nonfiction'});continue
        # Exclude original French fiction rather than its later Russian translation.
        if row['author_key']=='толстой_ак' and title_key(row['title']) in {'семьявурдалака','встречачерезтристалет'}:
            excluded.append({k:v for k,v in row.items() if k!='text'}|{'reason':'original_written_in_french'});continue
        if row['title_key']=='подписикрисункам':
            excluded.append({k:v for k,v in row.items() if k!='text'}|{'reason':'captions_not_narrative_fiction'});continue
        if len(row['text'])<100:
            excluded.append({k:v for k,v in row.items() if k!='text'}|{'reason':'too_short'});continue
        key=(row['author_key'],edition_title_key(row['title']))
        previous=by_title.get(key)
        if previous is None:by_title[key]=row;continue
        # Choose an unabridged version; prefer the curated edition for similar lengths.
        if len(row['text'])>len(previous['text'])*1.15 or (len(row['text'])>=len(previous['text'])*.9 and row['source_priority']<previous['source_priority']):
            by_title[key]=row;dropped,kept=previous,row
        else:dropped,kept=row,previous
        excluded.append({k:v for k,v in dropped.items() if k!='text'}|{'reason':'same_author_and_title','kept_source_record':kept['source_record']})
    exact={};candidates=[]
    for row in sorted(by_title.values(),key=lambda r:(r['source_priority'],-len(r['text']),r['source_record'])):
        digest=text_hash(row['text'])
        if digest in exact:
            excluded.append({k:v for k,v in row.items() if k!='text'}|{'reason':'same_normalized_text','kept_source_record':exact[digest]['source_record']});continue
        row['normalized_text_sha256']=digest
        row['text_sha256']=hashlib.sha256(row['text'].encode()).hexdigest()
        row['document_id']=hashlib.sha256((row['author_key']+'|'+row['title']+'|'+row['text_sha256']).encode()).hexdigest()[:24]
        exact[digest]=row;candidates.append(row)
    parent={}
    def find(key):
        parent.setdefault(key,key)
        if parent[key]!=key:parent[key]=find(parent[key])
        return parent[key]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    kept=[];author_index=collections.defaultdict(lambda:collections.defaultdict(list))
    author_rows=collections.defaultdict(list)
    for row in candidates:author_rows[row['author_key']].append(row)
    overlap_links=[]
    for author,rows in sorted(author_rows.items()):
        index=author_index[author]
        for row in sorted(rows,key=lambda r:(-len(r['text']),r['source_priority'],r['source_record'])):
            features=fingerprints(row['text']);mass=sum(features.values())
            scores=collections.Counter();matches=collections.Counter()
            for fingerprint,weight in features.items():
                for other in index.get(fingerprint,[]):scores[other]+=weight;matches[other]+=1
            duplicate=None
            for other,score in scores.most_common():
                containment=score/max(1,mass)
                if containment>=.90 and matches[other]>=3:
                    duplicate=(other,containment);break
            if duplicate:
                other,containment=duplicate
                excluded.append({k:v for k,v in row.items() if k!='text'}|{'reason':'near_duplicate_or_contained_in_collection',
                    'kept_document_id':other,'sentence_containment':containment});continue
            doc=row['document_id'];find(doc)
            for other,score in scores.items():
                if score>=3000 and score/max(1,mass)>=.15:
                    union(doc,other);overlap_links.append({'a':doc,'b':other,'matched_characters':score})
            for fingerprint in features:index[fingerprint].append(doc)
            kept.append(row)
        event('deduplicated_author',author=author,input=len(rows),retained=sum(r['author_key']==author for r in kept))
    # Different volumes of the same work belong to the same data split.
    by_group={}
    for row in kept:
        group=(row['author_key'],row['title_key'])
        if group in by_group:union(row['document_id'],by_group[group])
        else:by_group[group]=row['document_id']
    for row in kept:row['work_group']=find(row['document_id'])
    return sorted(kept,key=lambda r:(r['author_key'],r['title'],r['document_id'])),excluded,overlap_links

def split_records(records):
    groups=collections.defaultdict(list)
    for row in records:groups[row['work_group']].append(row)
    split={}
    for group in groups:
        value=int(hashlib.sha256(('russian-prose-20261109|'+group).encode()).hexdigest()[:16],16)/2**64
        split[group]='test' if value<.05 else 'validation' if value<.10 else 'train'
    by_author=collections.defaultdict(list)
    for group,rows in groups.items():by_author[rows[0]['author_key']].append(group)
    for author,author_groups in by_author.items():
        if not any(split[g]=='train' for g in author_groups):
            group=max(author_groups,key=lambda g:sum(len(r['text']) for r in groups[g]))
            split[group]='train'
    for row in records:row['split']=split[row['work_group']]
    return groups

def training_chunks(records):
    for row in records:
        if row['split']!='train':continue
        text=row['text'];position=0
        while position<len(text):
            end=min(position+16384,len(text))
            if end<len(text):
                cut=text.rfind(' ',max(position+8192,end-512),end)
                if cut>position:end=cut+1
            yield text[position:end];position=end

def main():
    started=time.perf_counter()
    download=json.loads((BASE/'az-download-summary.json').read_text())
    if download['errors']:raise RuntimeError(f"Unresolved source downloads: {len(download['errors'])}")
    loaders=(load_sources.narrative,load_sources.rsd,load_sources.ruslit,load_sources.az)
    records=[];source_totals={}
    for loader in loaders:
        rows=loader();records.extend(rows)
        source_totals[loader.__name__]={'documents':len(rows),'utf8_bytes':sum(len(r['text'].encode()) for r in rows)}
        event('source_loaded',source=loader.__name__,**source_totals[loader.__name__])
    before=len(records);records,excluded,links=deduplicate(records)
    event('deduplication_complete',before=before,retained=len(records),excluded=len(excluded))
    groups=split_records(records)
    (BASE/'deduplication-exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
    (BASE/'overlap-links.json').write_text(json.dumps(links,ensure_ascii=False,indent=2))
    event('tokenizer_training_started',documents=sum(r['split']=='train' for r in records),vocabulary=8192)
    tokenizer=Tokenizer(models.BPE(unk_token='<|unk|>'))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=8192,min_frequency=2,show_progress=False,
        special_tokens=['<|endoftext|>','<|unk|>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),max_token_length=64)
    token_start=time.perf_counter();tokenizer.train_from_iterator(training_chunks(records),trainer=trainer)
    token_training_seconds=time.perf_counter()-token_start
    assert tokenizer.get_vocab_size()==8192
    tokenizer.save(str(OUTPUT/'tokenizer.json'))
    eos=tokenizer.token_to_id('<|endoftext|>');unk=tokenizer.token_to_id('<|unk|>')
    event('tokenizer_training_complete',seconds=token_training_seconds,vocabulary=tokenizer.get_vocab_size())
    bins={name:(OUTPUT/(name+'.bin')).open('wb') for name in ('train','validation','test')}
    split_stats={name:{'documents':0,'text_utf8_bytes':0,'characters':0,'tokens':0,'tokens_without_eos':0} for name in bins}
    author_stats={};manifest=[];unknown_tokens=0
    encode_start=time.perf_counter()
    try:
        with gzip.open(OUTPUT/'texts.jsonl.gz','wt',encoding='utf-8',compresslevel=6) as textfile:
            for index,row in enumerate(records):
                encoding=tokenizer.encode(row['text'],add_special_tokens=False);ids=encoding.ids
                unknown_tokens+=ids.count(unk)
                if index<10 or index%200==0:
                    assert tokenizer.decode(ids,skip_special_tokens=False)==row['text']
                packed=np.asarray(ids+[eos],dtype='<u2');packed.tofile(bins[row['split']])
                row['tokens_without_eos']=len(ids);row['tokens']=len(ids)+1
                row['characters']=len(row['text']);row['text_utf8_bytes']=len(row['text'].encode())
                row['token_offset']=split_stats[row['split']]['tokens']
                for key in ('text_utf8_bytes','characters','tokens','tokens_without_eos'):
                    split_stats[row['split']][key]+=row[key]
                split_stats[row['split']]['documents']+=1
                meta={k:v for k,v in row.items() if k!='text'};manifest.append(meta)
                textfile.write(json.dumps(row,ensure_ascii=False)+'\n')
                author=author_stats.setdefault(row['author_key'],{'author':row['author'],'documents':0,'text_utf8_bytes':0,
                    'tokens':0,'train_tokens':0,'validation_tokens':0,'test_tokens':0})
                author['documents']+=1;author['text_utf8_bytes']+=row['text_utf8_bytes'];author['tokens']+=row['tokens']
                author[row['split']+'_tokens']+=row['tokens']
                if index%100==0:event('encoded',documents=index+1,total=len(records))
    finally:
        for file in bins.values():file.close()
    encoding_seconds=time.perf_counter()-encode_start
    assert unknown_tokens==0
    for name,stats in split_stats.items():assert (OUTPUT/(name+'.bin')).stat().st_size==stats['tokens']*2
    # Verify the token count independently from the saved uint16 arrays.
    for name,stats in split_stats.items():
        array=np.memmap(OUTPUT/(name+'.bin'),dtype='<u2',mode='r')
        assert len(array)==stats['tokens'] and int(np.count_nonzero(array==eos))==stats['documents']
        assert int(array.max())<8192
    assert all(len({r['split'] for r in rows})==1 for rows in groups.values())
    assert len({r['normalized_text_sha256'] for r in records})==len(records)
    rate=14433.257747262063;planned=rate*.8;train=split_stats['train']['tokens']
    steps_per_epoch=(train-1)//8192
    core_keys={a['author_key'] for a in json.loads((BASE/'az-inventory.json').read_text())['authors']}
    core=[r for r in records if r['author_key'] in core_keys]
    statistics={
        'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'complete',
        'scope':'Original Russian narrative fiction: novels, novellas, short stories; prose fairy tales and fictional sketches included. No standalone poetry, plays, letters, diaries, criticism or journalism.',
        'coverage_note':'A measured union of the listed public sources and selected author catalogues; not a bibliographic claim of every historical work by every author.',
        'sources':source_totals,'records_before_deduplication':before,'documents':len(records),
        'authors':len(author_stats),'text_utf8_bytes':sum(r['text_utf8_bytes'] for r in records),
        'characters':sum(r['characters'] for r in records),'tokens':sum(r['tokens'] for r in records),
        'tokens_without_eos':sum(r['tokens_without_eos'] for r in records),'splits':split_stats,
        'excluded_records':len(excluded),'exclusion_reasons':dict(collections.Counter(r['reason'] for r in excluded)),
        'split_policy':'Seeded SHA256 assignment by work groups, with at least one training group per author; shared long passages and volumes stay together.',
        'core_catalogue_subset':{'authors':len({r['author_key'] for r in core}),'documents':len(core),
            'text_utf8_bytes':sum(r['text_utf8_bytes'] for r in core),'tokens':sum(r['tokens'] for r in core),
            'train_tokens':sum(r['tokens'] for r in core if r['split']=='train')},
        'tokenizer':{'kind':'ByteLevel BPE','vocab_size':8192,'trained_only_on':'train','eos_id':eos,'unk_id':unk,
            'unknown_tokens':unknown_tokens,'training_seconds':token_training_seconds,'encoding_seconds':encoding_seconds,
            'library_version':tokenizers.__version__},
        'model':{'parameters':29676544,'layers':8,'width':512,'heads':8,'context':512,'batch':16,'precision':'fp32'},
        'training_projection':{'measured_training_tokens_per_second':rate,'planning_reserve_fraction':.2,
            'planning_training_tokens_per_second':planned,'training_tokens_one_full_pass':steps_per_epoch*8192,
            'optimizer_steps_per_epoch':steps_per_epoch,'minutes_per_epoch_with_reserve':steps_per_epoch*8192/planned/60,
            'five_hours':{'optimizer_steps':math.floor(planned*18000/8192),
                'training_tokens':math.floor(planned*18000/8192)*8192,
                'equivalent_epochs':math.floor(planned*18000/8192)/steps_per_epoch},
            'six_hours':{'optimizer_steps':math.floor(planned*21600/8192),
                'training_tokens':math.floor(planned*21600/8192)*8192,
                'equivalent_epochs':math.floor(planned*21600/8192)/steps_per_epoch}},
        'processing_seconds':time.perf_counter()-started,
    }
    (OUTPUT/'statistics.json').write_text(json.dumps(statistics,ensure_ascii=False,indent=2))
    with (OUTPUT/'manifest.jsonl').open('w') as file:
        for row in manifest:file.write(json.dumps(row,ensure_ascii=False)+'\n')
    (OUTPUT/'authors.json').write_text(json.dumps(sorted(author_stats.values(),key=lambda x:-x['tokens']),ensure_ascii=False,indent=2))
    (OUTPUT/'excluded-records.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
    event('complete',**{k:statistics[k] for k in ('documents','authors','text_utf8_bytes','tokens','splits','training_projection','processing_seconds')})

if __name__=='__main__':main()
