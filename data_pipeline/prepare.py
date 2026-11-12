"""Download the pinned source snapshot and rebuild the prepared literary corpus."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import urllib.request

HERE=Path(__file__).resolve().parent


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as file:
        while block:=file.read(1024*1024):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,default=HERE/'cache')
    parser.add_argument('--output-dir',type=Path,default=HERE.parent/'data')
    args=parser.parse_args()
    cache=args.cache_dir.resolve();output=args.output_dir.resolve()
    if (output/'statistics.json').exists():
        parser.error('Output already contains a prepared corpus. Choose a new --output-dir; existing data will not be overwritten.')
    os.environ['RUSLIT_SOURCE_CACHE']=str(cache)
    os.environ['RUSLIT_CORPUS_OUTPUT']=str(output)
    (cache/'raw').mkdir(parents=True,exist_ok=True)
    shutil.copytree(HERE/'snapshot'/'source_metadata',cache/'source_metadata',dirs_exist_ok=True)
    shutil.copy2(HERE/'snapshot'/'az-inventory.json',cache/'az-inventory.json')
    sources=json.loads((HERE/'snapshot'/'source-archives.json').read_text())
    for name,source in sources.items():
        target=cache/'raw'/name
        if not target.exists():
            print(json.dumps({'event':'download','file':name,'url':source['url']}),flush=True)
            if name=='narrative-texts.zip':
                runpy.run_path(str(HERE/'download_narrative.py'),run_name='__main__')
            else:
                request=urllib.request.Request(source['url'],headers={'User-Agent':'LiteratureResearch/1.0'})
                temporary=target.with_suffix('.part')
                with urllib.request.urlopen(request,timeout=60) as response,temporary.open('wb') as file:
                    while block:=response.read(1024*1024):file.write(block)
                temporary.replace(target)
        if target.stat().st_size!=source['bytes'] or digest(target)!=source['sha256']:
            raise RuntimeError(f'Source archive checksum differs from the recorded snapshot: {target}')
    sys.path.insert(0,str(HERE))
    import load_sources
    import collect_az
    rows=load_sources.ruslit(False)+load_sources.rsd(False)+load_sources.narrative(False)
    keys=sorted({(r['author_key'],r['title_key']) for r in rows})
    (cache/'existing-work-keys.json').write_text(json.dumps(keys,ensure_ascii=False))
    collect_az.download()
    runpy.run_path(str(HERE/'reconcile_sources.py'),run_name='__main__')
    runpy.run_path(str(HERE/'build_corpus.py'),run_name='__main__')
    print(f'Prepared corpus: {output}',flush=True)


if __name__=='__main__':main()
