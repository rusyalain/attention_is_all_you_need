"""Fetch a public Dataverse file in resumable ranges; verify publisher's MD5."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request
import zipfile

import os
BASE=Path(os.environ.get('RUSLIT_SOURCE_CACHE',Path(__file__).resolve().parent/'cache')).resolve()
URL='https://dataverse.pushdom.ru/api/access/datafile/5915'
SIZE=97829899
EXPECTED='8dcc41eaa715dc0bece1db6fcf25242a'
PART=2*1024*1024
DIRECTORY=BASE/'raw'/'narrative-parts'
DIRECTORY.mkdir(parents=True,exist_ok=True)
lock=threading.Lock()
progress={}

def fetch(index):
    start=index*PART;end=min(SIZE,start+PART)-1
    path=DIRECTORY/f'{index:03d}.part'
    if path.exists() and path.stat().st_size==end-start+1:
        with lock:progress[index]=path.stat().st_size
        return path
    for attempt in range(3):
        try:
            request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}',
                'User-Agent':'LiteratureResearch/1.0'})
            with urllib.request.urlopen(request,timeout=45) as response:
                expected_length=int(response.headers.get('Content-Length','0'))
                if expected_length!=end-start+1:
                    raise RuntimeError(f'Range length mismatch: {expected_length}')
                with path.with_suffix('.tmp').open('wb') as file:
                    received=0
                    while chunk:=response.read(32768):
                        file.write(chunk);received+=len(chunk)
                        with lock:progress[index]=received
            if received!=end-start+1:raise RuntimeError('Incomplete range')
            path.with_suffix('.tmp').replace(path)
            return path
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures=[executor.submit(fetch,i) for i in range((SIZE+PART-1)//PART)]
    while not all(f.done() for f in futures):
        with lock:downloaded=sum(progress.values())
        print(json.dumps({'downloaded_bytes':downloaded,'total_bytes':SIZE,
                          'finished_parts':sum(f.done() for f in futures)},ensure_ascii=False),flush=True)
        concurrent.futures.wait(futures,timeout=10)
    paths=[f.result() for f in futures]
target=BASE/'raw'/'narrative-texts.zip'
temporary=target.with_suffix('.assembling')
digest=hashlib.md5()
with temporary.open('wb') as output:
    for path in paths:
        raw=path.read_bytes();digest.update(raw);output.write(raw)
assert temporary.stat().st_size==SIZE
assert digest.hexdigest()==EXPECTED,(digest.hexdigest(),EXPECTED)
with zipfile.ZipFile(temporary) as archive:assert archive.testzip() is None
temporary.replace(target)
print(json.dumps({'status':'complete','bytes':SIZE,'md5':digest.hexdigest()}),flush=True)
