"""Re-extract cached HTML after format QA and account for every requested work."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import threading
import collect_az

import os
BASE=Path(os.environ.get('RUSLIT_SOURCE_CACHE',Path(__file__).resolve().parent/'cache')).resolve()
inventory=json.loads((BASE/'az-inventory.json').read_text())
existing=set(tuple(x) for x in json.loads((BASE/'existing-work-keys.json').read_text()))
works={w['url']:w for a in inventory['authors'] for w in a['works'] if w['selected'] and (w['author_key'],w['title_key']) not in existing}
done=0;lock=threading.Lock()
def process(work):
    global done
    identifier=hashlib.sha256(work['url'].encode()).hexdigest()[:24]
    html_path=BASE/'az-html'/(identifier+'.html')
    try:
        html=collect_az.fetch(work['url'],html_path)
        text=collect_az.extract(html)
        if len(text)<100:raise ValueError('No usable main text')
        value={**work,'text':text,'source':'az.lib.ru','source_record':work['url'],'status':'complete'}
        (BASE/'az-texts'/(identifier+'.json')).write_text(json.dumps(value,ensure_ascii=False))
        error=None
    except Exception as exc:error={**work,'status':'error','error':repr(exc)}
    with lock:
        done+=1
        if done%200==0:print(json.dumps({'reextracted':done,'total':len(works)}),flush=True)
    return error
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    errors=[x for x in executor.map(process,works.values()) if x]
summary=json.loads((BASE/'az-download-summary.json').read_text())
summary['original_errors']=summary['errors']
summary['errors']=errors
summary['complete']=len(works)-len(errors)
summary['html_reextracted_preserving_layout_tables']=True
(BASE/'az-download-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({'status':'complete','reextracted':len(works),'errors':errors},ensure_ascii=False),flush=True)
if errors:raise SystemExit(1)
