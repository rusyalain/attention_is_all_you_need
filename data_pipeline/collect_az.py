"""Inventory and retrieve original fiction by a fixed list of classic authors."""
import argparse
import bisect
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
from corpus_common import author_key,title_key,clean_text,norm

import os
BASE=Path(os.environ.get('RUSLIT_SOURCE_CACHE',Path(__file__).resolve().parent/'cache')).resolve()
AUTHORS=['Аксаков С.Т.','Андреев Л.Н.','Аверченко А.Т.','Белый А.','Брюсов В.Я.',
 'Бунин И.А.','Гаршин В.М.','Герцен А.И.','Гоголь Н.В.','Гончаров И.А.','Горький М.',
 'Григорович Д.В.','Достоевский Ф.М.','Карамзин Н.М.','Короленко В.Г.','Куприн А.И.',
 'Лермонтов М.Ю.','Лесков Н.С.','Мамин-Сибиряк Д.Н.','Мережковский Д.С.','Одоевский В.Ф.',
 'Писемский А.Ф.','Помяловский Н.Г.','Пушкин А.С.','Салтыков-Щедрин М.Е.','Сологуб Ф.',
 'Тэффи','Толстой А.К.','Толстой Л.Н.','Тургенев И.С.','Чехов А.П.','Шмелев И.С.',
 'Зайцев Б.К.','Арцыбашев М.П.','Ремизов А.М.','Вересаев В.В.','Загоскин М.Н.',
 'Данилевский Г.П.','Боборыкин П.Д.','Эртель А.И.','Успенский Г.И.','Булгаков М.А.']
NEG_SECTION=re.compile(r'стих|поэзи|публицист|критик|переписк|письма|эпистол|мемуар|дневник|перевод|об авторе|о творчестве|материал|библиограф|литература об|воспоминан|драматург|пьес|лекци|религи|философ',re.I)
POS_GENRE=re.compile(r'\b(?:Проза|Сказки|Юмор и сатира|Приключения|Фантастика)\b',re.I)
NEG_GENRE=re.compile(r'\b(?:Поэзия|Переводы|Мемуары|Эпистолярий|Публицистика|Критика|Справочная|Драматургия|История|Философия|Религия|Этнография)\b',re.I)

def fetch(url,path):
    if path.exists():return path.read_text()
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'LiteratureResearch/1.0','Accept-Encoding':'gzip'})
            with urllib.request.urlopen(req,timeout=35) as response:
                raw=response.read()
                if response.headers.get('Content-Encoding')=='gzip':raw=gzip.decompress(raw)
                charset=response.headers.get_content_charset() or 'cp1251'
            text=raw.decode(charset,errors='replace')
            path.write_text(text)
            time.sleep(0.1)
            return text
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)

def inventory_author(item):
    label,url=item;key=author_key(label)
    try:
        source=fetch(url,BASE/'az-indexes'/(key+'.html'))
        headings=[(m.start(),BeautifulSoup(m.group(1),'html.parser').get_text(' ',strip=True))
                  for m in re.finditer(r'<a name=gr\d+>(.*?)</b>',source,re.I|re.S)]
        positions=[x[0] for x in headings]
        works=[]
        for match in re.finditer(r'<DL><DT><li>(.*?)</DL>',source,re.I|re.S):
            fragment=BeautifulSoup(match.group(1),'html.parser');a=fragment.find('a',href=True)
            if a is None:continue
            target=urllib.parse.urljoin(url,a['href'])
            if not target.startswith(url) or not target.endswith(('.shtml','.html','.txt')):continue
            title=a.get_text(' ',strip=True)
            index=bisect.bisect_right(positions,match.start())-1
            section=headings[index][1] if index>=0 else ''
            smalls=fragment.find_all('small')
            genre=' '.join(s.get_text(' ',strip=True) for s in smalls[:2])
            before=match.group(1).split(str(a),1)[0] if str(a) in match.group(1) else match.group(1).split('<A ',1)[0]
            byline=BeautifulSoup(before,'html.parser').get_text(' ',strip=True)
            reason=None
            if NEG_SECTION.search(section):reason='nonfiction_poetry_or_secondary_section'
            elif NEG_GENRE.search(genre):reason='nonfiction_poetry_or_translation_genre'
            elif not POS_GENRE.search(genre):reason='not_explicitly_fiction'
            elif byline and norm(label.split()[0]) not in norm(byline):reason='other_author_byline'
            elif re.match(r'^\s*(?:[А-Я]\.[ ]*){1,3}[А-Я][а-я]',title):reason='other_author_in_title'
            elif re.match(r'^\s*[А-Я][а-я-]+\s+[А-Я]\.',title):reason='other_author_in_title'
            elif title_key(title) in {'историяпугачева','стихотворениявпрозе'}:reason='outside_requested_fiction_scope'
            size=re.search(r'<b>(\d+)k</b>',match.group(1),re.I)
            works.append({'author':label,'author_key':key,'title':title,'title_key':title_key(title),
                          'url':target,'section':section,'genre':genre,'index_kb':int(size[1]) if size else None,
                          'selected':reason is None,'excluded_reason':reason})
        print(json.dumps({'author':label,'candidates':sum(w['selected'] for w in works),'listed':len(works)},ensure_ascii=False),flush=True)
        return {'author':label,'author_key':key,'url':url,'status':'complete','works':works}
    except Exception as exc:
        print(json.dumps({'author':label,'error':repr(exc)},ensure_ascii=False),flush=True)
        return {'author':label,'author_key':key,'url':url,'status':'error','error':repr(exc),'works':[]}

def inventory():
    root=BeautifulSoup((BASE/'source_metadata'/'az-root.txt').read_text(),'html.parser')
    links=[(a.get_text(' ',strip=True),urllib.parse.urljoin('http://az.lib.ru/',a['href']))
           for a in root.find_all('a',href=True) if re.fullmatch(r'/[a-z]/[a-z0-9_]+/',a['href'])]
    mapping={norm(name):url for name,url in links}
    items=[];missing=[]
    for name in AUTHORS:
        url=mapping.get(norm(name))
        if url is None:
            options=[(label,u) for label,u in links if norm(label).startswith(norm(name))]
            if len(options)==1:url=options[0][1]
        if url:items.append((name,url))
        else:missing.append(name)
    print(json.dumps({'found_authors':len(items),'missing_authors':missing},ensure_ascii=False),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:authors=list(executor.map(inventory_author,items))
    result={'selected_authors':AUTHORS,'missing_authors':missing,'authors':authors}
    (BASE/'az-inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))

def extract(source):
    match=re.search(r'<!--\s*Section Begins\s*-->(.*?)<!--\s*Section Ends\s*-->',source,re.I|re.S)
    if match:body=match.group(1)
    else:
        match=re.search(r'<!--[^>]*Собственно произведение[^>]*-->(.*?)</noindex>',source,re.I|re.S)
        if not match:raise ValueError('Missing main-work HTML boundaries')
        body=match.group(1)
    body=re.sub(r'<(?:br|dd|p|/p|/h[1-6]|/div)\b[^>]*>','\n',body,flags=re.I)
    soup=BeautifulSoup(body,'html.parser')
    # Some legacy pages put the whole story inside a layout table.
    for tag in soup.find_all(['script','style','sup']):tag.decompose()
    return clean_text(soup.get_text(' ',strip=False))

def download():
    data=json.loads((BASE/'az-inventory.json').read_text())
    existing=set(tuple(x) for x in json.loads((BASE/'existing-work-keys.json').read_text()))
    works={w['url']:w for author in data['authors'] for w in author['works'] if w['selected']}
    selected=[];already=[]
    for work in works.values():
        if (work['author_key'],work['title_key']) in existing:already.append(work)
        else:selected.append(work)
    print(json.dumps({'download_candidates':len(selected),'already_in_archives':len(already)},ensure_ascii=False),flush=True)
    status=[];lock=threading.Lock()
    destination=BASE/'az-texts';destination.mkdir(exist_ok=True)
    def process(work):
        identifier=hashlib.sha256(work['url'].encode()).hexdigest()[:24]
        result=destination/(identifier+'.json')
        if result.exists():return json.loads(result.read_text())
        try:
            source=fetch(work['url'],BASE/'az-html'/(identifier+'.html'))
            text=extract(source)
            if len(text)<100:raise ValueError('Empty or very short main text')
            record={**work,'text':text,'source':'az.lib.ru','source_record':work['url'],'status':'complete'}
            result.write_text(json.dumps(record,ensure_ascii=False))
        except Exception as exc:record={**work,'status':'error','error':repr(exc)}
        with lock:
            status.append({'url':work['url'],'status':record['status'],'error':record.get('error')})
            if len(status)%25==0:print(json.dumps({'processed':len(status),'planned':len(selected),'errors':sum(x['status']=='error' for x in status)},ensure_ascii=False),flush=True)
        return record
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:results=list(executor.map(process,selected))
    manifest={'already_in_archives':already,'requested':len(selected),'complete':sum(x['status']=='complete' for x in results),
              'errors':[{k:v for k,v in x.items() if k!='text'} for x in results if x['status']!='complete']}
    (BASE/'az-download-summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'status':'complete','downloaded':manifest['complete'],'errors':len(manifest['errors'])},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['inventory','download']);args=parser.parse_args()
    inventory() if args.phase=='inventory' else download()
