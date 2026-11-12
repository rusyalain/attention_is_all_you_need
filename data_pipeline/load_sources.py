import csv
import io
import json
from pathlib import Path
import zipfile
from corpus_common import author_key,title_key,clean_text

import os
BASE=Path(os.environ.get('RUSLIT_SOURCE_CACHE',Path(__file__).resolve().parent/'cache')).resolve()
RUSLIT_NAMES={'Blok':'Блок А.А.','Bryusov':'Брюсов В.Я.','Chekhov':'Чехов А.П.',
 'Dostoevsky':'Достоевский Ф.М.','Gogol':'Гоголь Н.В.','Gorky':'Горький М.',
 'Herzen':'Герцен А.И.','Lermontov':'Лермонтов М.Ю.','Pushkin':'Пушкин А.С.',
 'Tolstoy':'Толстой Л.Н.','Turgenev':'Тургенев И.С.'}

def decode(raw):
    try:return raw.decode('utf-8-sig')
    except UnicodeDecodeError:return raw.decode('cp1251')

def record(author,title,source,source_record,url,text=None,**extra):
    value={'author':author,'author_key':author_key(author),'title':title,'title_key':title_key(title),
           'source':source,'source_record':source_record,'source_url':url,**extra}
    if text is not None:value['text']=clean_text(text)
    return value

def ruslit(text=True):
    result=[]
    with zipfile.ZipFile(BASE/'raw'/'RusLit.zip') as archive:
        for path in archive.namelist():
            parts=Path(path).parts
            if len(parts)!=4 or parts[1]!='prose' or not path.endswith('.txt'):continue
            author=RUSLIT_NAMES[parts[2]];title=Path(path).stem
            if title_key(title) in {'историяпугачева','ревизор','вишневыйсад','трисестры'}:continue
            result.append(record(author,title,'RusLit',path,
                'https://github.com/d0rj/RusLit/blob/4509680428729da225a73a22008133660f92bb7e/'+ '/'.join(parts[1:]),
                decode(archive.read(path)) if text else None,
                source_version='4509680428729da225a73a22008133660f92bb7e',source_priority=2))
    return result

def rsd(text=True):
    result=[]
    with zipfile.ZipFile(BASE/'raw'/'RSD.zip') as archive:
        members=set(archive.namelist())
        for path in archive.namelist():
            if '/author/fiction/' not in path or not path.endswith('/corpus.tsv'):continue
            rows=csv.DictReader(io.StringIO(decode(archive.read(path))),delimiter='\t')
            for row in rows:
                member=str(Path(path).parent/'corpus'/row['filename'])
                if member not in members:raise ValueError(f'RSD text missing: {member}')
                if title_key(row['title'])=='стихотворениявпрозе':continue
                result.append(record(row['author'],row['title'],'RSD',member,
                    'https://github.com/nevmenandr/RSD/blob/476876e8a5c430fb0f6e147bf1519c9b43699261/'+ '/'.join(Path(member).parts[1:]),
                    decode(archive.read(member)) if text else None,genre=row.get('genre'),year=row.get('year'),
                    source_version='476876e8a5c430fb0f6e147bf1519c9b43699261',source_priority=1))
    return result

def narrative(text=True):
    rows=list(csv.DictReader((BASE/'source_metadata'/'narrative-metadata.tsv').open(),delimiter='\t'))
    result=[]
    archive=zipfile.ZipFile(BASE/'raw'/'narrative-texts.zip') if text else None
    try:
        by_name={Path(path).name:path for path in archive.namelist() if path.endswith('.txt')} if archive else {}
        for row in rows:
            path=by_name.get(row['filename'])
            if text and not path:raise ValueError(f'Narrative text missing: {row["filename"]}')
            result.append(record(row['author'],row['title'],'PushkinHouseNarrative',row['filename'],
                'https://doi.org/10.31860/openlit-2020.10-C004',
                decode(archive.read(path)) if text else None,year=row.get('year'),edition=row.get('colophon'),
                source_version='4.0; file 5915; md5 8dcc41eaa715dc0bece1db6fcf25242a',source_priority=0))
    finally:
        if archive:archive.close()
    return result

def az():
    result=[]
    for path in (BASE/'az-texts').glob('*.json'):
        row=json.loads(path.read_text())
        if row.get('status')!='complete':continue
        result.append(record(row['author'],row['title'],'az.lib.ru',row['url'],row['url'],row['text'],
                             genre=row.get('genre'),section=row.get('section'),source_priority=3))
    return result

if __name__=='__main__':
    rows=ruslit(text=False)+rsd(text=False)+narrative(text=False)
    keys=sorted(set((row['author_key'],row['title_key']) for row in rows))
    (BASE/'existing-work-keys.json').write_text(json.dumps(keys,ensure_ascii=False))
    print(json.dumps({'ruslit':len(ruslit(False)),'rsd':len(rsd(False)),'narrative':len(narrative(False)),
                      'distinct_author_title_keys':len(keys),'authors':len(set(row['author_key'] for row in rows))},ensure_ascii=False))
