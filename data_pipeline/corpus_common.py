import re
import unicodedata

def norm(value):
    return re.sub(r'[^a-zа-я0-9]+','',value.casefold().replace('ё','е'))

def author_key(name):
    clean=re.sub(r'\([^)]*\)','',name).strip().replace('ё','е')
    lower=clean.casefold()
    if 'чехонте' in lower:return 'чехов_ап'
    if 'мамин' in lower:return 'маминсибиряк_дн'
    if 'погорельск' in lower:return 'погорельский_а'
    if 'микулич' in lower or 'веселитск' in lower:return 'веселитская_ли'
    if 'салиас' in lower or 'евгения тур' in lower:return 'салиасдетурнемир_ев'
    if 'горький' in lower:return 'горький_м'
    if 'белый' in lower:return 'белый_а'
    if ',' in clean:
        surname,rest=clean.split(',',1)
        initials=''.join(x[0] for x in rest.split() if x and x[0].isalpha())
    else:
        initials=''.join(re.findall(r'\b([А-ЯA-Z])[.]',clean))
        rest=re.sub(r'\b[А-ЯA-Z][.]',' ',clean).strip().split()
        surname=rest[-1] if rest else clean
        if not initials and len(rest)>1:
            surname=rest[-1];initials=''.join(w[0] for w in rest[:-1])
    surname=norm(surname)
    if surname=='бестужев' and norm(initials)=='аа':surname='бестужевмарлинский'
    return surname+'_'+norm(initials)

def title_key(title):
    title=title.casefold().replace('ё','е')
    title=re.sub(r'\([^)]*(?:редакц|вариант|издан)[^)]*\)','',title)
    title=re.sub(r'\b(?:том|часть|книга)\s*[ivx0-9]+\b.*$','',title)
    return norm(title)

def edition_title_key(title):
    title=title.casefold().replace('ё','е')
    numbers={'первый':'1','второй':'2','третий':'3','четвертый':'4','пятый':'5',
             'шестой':'6','седьмой':'7','восьмой':'8','девятый':'9','десятый':'10',
             'i':'1','ii':'2','iii':'3','iv':'4','v':'5','vi':'6','vii':'7','viii':'8','ix':'9','x':'10'}
    def replace(match):return match[1]+numbers.get(match[2],match[2])
    title=re.sub(r'\b(том|часть|книга)\s+(\d+|первый|второй|третий|четвертый|пятый|шестой|седьмой|восьмой|девятый|десятый|[ivx]+)\b',replace,title)
    return norm(title)

def clean_text(text):
    text=unicodedata.normalize('NFC',text.lstrip('\ufeff')).replace('\r\n','\n').replace('\r','\n')
    text=text.replace('\xa0',' ').replace('\u00ad','').replace('\u200b','')
    # Corpus editions use braces for page references and editorial footnotes.
    for _ in range(3):
        updated=re.sub(r'\{[^{}]{0,20000}\}','',text)
        if updated==text:break
        text=updated
    text=re.sub(r'\[(?:\d{1,3})\]','',text)
    text=re.sub(r'</?p\s*>','\n\n',text,flags=re.I)
    lines=[re.sub(r'[ \t]+',' ',line).strip() for line in text.splitlines()]
    text='\n'.join(lines)
    text=re.sub(r'\n{3,}','\n\n',text).strip()
    # Cut a separately headed editorial appendix, retaining the work itself.
    for match in re.finditer(r'(?m)^(?:ПРИМЕЧАНИЯ|КОММЕНТАРИИ|Примечания|Комментарии)\s*$',text):
        if match.start()>len(text)*0.65:
            text=text[:match.start()].rstrip();break
    for match in re.finditer(r'(?mi)^(?:Подготовка текста|Электронная версия|Публикация текста)\b',text):
        if match.start()>len(text)*.65:
            text=text[:match.start()].rstrip();break
    return text
