"""A local, offline interface for the trained prose model. No external API calls."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import torch
from generate import load_model

PROJECT = Path(__file__).resolve().parent
APP_ID = 'ruslit-lm-demo'
PAGE = '''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Продолжение русской прозы</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#171717;font:19px/1.6 system-ui,sans-serif}
main{max-width:960px;padding:58px 28px 70px;margin:auto}h1{font-size:42px;line-height:1.15;letter-spacing:-.025em;margin:0 0 18px}
p{margin:0 0 28px}.intro{color:#555;font-size:17px;max-width:720px}label{display:block;font-weight:600;margin:24px 0 9px}
textarea,input,button{font:inherit}textarea{width:100%;min-height:126px;resize:vertical;padding:14px;border:1px solid #777;border-radius:3px}
.controls{display:flex;gap:24px;align-items:end;flex-wrap:wrap}.controls label{font-size:16px;margin-top:18px}.controls input{width:100px;padding:7px 9px;border:1px solid #888;border-radius:3px}
button{padding:10px 22px;border:1px solid #111;background:#111;color:#fff;border-radius:3px;cursor:pointer;margin-top:24px}
button:disabled{opacity:.5;cursor:wait}.status{font-size:15px;color:#666;min-height:26px;margin-top:18px}
#result{white-space:pre-wrap;overflow-wrap:anywhere;font-family:Georgia,serif;font-size:23px;line-height:1.65;margin:20px 0 42px}
details{border-top:1px solid #ddd;padding-top:18px;font-size:16px}summary{cursor:pointer}article{margin:24px 0}article p{white-space:pre-wrap;margin:10px 0}
.note{font-size:15px;color:#666;margin-top:36px}a{color:inherit}
@media(max-width:600px){main{padding-top:28px}h1{font-size:34px}#result{font-size:21px}}
</style>
<main>
<h1>Продолжение русской прозы</h1>
<p class="intro">Экспериментальная модель из 29,7 млн параметров, обученная с нуля на художественной прозе. Вычисление выполняется на этом компьютере.</p>
<form id="form">
<label for="prompt">Начало текста</label>
<textarea id="prompt" maxlength="4000" required>Она остановилась у окна. За стеклом медленно</textarea>
<div class="controls">
<div><label for="temperature">Температура</label><input id="temperature" type="number" min="0" max="1.5" step="0.1" value="0.8" required></div>
<div><label for="tokens">Новых токенов</label><input id="tokens" type="number" min="1" max="320" value="160" required></div>
<div><label for="seed">Seed</label><input id="seed" type="number" min="0" max="2147483647" value="42" required></div>
</div>
<button id="submit">Продолжить</button>
</form>
<div class="status" id="status" role="status" aria-live="polite"></div>
<div id="result" aria-live="polite"></div>
<p class="note">Низкая температура усиливает вероятные варианты. Seed задаёт вариант случайного выбора. Контекст ограничен последними 512 токенами. Модель может повторяться, ошибаться в согласовании и терять сюжет.</p>
</main>
<script>
const form=document.getElementById('form'), button=document.getElementById('submit'), statusText=document.getElementById('status');
form.addEventListener('submit',async event=>{
 event.preventDefault();button.disabled=true;statusText.textContent='Модель продолжает текст…';document.getElementById('result').textContent='';
 try{
  const response=await fetch('/api/complete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:document.getElementById('prompt').value,temperature:Number(document.getElementById('temperature').value),tokens:Number(document.getElementById('tokens').value),seed:Number(document.getElementById('seed').value)})});
  const result=await response.json();if(!response.ok)throw Error(result.error||'Ошибка генерации');
  document.getElementById('result').textContent=result.text;
  statusText.textContent=result.new_tokens+' новых токенов за '+result.seconds.toFixed(1)+' с. Текст без правки.';
 }catch(error){statusText.textContent=error instanceof TypeError?'Нет связи с демо. Запустите python demo.py в терминале и обновите страницу.':error.message;}finally{button.disabled=false;}
});
</script></html>'''


def existing_demo(port):
    """Recognise our running server without mistaking another app for it."""
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health',timeout=1) as response:
            health=json.loads(response.read(4096))
        return isinstance(health,dict) and health.get('application')==APP_ID and health.get('status')=='ready'
    except (OSError,ValueError,urllib.error.URLError):
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--device',choices=['mps','cpu'])
    parser.add_argument('--open-browser',action='store_true')
    args=parser.parse_args()
    if not 1<=args.port<=65535:parser.error('Номер порта должен быть от 1 до 65535')
    if args.open_browser and existing_demo(args.port):
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{args.port}/')
        print('Демо уже запущено; страница открыта в браузере.',flush=True)
        return
    torch.set_num_threads(4)
    model,tokenizer=load_model(device=args.device)
    device=next(model.parameters()).device
    generation_lock=threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        # Browsers may pre-open sockets without sending a request. Each socket
        # has its own worker and a timeout, so it cannot block the whole demo.
        timeout=15

        def send(self,body,status=200,content_type='application/json; charset=utf-8'):
            raw=body.encode('utf-8') if isinstance(body,str) else json.dumps(body,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
            self.end_headers()
            if self.command!='HEAD':
                try:self.wfile.write(raw)
                except (BrokenPipeError,ConnectionResetError):pass

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if self.path=='/':return self.send(PAGE,content_type='text/html; charset=utf-8')
            if self.path=='/api/health':
                return self.send({'application':APP_ID,'status':'ready','device':str(device)})
            self.send({'error':'Страница не найдена'},404)

        def do_POST(self):
            if self.path!='/api/complete':return self.send({'error':'Страница не найдена'},404)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=32000:raise ValueError('Недопустимый размер запроса')
                data=json.loads(self.rfile.read(length))
                prompt=data['prompt'];temperature=float(data.get('temperature',.8))
                tokens=int(data.get('tokens',160));seed=int(data.get('seed',42))
                if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>4000:
                    raise ValueError('Введите начало текста длиной до 4000 символов')
                if not 0<=temperature<=1.5 or not 1<=tokens<=320 or not 0<=seed<=2**31-1:
                    raise ValueError('Проверьте температуру, число токенов и seed')
                started=time.perf_counter()
                # Keep model inference serial while static pages and health
                # checks remain responsive in their own request workers.
                with generation_lock:
                    ids=tokenizer.encode(prompt).ids
                    inputs=torch.tensor([ids],device=device,dtype=torch.long)
                    output=model.generate(inputs,max_new_tokens=tokens,temperature=temperature,seed=seed)
                    text=tokenizer.decode(output[0].tolist())
                    new_tokens=output.shape[1]-len(ids)
                self.send({'text':text,'new_tokens':new_tokens,'seconds':time.perf_counter()-started})
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                self.send({'error':str(exc)},400)
            except Exception as exc:
                print(f'Generation failed: {exc!r}',flush=True)
                self.send({'error':'Не удалось выполнить генерацию. Подробности в терминале.'},500)

    try:server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    except OSError as exc:
        if args.open_browser and existing_demo(args.port):
            import webbrowser
            webbrowser.open(f'http://127.0.0.1:{args.port}/')
            return
        parser.exit(1,f'Не удалось запустить демо на порту {args.port}: {exc}\n')
    print(f'Local demo: http://127.0.0.1:{args.port}/ ; device={device}; Ctrl+C stops the server',flush=True)
    if args.open_browser:
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{args.port}/')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
