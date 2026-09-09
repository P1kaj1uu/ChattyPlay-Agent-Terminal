from __future__ import annotations

import json
import os
import re
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import ConfigStore
from .ollama import CHAT_MODEL, provider as ollama_provider, status as ollama_status
from .rag import RAGIndex
from .skills import discover


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>配置中心</title><style>
:root{color-scheme:dark;--bg:#0a0d10;--panel:#12171c;--line:#273039;--text:#eef3f6;--muted:#8c9ba7;--mint:#62e6b2;--red:#ff7777}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#15312c 0,transparent 34%),var(--bg);color:var(--text);font:15px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}main{max-width:1100px;margin:auto;padding:44px 22px 80px}header{display:flex;align-items:end;justify-content:space-between;margin-bottom:28px}h1{font:700 clamp(28px,5vw,52px)/1 system-ui;margin:0;letter-spacing:-.04em}header p{color:var(--muted);margin:9px 0 0}.badge{border:1px solid #357661;color:var(--mint);padding:6px 10px;border-radius:999px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.card{background:color-mix(in srgb,var(--panel) 92%,transparent);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 14px 40px #0005}.wide{grid-column:1/-1}h2{font:650 17px system-ui;margin:0 0 16px}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}label{display:block;color:var(--muted);font-size:12px;margin:0 0 12px}input,select,textarea{display:block;width:100%;margin-top:6px;border:1px solid var(--line);border-radius:9px;background:#090c0f;color:var(--text);padding:10px 11px;font:inherit;outline:none}input:focus,select:focus,textarea:focus{border-color:var(--mint)}textarea{min-height:190px;resize:vertical}.checks{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;max-height:270px;overflow:auto}.check{display:flex;gap:9px;align-items:center;background:#0b0f12;border-radius:8px;padding:9px;color:var(--text);margin:0}.check input{width:auto;margin:0}.actions{position:sticky;bottom:16px;display:flex;justify-content:flex-end;align-items:center;gap:12px;margin-top:20px}.status{color:var(--muted)}button{border:0;border-radius:10px;background:var(--mint);color:#06251b;padding:11px 18px;font:700 14px system-ui;cursor:pointer}button.secondary{background:#202830;color:var(--text)}code{color:var(--mint)}@media(max-width:720px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.row{grid-template-columns:1fr}header{align-items:start;flex-direction:column;gap:16px}}
.actions{justify-content:initial;padding:10px;background:#12171cf5;border:1px solid var(--line);border-radius:14px;box-shadow:0 10px 30px #0008}.actions .status{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-right:auto}
</style></head><body><main><header><div><h1>Control Room</h1><p>本地模型、权限、Skills 与 MCP 配置</p></div><span class="badge">LOCAL · 127.0.0.1</span></header>
<div class="grid"><section class="card wide"><h2>模型</h2><div class="row"><label>API 协议<select id="apistyle"><option value="openai">OpenAI compatible</option><option value="anthropic">Anthropic Messages</option></select></label><label>模型名称<input id="model"></label><label>Base URL<input id="base"></label><label>API Key 环境变量<input id="keyenv"></label><label>最大输出 Token<input id="tokens" type="number" min="1"></label><label>思考模式<select id="thinking"><option value="false">关闭</option><option value="true">开启</option></select></label><label>推理强度<select id="effort"><option>low</option><option>medium</option><option>high</option><option>max</option></select></label><label>本地 Ollama 模型<select id="ollamamodel"></select></label></div><p id="ollama" class="status"></p><button type="button" onclick="useOllama()">切换所选本地模型（免 Key）</button></section>
<section class="card"><h2>权限策略</h2><div id="permissions"></div></section>
<section class="card"><h2>Skills</h2><label>筛选<input id="skillfilter" placeholder="输入名称…" oninput="filterSkills()"></label><div id="skills" class="checks"></div><p style="color:var(--muted)">在 <code>.chattyplay/skills/&lt;name&gt;/SKILL.md</code> 添加自定义技能。</p></section>
<section class="card wide"><h2>Skill 编辑器</h2><div class="row"><label>Skill 名称<input id="skillname" list="skillnames" placeholder="例如 reviewer"><datalist id="skillnames"></datalist></label><div><button class="secondary" style="margin-top:23px" onclick="loadSkill()">载入</button></div></div><label>SKILL.md<textarea id="skillbody" spellcheck="false" placeholder="---\nname: reviewer\ndescription: ...\n---\n\nInstructions..."></textarea></label><button onclick="saveSkill()">保存项目 Skill</button></section>
<section class="card wide"><h2>MCP Servers</h2><label>JSON 配置<textarea id="mcp" spellcheck="false"></textarea></label></section>
<section class="card wide"><h2>Provider Profiles</h2><label>用于终端 <code>/provider name</code> 热切换<textarea id="profiles" spellcheck="false"></textarea></label></section>
<section class="card wide"><h2>RAG 语义检索</h2><label>RAG JSON<textarea id="rag" spellcheck="false"></textarea></label><p id="ragstatus" class="status"></p><button type="button" onclick="indexRag()">重建项目索引</button></section>
<section class="card wide"><h2>高级配置</h2><label>Agent JSON<textarea id="agent" spellcheck="false"></textarea></label></section></div>
<div class="actions"><span id="status" class="status"></span><button class="secondary" onclick="load()">撤销修改</button><button onclick="save()">保存配置</button></div></main>
<script>
const token='__TOKEN__';let state;
const esc=s=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function load(){const r=await fetch('/api/state');state=await r.json();const c=state.config;by('apistyle').value=c.provider.api_style||'openai';by('base').value=c.provider.base_url;by('model').value=c.provider.model;by('keyenv').value=c.provider.api_key_env;by('tokens').value=c.provider.max_tokens;by('thinking').value=String(Boolean(c.provider.thinking_enabled));by('effort').value=c.provider.reasoning_effort||'medium';by('mcp').value=JSON.stringify(c.mcpServers,null,2);by('profiles').value=JSON.stringify(c.providerProfiles||{},null,2);by('rag').value=JSON.stringify(c.rag||{},null,2);by('agent').value=JSON.stringify(c.agent,null,2);const models=state.ollama.models.length?state.ollama.models:[state.recommended_model],selected=models.includes(c.provider.model)?c.provider.model:models.includes(state.recommended_model)?state.recommended_model:models[0];by('ollamamodel').innerHTML=models.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');by('ollamamodel').value=selected;by('ollama').textContent=`Ollama: ${state.ollama.running?'运行中':state.ollama.installed?'已安装但未运行':'未安装'} · ${state.ollama.models.join(', ')||'无模型'}`;by('ragstatus').textContent=`索引: ${state.rag.indexed?'已建立':'未建立'} · ${state.rag.chunks} chunks`;by('permissions').innerHTML=Object.entries(c.permissions).map(([k,v])=>`<label>${esc(k)}<select data-perm="${esc(k)}">${['allow','ask','deny'].map(x=>`<option ${x===v?'selected':''}>${x}</option>`).join('')}</select></label>`).join('');const enabled=new Set(c.skills.enabled||[]);by('skills').innerHTML=state.skills.map(s=>`<label class="check" title="${esc(s.description||s.path)}"><input type="checkbox" data-skill="${esc(s.name)}" ${enabled.has(s.name)?'checked':''}>${esc(s.name)}</label>`).join('')||'<span class="status">尚未发现 Skill</span>';by('skillnames').innerHTML=state.skills.map(s=>`<option value="${esc(s.name)}">`).join('');msg('已载入 '+state.path)}
async function save(){try{const c=structuredClone(state.config);c.provider.api_style=by('apistyle').value;c.provider.base_url=by('base').value.trim();c.provider.model=by('model').value.trim();c.provider.api_key_env=by('keyenv').value.trim();c.provider.max_tokens=Number(by('tokens').value);c.provider.thinking_enabled=by('thinking').value==='true';c.provider.reasoning_effort=by('effort').value;c.permissions=Object.fromEntries([...document.querySelectorAll('[data-perm]')].map(x=>[x.dataset.perm,x.value]));c.skills.enabled=[...document.querySelectorAll('[data-skill]:checked')].map(x=>x.dataset.skill);c.mcpServers=JSON.parse(by('mcp').value);c.providerProfiles=JSON.parse(by('profiles').value);c.rag=JSON.parse(by('rag').value);c.agent=JSON.parse(by('agent').value);const r=await fetch('/api/config',{method:'PUT',headers:{'Content-Type':'application/json','X-ChattyPlay-Token':token},body:JSON.stringify(c)});const out=await r.json();if(!r.ok)throw Error(out.error);msg('保存成功，终端下次会话生效');await load()}catch(e){msg(e.message,true)}}
function by(id){return document.getElementById(id)}function msg(s,bad=false){by('status').textContent=s;by('status').style.color=bad?'var(--red)':'var(--muted)'}load().catch(e=>msg(e.message,true));
function filterSkills(){const q=by('skillfilter').value.toLowerCase();document.querySelectorAll('.check').forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(q))}
async function loadSkill(){try{const name=by('skillname').value.trim();const r=await fetch('/api/skill?name='+encodeURIComponent(name));const out=await r.json();if(!r.ok)throw Error(out.error);by('skillbody').value=out.content;msg('已载入 '+out.path+(out.editable?'':'（保存时创建项目覆盖）'))}catch(e){msg(e.message,true)}}
async function saveSkill(){try{const name=by('skillname').value.trim(),content=by('skillbody').value;const r=await fetch('/api/skill',{method:'PUT',headers:{'Content-Type':'application/json','X-ChattyPlay-Token':token},body:JSON.stringify({name,content})});const out=await r.json();if(!r.ok)throw Error(out.error);msg('已保存 '+out.path);await load()}catch(e){msg(e.message,true)}}
async function action(path,working,body){try{msg(working);const r=await fetch(path,{method:'PUT',headers:{'Content-Type':'application/json','X-ChattyPlay-Token':token},body:body?JSON.stringify(body):null}),out=await r.json();if(!r.ok)throw Error(out.error);msg(out.message||'完成');await load()}catch(e){msg(e.message,true)}}
function useOllama(){action('/api/ollama/use','正在切换…',{model:by('ollamamodel').value})}function indexRag(){action('/api/rag/index','正在生成 embeddings 与索引…')}
</script></body></html>'''


def make_handler(store: ConfigStore, token: str):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, data: object) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                body = PAGE.replace("__TOKEN__", token).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/state":
                config = store.load()
                skill_cfg = config.get("skills", {})
                skills = discover(store.workspace, skill_cfg.get("dirs", []))
                self._json(200, {"config": config, "path": str(store.project_path), "ollama": ollama_status(), "recommended_model": CHAT_MODEL, "rag": RAGIndex(store.workspace, config.get("rag", {})).status(), "skills": [
                    {"name": s.name, "description": s.description, "path": str(s.path)} for s in skills.values()
                ]})
                return
            if path == "/api/skill":
                name = parse_qs(parsed.query).get("name", [""])[0]
                if not re.fullmatch(r"[\w.-]+", name):
                    self._json(400, {"error": "invalid skill name"})
                    return
                config = store.load().get("skills", {})
                skill = discover(store.workspace, config.get("dirs", [])).get(name)
                if not skill:
                    self._json(404, {"error": "skill not found; enter content to create it"})
                    return
                project_root = (store.workspace / ".chattyplay" / "skills").resolve()
                self._json(200, {"name": name, "content": skill.path.read_text(encoding="utf-8", errors="replace"), "path": str(skill.path), "editable": skill.path.resolve().is_relative_to(project_root)})
                return
            self._json(404, {"error": "not found"})

        def do_PUT(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path not in {"/api/config", "/api/skill", "/api/ollama/use", "/api/rag/index"}:
                self._json(404, {"error": "not found"})
                return
            if self.headers.get("X-ChattyPlay-Token") != token:
                self._json(403, {"error": "invalid token"})
                return
            try:
                if path == "/api/ollama/use":
                    length = int(self.headers.get("Content-Length", "0"))
                    if length > 10_000:
                        raise ValueError("request is too large")
                    data = json.loads(self.rfile.read(length)) if length else {}
                    model = data.get("model", CHAT_MODEL) if isinstance(data, dict) else ""
                    if not isinstance(model, str) or not model.strip() or len(model) > 300:
                        raise ValueError("model must be a non-empty string")
                    project = store.project_config(); project["provider"] = ollama_provider(model.strip()); store.save_project(project)
                    self._json(200, {"ok": True, "message": f"已切换本地模型 {model.strip()}（免 API Key）"})
                    return
                if path == "/api/rag/index":
                    result = RAGIndex(store.workspace, store.load().get("rag", {})).index()
                    self._json(200, {"ok": True, "message": f"已索引 {result['files']} files / {result['chunks']} chunks"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1_000_000:
                    raise ValueError("config is too large")
                data = json.loads(self.rfile.read(length))
                if path == "/api/config":
                    store.save_project(data)
                    self._json(200, {"ok": True})
                else:
                    name = data.get("name", "") if isinstance(data, dict) else ""
                    content = data.get("content", "") if isinstance(data, dict) else ""
                    if not re.fullmatch(r"[\w.-]+", name) or not isinstance(content, str) or not content.strip():
                        raise ValueError("skill name and content are required")
                    if len(content) > 500_000:
                        raise ValueError("skill is too large")
                    target = store.workspace / ".chattyplay" / "skills" / name / "SKILL.md"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_suffix(".tmp")
                    temp.write_text(content, encoding="utf-8")
                    os.replace(temp, target)
                    self._json(200, {"ok": True, "path": str(target)})
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def create_server(store: ConfigStore, port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, token))
    return server, f"http://127.0.0.1:{server.server_port}/"


def run(store: ConfigStore, port: int = 0, open_page: bool = True) -> None:
    server, url = create_server(store, port)
    print(f"ChattyPlay config: {url}\nPress Ctrl+C to stop.")
    if open_page:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
