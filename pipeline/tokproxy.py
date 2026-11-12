"""Logging proxy between the agent (LiteLLM) and vLLM that records exact per-call token IDs for SFT.

usage: tokproxy.py LISTEN_PORT TAG_FILE OUT_JSONL [UPSTREAM]
- Adds "return_token_ids": true to every chat completion request.
- Appends {task, ts, prompt_token_ids, completion_token_ids, finish_reason, usage} to OUT_JSONL.
- TAG_FILE holds the current task id; the runner rewrites it before each task.
- Strips the token-id fields before returning the response, so the client sees a normal reply.
"""
import json
import sys
import time

from aiohttp import ClientSession, ClientTimeout, web

PORT = int(sys.argv[1])
TAG_FILE = sys.argv[2]
OUT = sys.argv[3]
UP = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:8000"


def current_task():
    try:
        return open(TAG_FILE).read().strip()
    except OSError:
        return ""


async def chat(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    task = current_task()
    body["return_token_ids"] = True
    stream = bool(body.get("stream"))
    async with ClientSession(timeout=ClientTimeout(total=3600)) as s:
        async with s.post(f"{UP}/v1/chat/completions", json=body, headers={"Authorization": request.headers.get("Authorization", "")}) as r:
            if not stream:
                data = await r.json()
                rec = {"task": task, "ts": time.time(), "status": r.status}
                if r.status == 200 and data.get("choices"):
                    ch = data["choices"][0]
                    rec.update(prompt_token_ids=data.pop("prompt_token_ids", None) or ch.pop("prompt_token_ids", None),
                               completion_token_ids=ch.pop("token_ids", None), finish_reason=ch.get("finish_reason"),
                               usage=data.get("usage"))
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                return web.json_response(data, status=r.status)
            resp = web.StreamResponse(status=r.status, headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            prompt_ids, comp_ids, finish = None, [], None
            buf = b""
            async for chunk in r.content.iter_any():
                buf += chunk
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    line = ev.decode(errors="replace")
                    if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                        try:
                            d = json.loads(line[6:])
                            if d.get("prompt_token_ids"):
                                prompt_ids = d.pop("prompt_token_ids")
                            for ch in d.get("choices", []):
                                comp_ids.extend(ch.pop("token_ids", None) or [])
                                finish = ch.get("finish_reason") or finish
                            line = "data: " + json.dumps(d)
                        except json.JSONDecodeError:
                            pass
                    await resp.write((line + "\n\n").encode())
            if buf:
                await resp.write(buf)
            with open(OUT, "a") as f:
                f.write(json.dumps({"task": task, "ts": time.time(), "status": r.status, "prompt_token_ids": prompt_ids,
                                    "completion_token_ids": comp_ids, "finish_reason": finish, "stream": True}) + "\n")
            await resp.write_eof()
            return resp


async def passthrough(request: web.Request) -> web.Response:
    async with ClientSession() as s:
        async with s.request(request.method, f"{UP}{request.path_qs}", data=await request.read()) as r:
            return web.Response(body=await r.read(), status=r.status, content_type=r.content_type)


app = web.Application(client_max_size=64 * 1024 * 1024)
app.router.add_post("/v1/chat/completions", chat)
app.router.add_route("*", "/{tail:.*}", passthrough)
web.run_app(app, host="127.0.0.1", port=PORT, print=None)
