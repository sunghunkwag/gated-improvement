"""A stand-in OpenAI-compatible chat model for end-to-end tests of the env.

It recognises which world a prompt belongs to (it is told the salt, which a
real model never is) and answers with a scripted policy, so a full `vf-run`
rollout -- harness, model call, trace, scoring -- can be exercised without
an API key:

    world 0 -> the true motif library  (a real improvement)
    world 1 -> the baseline            (no-op)
    world 2 -> prose without JSON      (invalid)

Usage: python scripts/mock_model_server.py --salt e2e --port 18765
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gated_improvement.core import world as W  # noqa: E402
from gated_improvement.taskset import build_prompt  # noqa: E402


def make_policy(salt, n_worlds, n_shown):
    prompts = {build_prompt(W.world_for(salt, i), n_shown): i for i in range(n_worlds)}

    def answer(user_text):
        i = prompts.get(user_text.strip())
        if i is None:
            for p, j in prompts.items():
                if p in user_text:
                    i = j
                    break
        if i == 0:
            w = W.world_for(salt, 0)
            lib = [list(u) for u in w.atoms + w.compounds]
            return "These look like recurring blocks.\n```json\n" + json.dumps({"macros": lib}) + "\n```"
        if i == 1:
            return "I will keep the baseline.\n```json\n{}\n```"
        return "I am not sure what to change."

    return answer


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salt", required=True)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--n-shown", type=int, default=40)
    a = ap.parse_args()
    answer = make_policy(a.salt, a.worlds, a.n_shown)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("mock: " + (fmt % args) + "\n")

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self._json(200, {"object": "list", "data": [{"id": "mock", "object": "model"}]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self._json(404, {"error": "unsupported path %s" % self.path})
            users = [text_of(m.get("content")) for m in req.get("messages", []) if m.get("role") == "user"]
            reply = answer(users[-1] if users else "")
            created = int(time.time())
            usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
                ]
                for c in chunks:
                    c.update({"id": "mock-1", "object": "chat.completion.chunk", "created": created, "model": req.get("model", "mock")})
                    self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                return
            self._json(
                200,
                {
                    "id": "mock-1",
                    "object": "chat.completion",
                    "created": created,
                    "model": req.get("model", "mock"),
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}
                    ],
                    "usage": usage,
                },
            )

    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
