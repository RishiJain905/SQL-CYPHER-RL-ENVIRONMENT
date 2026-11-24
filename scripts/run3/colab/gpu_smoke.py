"""The paid Colab smoke's GPU half (design.md 7.1: 0.5 h). Runs on a VM made by
`colab new -s run3 --gpu A100`; never trains.

The kernel keeps its state between `colab exec` calls, so this file is sent once (it defines the
steps and runs the first two) and each later step is one call:

    colab exec -s run3 -f scripts/run3/colab/gpu_smoke.py    # gpu(), heartbeat()
    echo "fetch_model()" | colab exec -s run3 --timeout 1800 # the released model to /content/model
    echo "hf_load()" | colab exec -s run3 --timeout 1800
    echo "llamacpp_build()" | colab exec -s run3 --timeout 3600
    echo "gguf_convert()" | colab exec -s run3 --timeout 1800
    echo "llamacpp_bench()" | colab exec -s run3 --timeout 3600
    echo "llamacpp_server()" | colab exec -s run3 --timeout 1800
    echo "wandb_offline()" | colab exec -s run3              # the results as an offline W&B run

Always pass --timeout: the CLI gives up on a cell after 30 s without output (smoke 1).
`vllm_probe()` is kept for the record: vLLM refuses Spark2_5ForCausalLM (smoke 1, 2026-11-23).

Every result is one JSON line, printed and appended to /content/smoke_results.jsonl. What it
answers:
1. which GPU the account got (the CLI has no 40 / 80 GB choice) and its memory;
2. whether a detached process outlives the cell that started it (a heartbeat file, read later
   with `colab download`): how a 5-hour run must be launched;
3. whether the released model loads on NVIDIA with its own code and step 40's adapter applied
   unmerged (PEFT), the adapter's dtype, and that the adapter changes the output (the logits
   with the adapter against without it);
4. whether vLLM loads the model with the adapter at run time, and how fast it samples.

W&B: offline on the VM (no key goes to Colab); the run folder comes back with `colab download`
and the PC syncs it (`guides/colab.md`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback

MODEL = "trjxter/Spark-X2.6-4B-SQL-Cypher-Agent"
MODEL_DIR = "/content/model"
TRANSFORMERS = "4.57.1"  # the version the model card pins (trust_remote_code)
ADAPTER = "/content/step40-adapter"  # uploaded: the PEFT folder (adapter_config.json + weights)
RESULTS = "/content/smoke_results.jsonl"


def say(step: str, **kw) -> None:
    line = json.dumps({"step": step, "t": round(time.time(), 1), **kw}, default=str)
    print(line, flush=True)
    with open(RESULTS, "a") as f:
        f.write(line + "\n")


def gpu() -> None:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True)  # fmt: skip
    say("gpu", nvidia_smi=out.stdout.strip(), err=out.stderr.strip()[:200])


def heartbeat() -> None:
    script = ("import time\nwhile True:\n    open('/content/heartbeat.txt','w').write("
              "str(time.time()))\n    time.sleep(30)\n")  # fmt: skip
    with open("/content/heartbeat.py", "w") as f:
        f.write(script)
    p = subprocess.Popen([sys.executable, "/content/heartbeat.py"], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    say("heartbeat", pid=p.pid, started=time.time())


def fetch_model() -> None:
    """The released model into MODEL_DIR (the HF load and the GGUF conversion both read it)."""
    # token=False: in a Colab kernel huggingface_hub otherwise asks Colab's secrets vault for
    # HF_TOKEN, which waits for the notebook UI; through the CLI it hung for 20 minutes (smoke 2).
    # The model is public. A child process with a time limit, so a hang can't hold the kernel.
    code = (f"from huggingface_hub import snapshot_download; "
            f"snapshot_download({MODEL!r}, local_dir={MODEL_DIR!r}, token=False)")  # fmt: skip
    t0 = time.time()
    try:
        env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=1200, env=env)  # fmt: skip
        ok = p.returncode == 0
        say("fetch_model", ok=ok, s=round(time.time() - t0, 1), dir=MODEL_DIR,
            **({} if ok else {"log": (p.stderr or p.stdout)[-1200:]}))  # fmt: skip
    except Exception:
        say("fetch_model", ok=False, error=traceback.format_exc()[-1200:])


def hf_load() -> None:
    """Smoke 1 failed here with Colab's own transformers ('list' object has no attribute 'keys'):
    the model card pins 4.57.1, so that is installed first (before transformers is imported), and
    the base loads through the trainer's `load_base`, as on the PC."""
    try:
        # pydantic-settings: the trainer's settings module needs it (smoke 2's first try failed)
        pins = [f"transformers=={TRANSFORMERS}", "peft>=0.17", "accelerate", "pydantic-settings"]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pins], check=False)
        import torch

        sys.path.insert(0, "/content/trainer/src")
        from asktrainer.modelio import load_base
        from peft import PeftModel

        t0 = time.time()
        base, tok, _info = load_base(MODEL_DIR, device="cuda", log=lambda *a: None)
        say("hf_base", s=round(time.time() - t0, 1),
            params=sum(p.numel() for p in base.parameters()),
            transformers=__import__("transformers").__version__,
            torch=torch.__version__)  # fmt: skip
        if not os.path.exists(ADAPTER):
            say("hf_adapter", skipped=f"{ADAPTER} not uploaded")
            return
        model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False)
        dtypes = sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" in n})
        msgs = [{"role": "user", "content": "How many rows does the table orders have?"}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to("cuda")  # fmt: skip
        with torch.no_grad():
            with_lora = model(ids).logits.float()
            with model.disable_adapter():
                without = model(ids).logits.float()
        diff = (with_lora - without).abs()
        say(
            "hf_adapter",
            lora_dtypes=dtypes,
            merged=False,
            logit_diff_mean=round(diff.mean().item(), 5),
            logit_diff_max=round(diff.max().item(), 4),
            same_next_token=bool((with_lora[0, -1].argmax() == without[0, -1].argmax()).item()),
        )
        out = model.generate(ids, max_new_tokens=96, do_sample=False, use_cache=True)
        say("hf_generate", text=tok.decode(out[0][ids.shape[1] :])[:400])
        del model, base
        torch.cuda.empty_cache()
    except Exception:
        say("hf_load", error=traceback.format_exc()[-1500:])


# --- llama.cpp with CUDA (design.md 7.3, plan B: vLLM refused Spark2_5ForCausalLM in smoke 1) ---
# Upstream llama.cpp has had Spark since 3ad1ba733 (#27868); b11779 is the build measured against
# the PC's fork (same perplexity and speed; TurboQuant doesn't help Spark's small KV cache).

LLAMA_TAG = "b11779"
LC = "/content/llama.cpp"
GGUF = "/content/spark-q8_0.gguf"
LORA_GGUF = "/content/step40-lora-f32.gguf"


def _sh(cmd: str, log: str, timeout: int = 3000) -> tuple[bool, str]:
    p = subprocess.run(["bash", "-c", f"set -o pipefail; {cmd} > {log} 2>&1"], timeout=timeout)
    tail = open(log).read()[-1200:] if os.path.exists(log) else ""
    return p.returncode == 0, tail


def llamacpp_build() -> None:
    t0 = time.time()
    ok, tail = _sh(
        f"[ -d {LC} ] || git clone -q --depth 1 --branch {LLAMA_TAG} "
        f"https://github.com/ggml-org/llama.cpp {LC}; cd {LC} && "
        # No web UI: the prebuilt UI download fetched a newer bundle than b11779 expects and
        # failed the build (smoke 2); the server's API needs no UI.
        "cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=OFF "
        "-DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF "
        "-DCMAKE_BUILD_TYPE=Release && cmake --build build -j $(nproc) "
        "--target llama-server llama-batched-bench",
        "/content/llamacpp_build.log",
    )
    say("llamacpp_build", ok=ok, s=round(time.time() - t0, 1), tag=LLAMA_TAG,
        **({} if ok else {"log": tail}))  # fmt: skip


def gguf_convert() -> None:
    """Q8_0 of the released model (as runs 1 and 2 sample) and step 40 as an f32 LoRA GGUF,
    applied at run time: never merged into bf16."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "protobuf"],
                   check=False)  # fmt: skip
    t0 = time.time()
    ok, tail = _sh(f"cd {LC} && python convert_hf_to_gguf.py {MODEL_DIR} --outtype q8_0 "
                   f"--outfile {GGUF}", "/content/gguf_model.log")  # fmt: skip
    say(
        "gguf_model",
        ok=ok,
        s=round(time.time() - t0, 1),
        gb=round(os.path.getsize(GGUF) / 1e9, 2) if ok else None,
        **({} if ok else {"log": tail}),
    )
    t0 = time.time()
    ok, tail = _sh(f"cd {LC} && python convert_lora_to_gguf.py {ADAPTER} --base {MODEL_DIR} "
                   f"--outtype f32 --outfile {LORA_GGUF}", "/content/gguf_lora.log")  # fmt: skip
    say("gguf_lora", ok=ok, s=round(time.time() - t0, 1),
        mb=round(os.path.getsize(LORA_GGUF) / 1e6, 1) if ok else None,
        **({} if ok else {"log": tail}))  # fmt: skip


def llamacpp_bench() -> None:
    """Total generated tokens a second at 1-64 parallel sequences, without and with the LoRA."""
    for name, extra in (("base", ""), ("lora", f"--lora {LORA_GGUF}")):
        ok, tail = _sh(f"{LC}/build/bin/llama-batched-bench -m {GGUF} {extra} -ngl 99 -c 65536 "
                       "-npp 512 -ntg 256 -npl 1,8,16,32,64 --output-format jsonl",
                       f"/content/bench_{name}.log", timeout=1500)  # fmt: skip
        rows = []
        for line in open(f"/content/bench_{name}.log"):
            if line.startswith("{"):
                d = json.loads(line)
                rows.append({k: d.get(k) for k in ("pl", "speed_tg", "speed", "t")})
        say(f"llamacpp_bench_{name}", ok=ok, rows=rows, **({} if rows else {"log": tail}))


def llamacpp_server() -> None:
    """llama-server with the LoRA at run time: one greedy answer with it and one with its scale
    at 0 (they must differ, as the HF check's logits do), compared with transformers' answer."""
    import urllib.request

    p = subprocess.Popen([f"{LC}/build/bin/llama-server", "-m", GGUF, "--lora", LORA_GGUF,
                          "-ngl", "99", "-np", "16", "-c", str(16 * 8192), "--cache-ram", "0",
                          "-ctxcp", "0", "--host", "127.0.0.1", "--port", "8091"],
                         stdout=open("/content/server.log", "w"),
                         stderr=subprocess.STDOUT)  # fmt: skip
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        out = {}
        for name, scale in (("lora", 1.0), ("base", 0.0)):
            body = {
                "messages": [
                    {"role": "user", "content": "How many rows does the table orders have?"}
                ],
                "temperature": 0,
                "max_tokens": 96,
                "lora": [{"id": 0, "scale": scale}],
            }
            req = urllib.request.Request(
                "http://127.0.0.1:8091/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            m = r["choices"][0]["message"]
            out[name] = ((m.get("reasoning_content") or "") + "|" + (m.get("content") or ""))[:400]
        say(
            "llamacpp_server",
            ok=True,
            lora_text=out["lora"],
            base_text=out["base"],
            adapter_changes_output=out["lora"] != out["base"],
        )
    except Exception:
        say(
            "llamacpp_server",
            ok=False,
            error=traceback.format_exc()[-1200:],
            log=open("/content/server.log").read()[-800:],
        )
    finally:
        p.terminate()


VLLM_SCRIPT = r"""
import json, os, sys, time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
MODEL, ADAPTER = sys.argv[1], sys.argv[2]
def say(step, **kw):
    print("RESULT " + json.dumps({"step": step, **kw}, default=str), flush=True)
t0 = time.time()
try:
    llm = LLM(MODEL, trust_remote_code=True, model_impl="transformers", dtype="bfloat16",
              enable_lora=True, max_lora_rank=16, gpu_memory_utilization=0.6,
              max_model_len=16384)
except Exception as e:
    say("vllm_load", ok=False, error=f"{type(e).__name__}: {str(e)[:600]}")
    sys.exit(0)
say("vllm_load", ok=True, s=round(time.time() - t0, 1))
lora = LoRARequest("step40", 1, ADAPTER)
tok = llm.get_tokenizer()
prompt = tok.apply_chat_template([{"role": "user", "content":
    "Write a SQL query that lists the ten newest orders."}], tokenize=False,
    add_generation_prompt=True)
out = llm.generate([prompt], SamplingParams(max_tokens=128, temperature=0), lora_request=lora)
say("vllm_generate", text=out[0].outputs[0].text[:400])
t1 = time.time()
outs = llm.generate([prompt] * 64, SamplingParams(max_tokens=512, temperature=1.0, seed=1),
                    lora_request=lora)
n = sum(len(o.outputs[0].token_ids) for o in outs)
say("vllm_throughput", sequences=64, tokens=n, s=round(time.time() - t1, 1),
    tokens_per_s=round(n / (time.time() - t1), 1))
"""


def vllm_probe() -> None:
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"],
                       capture_output=True, text=True)  # fmt: skip
    say("vllm_install", ok=r.returncode == 0, s=round(time.time() - t0, 1),
        err=r.stderr.strip()[-400:])  # fmt: skip
    if r.returncode != 0:
        return
    with open("/content/vllm_probe.py", "w") as f:
        f.write(VLLM_SCRIPT)
    p = subprocess.run([sys.executable, "/content/vllm_probe.py", MODEL, ADAPTER],
                       capture_output=True, text=True, timeout=1500)  # fmt: skip
    found = False
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            found = True
            d = json.loads(line[7:])
            say(d.pop("step"), **d)
    if not found:
        say("vllm_probe", error=(p.stderr or p.stdout).strip()[-800:])


def wandb_offline() -> None:
    """The smoke's results as an offline W&B run in /content/wandb (synced from the PC)."""
    # In a child process, not the kernel: inside a Colab kernel wandb asks Colab's secrets vault
    # for an API key, which waits for the notebook UI (it held the kernel in smoke 2).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb"], check=False)
    script = (
        "import json, wandb\n"
        f"rows = [json.loads(x) for x in open({RESULTS!r})]\n"
        "run = wandb.init(project='sql-cypher-rl', group='run3', job_type='colab-smoke',\n"
        "                 name='run3-colab-smoke', dir='/content',\n"
        "                 tags=['run3', 'phase0', 'colab', 'launched-by:agent'],\n"
        f"                 config={{'model': {MODEL!r}, 'adapter': 'run2-step40'}})\n"
        "for r in rows:\n"
        "    for k, v in r.items():\n"
        "        if isinstance(v, (int, float, str, bool)) and k != 't':\n"
        "            run.summary[r['step'] + '/' + k] = v\n"
        "run.finish()\n"
    )
    env = {**os.environ, "WANDB_MODE": "offline", "WANDB_SILENT": "true"}
    env.pop("WANDB_API_KEY", None)
    try:
        p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                           timeout=300, env=env)  # fmt: skip
        ok = p.returncode == 0
        say("wandb_offline", ok=ok, dir="/content/wandb",
            **({} if ok else {"log": (p.stderr or p.stdout)[-800:]}))  # fmt: skip
    except Exception:
        say("wandb_offline", ok=False, error=traceback.format_exc()[-800:])


if __name__ == "__main__":
    gpu()
    heartbeat()
