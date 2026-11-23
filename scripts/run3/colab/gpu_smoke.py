"""The paid Colab smoke's GPU half (design.md 7.1: 0.5 h). Runs on a VM made by
`colab new -s run3 --gpu A100`; never trains.

The kernel keeps its state between `colab exec` calls, so this file is sent once (it defines the
steps and runs the first two) and each later step is one call:

    colab exec -s run3 -f scripts/run3/colab/gpu_smoke.py    # gpu(), heartbeat()
    echo "hf_load()" | colab exec -s run3
    echo "vllm_probe()" | colab exec -s run3                 # installs vLLM, runs it apart
    echo "wandb_offline()" | colab exec -s run3              # the results as an offline W&B run

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

MODEL = "trjxter/Spark-X2.6-4B-SQL-Cypher-Agent"
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


def hf_load() -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=False)
        from peft import PeftModel

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="cuda")  # fmt: skip
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
        out = model.generate(ids, max_new_tokens=96, do_sample=False)
        say("hf_generate", text=tok.decode(out[0][ids.shape[1] :])[:400])
        del model, base
        torch.cuda.empty_cache()
    except Exception as e:
        say("hf_load", error=f"{type(e).__name__}: {str(e)[:600]}")


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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb"], check=False)
    os.environ["WANDB_MODE"] = "offline"
    import wandb

    rows = [json.loads(x) for x in open(RESULTS)]
    run = wandb.init(project="sql-cypher-rl", group="run3", job_type="colab-smoke",
                     name="run3-colab-smoke", dir="/content",
                     tags=["run3", "phase0", "colab", "launched-by:agent"],
                     config={"model": MODEL, "adapter": "run2-step40"})  # fmt: skip
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float, str, bool)) and k not in ("t",):
                run.summary[f"{r['step']}/{k}"] = v
    run.finish()
    say("wandb_offline", dir="/content/wandb")


if __name__ == "__main__":
    gpu()
    heartbeat()
