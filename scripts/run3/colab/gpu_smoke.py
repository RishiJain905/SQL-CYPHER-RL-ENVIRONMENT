"""The paid Colab smoke's GPU half (design.md 7.1: 0.5 h). NOT RUN YET: it runs only after
Rishi's OK, on a VM made by `colab new -s run3 --gpu A100`, after `bootstrap.sh`.

    colab exec -s run3 -f scripts/run3/colab/gpu_smoke.py

It answers what can't be known from the PC, one JSON line each, and never trains:
1. which GPU the account got (the CLI has no 40 / 80 GB choice) and its memory;
2. whether a detached process outlives the cell that started it (a heartbeat file, read
   later with `colab download`): how a 5-hour run must be launched;
3. whether the released model loads on NVIDIA with its own code (trust_remote_code) and step
   40's adapter applied unmerged (PEFT), and in what dtype the adapter's matrices sit;
4. whether vLLM loads the model (its Transformers backend) with the adapter at run time.

Step 40's anchor check (design 5.1) is phase 0's next step on this VM, with the run-2 evalset:
not this script.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

MODEL = "trjxter/Spark-X2.6-4B-SQL-Cypher-Agent"
ADAPTER = "/content/step40-adapter"  # uploaded: the PEFT folder (adapter_config.json + weights)


def say(step: str, **kw) -> None:
    print(json.dumps({"step": step, **kw}, default=str), flush=True)


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
    say(
        "heartbeat",
        pid=p.pid,
        started=time.time(),
        check="colab download -s run3 /content/heartbeat.txt hb.txt (later; compare times)",
    )


def hf_load() -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="cuda")  # fmt: skip
        say(
            "hf_base",
            s=round(time.time() - t0, 1),
            params=sum(p.numel() for p in base.parameters()),
        )
        if not os.path.exists(ADAPTER):
            say("hf_adapter", skipped=f"{ADAPTER} not uploaded")
            return
        model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False)
        dtypes = sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" in n})
        say("hf_adapter", lora_dtypes=dtypes, merged=False,
            note="bf16 lora matrices = the change is computed in bf16 at run time; check it "
                 "against the anchor before trusting it")  # fmt: skip
        msgs = [{"role": "user", "content": "Write a SQLite query that counts the rows of t."}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(
            "cuda"
        )
        out = model.generate(ids, max_new_tokens=64, do_sample=False)
        say("hf_generate", text=tok.decode(out[0][ids.shape[1] :])[:300])
    except Exception as e:
        say("hf_load", error=f"{type(e).__name__}: {str(e)[:400]}")


def vllm_load() -> None:
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        t0 = time.time()
        llm = LLM(MODEL, trust_remote_code=True, model_impl="transformers", dtype="bfloat16",
                  enable_lora=os.path.exists(ADAPTER), max_lora_rank=16,
                  gpu_memory_utilization=0.5, max_model_len=16384)  # fmt: skip
        say("vllm_load", s=round(time.time() - t0, 1))
        lora = LoRARequest("step40", 1, ADAPTER) if os.path.exists(ADAPTER) else None
        out = llm.generate(["SELECT"], SamplingParams(max_tokens=16, temperature=0),
                           lora_request=lora)  # fmt: skip
        say("vllm_generate", text=out[0].outputs[0].text)
    except Exception as e:
        say("vllm_load", error=f"{type(e).__name__}: {str(e)[:400]}")


if __name__ == "__main__":
    gpu()
    heartbeat()
    hf_load()
    vllm_load()
