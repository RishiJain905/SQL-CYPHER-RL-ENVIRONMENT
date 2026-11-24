#!/usr/bin/env bash
# Run 3, phase 2: the Colab VM's setup for the RL run, run detached and logged:
#   nohup bash /content/rl_setup.sh > /content/setup.log 2>&1 &
# Expects the bundle's parts (run3-rl.part-*, run3-rl.sha256) and bootstrap.sh in /content.
# Stages print "[setup HH:MM:SS] ..."; the last line is "SETUP DONE" or "SETUP FAILED <stage>".
# llama.cpp builds and the models download while the engines install and the databases port.
set -uo pipefail
log() { echo "[setup $(date +%H:%M:%S)] $*"; }
fail() { log "SETUP FAILED $*"; exit 1; }
cd /content

log "bundle: reassemble and check"
cat run3-rl.part-* > run3-bundle.tar.gz || fail bundle
echo "$(cat run3-rl.sha256)  run3-bundle.tar.gz" | sha256sum -c || fail bundle-hash

log "llama.cpp b11779 with CUDA, no web UI (background)"
( [ -d llama.cpp ] || git clone -q --depth 1 --branch b11779 https://github.com/ggml-org/llama.cpp llama.cpp
  cd llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DLLAMA_CURL=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF -DCMAKE_BUILD_TYPE=Release > /content/llama-build.log 2>&1 \
  && cmake --build build -j "$(nproc)" --target llama-server >> /content/llama-build.log 2>&1 \
  && touch /content/LLAMA_OK ) &
BUILD=$!

log "models: the released Q8_0 and the HF weights (background; public, no token)"
( python - <<'EOF' > /content/models-dl.log 2>&1 && touch /content/MODELS_OK
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download("trjxter/Spark-X2.6-4B-SQL-Cypher-Agent-GGUF",
                "Spark-X2.6-4B-SQL-Cypher-Agent-Q8_0.gguf", local_dir="/content/release", token=False)
snapshot_download("trjxter/Spark-X2.6-4B-SQL-Cypher-Agent", local_dir="/content/model", token=False)
print("ok")
EOF
) &
DL=$!

log "engines, packages, repos, porting (bootstrap)"
CHECK_SOURCE=bird CHECK_DB=human_resources BUNDLE=/content/run3-bundle.tar.gz bash bootstrap.sh \
  > /content/bootstrap.log 2>&1 || { tail -30 /content/bootstrap.log; fail bootstrap; }
tail -25 /content/bootstrap.log

log "python pins (smoke 2): transformers 4.57.1, peft, pydantic-settings; torchao removed"
pip install -q "transformers==4.57.1" "peft>=0.17" accelerate pydantic-settings safetensors \
  >> /content/pip.log 2>&1 || fail pip
pip uninstall -y -q torchao >> /content/pip.log 2>&1 || true

log "paths config for the VM"
mkdir -p /content/rl/configs /content/models /content/rldata
cat > /content/rl/configs/paths.local.toml <<'EOF'
[paths]
data = '/content/rldata'
model = '/content/release/Spark-X2.6-4B-SQL-Cypher-Agent-Q8_0.gguf'
evalset = '/content/none'
greedy_scores = '/content/none'
trainer_repo = '/content/trainer'
wsl_venv = '/usr'
model_hf = '/content/model'
model_gguf = '/content/release/Spark-X2.6-4B-SQL-Cypher-Agent-Q8_0.gguf'
llama_cpp_src = '/content/llama.cpp'
models = '/content/models'
EOF

log "gold check again on this VM's engine versions (the bundle's databases)"
export RUN3_DATA=/content/run3data RUN3_TRAINER_REPO=/content/trainer PYTHONPATH=/content/rl/src
cd /content/rl
python - <<'EOF' > /content/goldcheck.log 2>&1 || fail goldcheck
import json
from sqlrl.run3 import pools
dbs = json.load(open("/content/run3data/rl/bundle-dbs.json"))
print(json.dumps(pools.run_check("bird", dbs)))
EOF
tail -2 /content/goldcheck.log
cd /content

log "waiting for the models and the llama.cpp build"
wait $DL; [ -f /content/MODELS_OK ] || { tail -5 /content/models-dl.log; fail models; }
wait $BUILD; [ -f /content/LLAMA_OK ] || { tail -20 /content/llama-build.log; fail llama-build; }
ls -la /content/release /content/llama.cpp/build/bin/llama-server
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
log "SETUP DONE"
