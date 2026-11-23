# Guide: Google Colab for run 3

Run 3's RL phase runs on a Colab A100, capped at 5 to 7 hours (design 7). This guide explains
the tool that lets an agent drive Colab, how run 3 uses it, and what is still unknown.

## 1. A primer

**Colab** rents a virtual machine with a GPU, normally used through notebooks in a browser.
Each hour on a GPU spends **compute units** from the account (the A100 was about 7.5 units an
hour in March 2026; `colab usage` shows today's rate). The machine is temporary: when the
session ends, its disk is gone.

**The Colab CLI** (`google-colab-cli`, Google's, Apache-2.0, 0.8.0 in November 2026) does the
same from a terminal, which is what lets an agent use it:

| Command | Does | Costs units |
|---|---|---|
| `colab sessions`, `colab usage`, `colab status -s NAME` | look | no |
| `colab new -s NAME --gpu A100` | rents a machine | **yes, until stopped** |
| `colab exec -s NAME -f script.py` | runs a local script on it (the kernel keeps its state between calls) | while it runs |
| `colab upload` / `download` / `ls` | moves files | while the machine is up |
| `colab stop -s NAME` | gives the machine back | stops the cost |
| `colab run --gpu A100 script.py` | new + exec + stop in one | while it runs |

It runs on Linux and macOS only: on this PC it lives in WSL (`uv tool install
google-colab-cli`), outside the repo.

## 2. Signing in (Rishi, once)

The CLI signs in with a browser consent: the first command prints a Google link; Rishi opens it,
allows Colab, and pastes the code back into the terminal. The token is kept in the operating
system's keyring, never in a file an agent reads. Agents never handle the code or the token;
they check that it works with `colab sessions` (read-only).

## 3. How run 3 uses it

```
PC (WSL)                                   Colab VM (A100)
---------                                  ---------------
bundle.py -> run3-bundle.tar.gz  --upload-->  /content/run3-bundle.tar.gz
step 40's adapter                --upload-->  /content/step40-adapter/
                                            bootstrap.sh: PostgreSQL, MySQL, Neo4j, DuckDB,
                                              the repos, port every database, check_env
                                            the RL phase (later): every step saves its adapter
each step's adapter             <--download--  /content/run3/...
colab stop -s run3
```

**Why not Drive:** mounting Drive (`colab drivemount`) needs a person to click a consent in
every session, so an agent couldn't resume a run alone after a session ends. Files go up with
`colab upload` and each step's adapter comes back with `colab download`; the PC keeps the run's
history, as it does for runs 1 and 2. Rishi can still mount Drive by hand when he is there.

**The engines on the VM** come from packages, not Docker (Colab has none): PostgreSQL 16 from
the PostgreSQL project's repository, MySQL from Ubuntu's, Neo4j from its tarball with a Java
runtime, DuckDB and SQLite from pip. All listen on the VM's loopback only, without passwords;
every model query runs as the read-only `run3_reader`. The VM re-runs the gold check on its own
engine versions.

## 4. What the paid smoke has to find out (about 0.5 h, not run yet)

| Unknown | How `gpu_smoke.py` answers it |
|---|---|
| A100 40 GB or 80 GB (the CLI can't choose) | `nvidia-smi` |
| Does a process started from a cell outlive it? (the backend may reclaim an idle kernel) | a heartbeat file written every 30 s, read later with `colab download` |
| Does the model load on NVIDIA with its own code and step 40's adapter unmerged? | transformers + PEFT, one greedy answer, the adapter's dtype |
| Does vLLM load it (the fast sampler) with the adapter at run time? | vLLM's Transformers backend |

Then `colab stop`. A cell that is still running blocks the next `exec` on that session, so a
long run is watched through files (`download`, `ls`), not by running code beside it.

## 5. Rules

- Every `colab new` is spending: only with Rishi's OK, with the hours stated first, and always
  followed by `colab stop`. `colab sessions` after any interruption, to stop orphans.
- Run 3's total is 5 to 7 hours: smoke 0.5, setup 0.5, RL about 5, warm start at most 1 if it
  can't run on the PC.
- Nothing secret goes to the VM: no `.netrc`, no tokens. W&B runs offline on the VM
  (`WANDB_MODE=offline`) and is synced from the PC afterwards.
- Never `colab repl`, bare `colab ssh`, `colab auth` or `colab drivemount` from an agent: they
  wait for a person.
