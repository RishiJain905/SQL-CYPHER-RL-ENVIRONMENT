# SQL-CYPHER-RL-ENVIRONMENT

Reinforcement learning for a small database agent: SQL and Cypher, on a single 16 GB consumer GPU
(AMD RX 7800 XT).

The starting point is
[trjxter/Spark-X2.6-4B-SQL-Cypher-Agent](https://huggingface.co/trjxter/Spark-X2.6-4B-SQL-Cypher-Agent),
a 4B tool-calling agent that answers questions about a database with one read-only SQLite SQL or
Neo4j Cypher query (43.7% execution accuracy on its test set, against 32.5% for the base model).
It was fine-tuned in [nfl-ask-trainer](https://github.com/RishiJain905/nfl-ask-trainer). This repo
holds the RL environments, the training loop and the measurements.

## The plan

[`documentation/rl.md`](documentation/rl.md):

1. **The sampling baseline:** each evalset question answered 8 times, sampled; pass@1 against
   pass@8 shows how much room RL has.
2. **The method pilots:** short runs of GRPO-family methods, SDPO and SRPO, scored the same way.
3. **The real RL run:** single-turn with an execution reward, then a multi-turn agent in real
   databases.

Status: planning. Nothing has run yet.
