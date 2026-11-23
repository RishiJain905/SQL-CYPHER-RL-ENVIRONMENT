# Guide: run 3's environment

Run 3 puts the model in front of real databases (design `run3/design.md`). This guide explains
the environment as built: what an engine is here, how one question becomes a question on five
engines, what an episode looks like, how it is scored, and how to look at it. Operating
commands are in `run3/README.md`.

## 1. A primer: why five engines

SQL is a family of languages, not one. The same question needs different text on different
engines:

Measured on this environment's copies of BIRD's superhero database (2026-11-23):

| Query | SQLite | DuckDB | PostgreSQL | MySQL |
|---|---|---|---|---|
| `SELECT superhero_name, count(*) FROM superhero` (a column beside an aggregate, no GROUP BY) | `3-D Man, 750` | error, with a hint | error | error (1140) |
| `... WHERE full_name = 1` (a text column against a number) | 0 | **error**: can't convert 'Charles Chandler' to INT32 | **error**: text = integer | 0 |
| `SELECT 5 / 2` | 2 | 2.5 | 2 | 2.5000 |
| `... WHERE superhero_name = 'a-bomb'` (the data says `A-Bomb`) | 0 | 0 | 0 | **1** (text compared without case) |

SQLite answers the first query with one arbitrary row's name beside the count; the other three
refuse it. Runs 1 and 2 checked every SQL answer on SQLite, so the model learned SQLite's
forgiving habits;
on DuckDB, a bare column beside GROUP BY failed (the NFL-DB benchmark). Run 3 asks each question
on every engine where its gold is proven to work, so the model has to write each engine's own
dialect, and read each engine's own error when it doesn't.

Neo4j is the fifth engine: a graph database queried with Cypher (`MATCH (a)-[:R]->(b) RETURN
...`).

## 2. The engines and how read-only is enforced

| Engine | On the PC | On Colab | A database is | The engine itself refuses writes by |
|---|---|---|---|---|
| SQLite | Python's module | same | a `.sqlite` file | `mode=ro`, `query_only`, an authorizer |
| DuckDB | the `duckdb` package | same | a `.duckdb` file | `read_only=True`, no file or network access |
| PostgreSQL 16 | container `run3-postgres` | package | database `r3_<source>_<db>` | role `run3_reader` (SELECT only), read-only transactions |
| MySQL 8 | container `run3-mysql` | package | database `r3_<source>_<db>` | user `run3_reader` (SELECT only), read-only transactions |
| Neo4j 5.26 | container `run3-neo4j` | tarball | the one graph loaded now | READ transactions, no procedures but the schema listings |

Before the engine sees a query, a gate refuses anything that isn't one read statement of a kind
the engine uses for reading and looking around (SQLite `PRAGMA table_info`, DuckDB `SHOW TABLES`
/ `DESCRIBE`, MySQL `SHOW` / `DESCRIBE`, PostgreSQL `TABLE`). The gate's job is a plain message;
the engine's own rule is the guard. `check_env` proves it by sending a write past the gate to
each engine and expecting the engine to refuse it:

```
ok    sqlite refuses a write by itself  refused: not authorized
ok    duckdb refuses a write by itself  refused: Invalid Input Error: Cannot execute statement of type "CREATE" ...
ok    postgres refuses a write by itself  refused: cannot execute CREATE TABLE in a read-only transaction
ok    mysql refuses a write by itself  refused: (1792, 'Cannot execute statement in a READ ONLY transaction.')
```

Every engine: a 5-second limit a query (`QUERY_TIME_S`), at most 10,000 rows read (more shows as
"more than 10,000"), and the engine's own error text. The error's first line is always shown;
the engine's hint lines (DuckDB's "Either add it to the GROUP BY list, or use ANY_VALUE(...)",
PostgreSQL's `HINT:`) are shown in some episodes and not others, because real harnesses differ.

## 3. One question on four SQL engines

**Porting** (`porting.py`). Every source database is SQLite (BIRD, Spider). It is copied into
DuckDB, PostgreSQL and MySQL by rules that keep the data as it is:
- a column's type comes from what it holds: integers only -> BIGINT, with reals -> DOUBLE,
  anything with text -> text, blobs -> binary. A `DATE` column that holds text stays text;
- PostgreSQL gets plain names in lower case (`Customer` -> `customer`); a name with a space or
  a symbol keeps its spelling and must be quoted, as in any real database;
- MySQL ignores the case of table names (`lower_case_table_names=1`);
- no keys or constraints in the copies; the prompt shows the source's foreign keys.

Loading is fast: superhero (10 tables), student_club and california_schools each take under a
second an engine.

**The gold check** (`gold.py`, `pools.py check`). For each question, the SQLite gold runs on
SQLite, then is translated with sqlglot into each engine's dialect and run there. The pair
(question, engine) is kept only if the result is the SQLite result under the checker's
comparison (below) and took at most half the episode's time limit. On the first three BIRD-dev
databases (376 questions):

| Engine | Kept | Most common reasons a gold failed |
|---|---|---|
| SQLite | 376 | - |
| DuckDB | 362 | rows differ (4), bare column beside GROUP BY, text that isn't a number |
| PostgreSQL | 354 | `operator does not exist: text = integer` (12), bare column beside GROUP BY |
| MySQL | 355 | rows differ (14, mostly its case-insensitive text comparison), GROUP BY |

A question is never asked on an engine where its gold didn't pass.

## 4. An episode

```
system   the engine and its version, the tools, the schema (full / names only / none), the rules
user     the question (and BIRD's evidence line, "(Note: ...)")
model    thinks, calls one tool                    <- trained on
tool     the result: rows, "0 rows", or "Error: ..."   <- read, never trained on
model    calls again ... up to 8 calls
model    ends: submit_answer(query) | decline(reason) | clarify(question)
```

**What is drawn per episode** in training (`protocol.draw_setting`; evaluation uses one fixed
setting): the schema shown (full 50%, names 30%, none 20%), the tool set (the query tool alone,
or with helpers), the tools' spellings (`run_query`, `execute_sql`, `query_database`, ...;
`submit_answer`, `submit_query`, `final_answer`, ...), rows shown (5, 10, 20, 50), the result's
form (table, JSON lines, CSV), whether error hints are shown.

**The helpers** (`env.py`, `graphs.run_helper`): `list_tables`, `describe_table(table)`,
`sample_rows(table, n)` and `find_values(table.column, text)`, which shows how the data spells
something (`find_values("Customer.city", "osl")` -> `Oslo`). They run as ordinary read-only
queries; the names they are given are matched against the catalog and quoted, never pasted in.

**Several sources:** a SQL question with the graph copy of the same database beside it (two
query tools, `run_query_1` and `query_graph_2`, for example). The result decides, so a right
answer through either one counts.

## 5. The reward

| Ending | Reward |
|---|---|
| `submit`, right | 1, less 0.02 for each call after the second (at least 0.9) |
| `submit`, wrong, or right rows that were typed in as constants | 0 |
| `decline` on an unanswerable question or a write request | 1 |
| `clarify` naming every meaning | 1 |
| `decline` / `clarify` on a clear question | 0 (and counted) |
| no ending, a malformed one, or a call after the last allowed one | -0.1 |

"Right" is the trainer's checker's comparison (`querycheck.align`): the same rows, columns in
any order, rows in any order unless the gold sorts them, numbers at the gold's precision, and
a different choice among rows tied at the gold's `LIMIT` is right too. Values are made
comparable first: PostgreSQL's and MySQL's `Decimal` become numbers, dates become ISO text.
A submitted query that only types in the answer (`SELECT 'Ann', 3`) earns nothing (run 2's
constants rule).

Each record also says whether the last query the model ran would have been right
(`last_query_ok`): a diagnostic of what the endings cost, never part of the reward.

## 6. Graph copies

A SQL database becomes a property graph (`graphs.convert`): rows become nodes labelled with the
table's name in CamelCase, declared foreign keys become relationships typed from the column
(`publisher_id` -> `PUBLISHER`), and a table that only joins two others becomes relationships
of its own name. Real numbers:

| Database | Nodes | Relationships | Join tables turned into relationships | Round trip |
|---|---|---|---|---|
| superhero | 1,051 | 14,800 | hero_attribute, hero_power | every row back |
| student_club | 42,185 | 540 | attendance | every row back |
| california_schools | 29,941 | 12,044 | - | every row back |

superhero's schema as the model reads it:

```
Nodes (properties are STRING unless typed):
- Superhero {full_name, height_cm: INTEGER, id: INTEGER, superhero_name, weight_kg: INTEGER}
- Superpower {id: INTEGER, power_name}
...
Relationships:
- (Superhero)-[:HERO_ATTRIBUTE {attribute_value}]->(Attribute)
- (Superhero)-[:HERO_POWER]->(Superpower)
- (Superhero)-[:PUBLISHER]->(Publisher)
```

The copy agrees with its source: "heroes with Intelligence" is 156 rows in PostgreSQL and 156
in Cypher. Neo4j Community holds one database, so episodes take a graph through a lock and
are grouped by graph (`GraphHost.acquire`); loading superhero takes under a second.

CypherBench's questions run on the benchmark's sampled graphs, as in runs 1 and 2 (soccer:
1,828 training questions, every gold runs).

## 7. Non-answers

| Kind | Built from | Right ending |
|---|---|---|
| Write request | templates over the database's real tables and columns ("Delete every row from the frpm table where ...") | `decline` |
| Wrong database | a question of another database whose gold reads no table that exists here | `decline` |
| Ambiguous | needs a writer (the teacher, paid): not built yet | `clarify` |

## 8. Looking at it

- A smoke's `episodes.jsonl` holds every turn (the model's text, the call, the result as the
  model saw it) and the grade; `summary.json` the counts by engine.
- The gold check's records: `<data>/run3/pools/goldcheck/<source>/<db>.jsonl`, one a question,
  with the translated gold per engine and why a pair was dropped.
- The databases themselves: any SQL client on the loopback ports, Neo4j Browser on
  `http://127.0.0.1:7477`.

## 9. Changing it safely

- Any change to what the model is shown (the prompt, the tools, result rendering, the
  limits) changes what an evaluation measures: record it and re-measure the start point.
- A new engine: an engine class in `engines.py` (gate, read-only role, timeout, error text),
  a dialect in `gold.SQLGLOT_DIALECT`, a port function, a `check_env` line.
- A new source: a loader in `sources.py`, then port and gold-check.

## 10. Limits and what's next

- 5-second query limit and a 300,000-row cap on graph copies: chosen for the PC, re-set in
  phase 0 on the A100.
- MySQL compares text without case by default; questions whose result depends on case are
  dropped for MySQL by the gold check.
- The rollouts for training will build the context token by token (run 2's rule that a tool
  result can't forge a turn); the smoke's policy renders text, which is fine for measuring.
