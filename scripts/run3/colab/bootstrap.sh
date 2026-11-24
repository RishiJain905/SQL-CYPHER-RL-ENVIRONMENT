#!/usr/bin/env bash
# Run 3's environment on a fresh Colab VM (root, Ubuntu, no Docker): the five engines, the
# Python packages, the two public repos, the data bundle, then the environment check.
#
#   colab ssh -s run3 -- bash < scripts/run3/colab/bootstrap.sh
#   (or: colab exec -s run3 with a %%bash cell holding this file)
#
# Expects the bundle at /content/run3-bundle.tar.gz (colab upload, or copied from Drive by a
# person who mounted it). Every engine listens on 127.0.0.1 only and has no password: nothing
# outside the VM can reach it, and every model query runs as the read-only run3_reader.
# Safe to run twice. NOT RUN YET on a real VM: the paid smoke (documentation/run3/README.md)
# is its first run, and its output decides what to fix.
set -euo pipefail

DATA=/content/run3data
REPO=/content/rl
TRAINER=/content/trainer
NEO4J_VERSION=5.26.12   # the PC runs neo4j:5.26 (the checker's major.minor)
log() { echo "[bootstrap $(date +%H:%M:%S)] $*"; }

log "packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates gnupg lsb-release >/dev/null
# PostgreSQL 16 from the PostgreSQL project's own repository (the PC runs 16)
if ! command -v /usr/lib/postgresql/16/bin/postgres >/dev/null; then
  install -d /usr/share/postgresql-common/pgdg
  curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc
  echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] \
https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
    > /etc/apt/sources.list.d/pgdg.list
  apt-get update -qq
fi
apt-get install -y -qq postgresql-16 mysql-server >/dev/null
apt-get install -y -qq openjdk-21-jre-headless >/dev/null 2>&1 \
  || apt-get install -y -qq openjdk-17-jre-headless >/dev/null

log "postgresql: trust on loopback only"
HBA=/etc/postgresql/16/main/pg_hba.conf
cat > "$HBA" <<'EOF'
local all all trust
host  all all 127.0.0.1/32 trust
host  all all ::1/128 trust
EOF
sed -i "s/^#\?listen_addresses.*/listen_addresses = '127.0.0.1'/" /etc/postgresql/16/main/postgresql.conf
echo "max_connections = 200" >> /etc/postgresql/16/main/postgresql.conf
service postgresql restart

log "mysql: lower_case_table_names=1 needs a fresh data folder"
service mysql stop || true
cat > /etc/mysql/mysql.conf.d/run3.cnf <<'EOF'
[mysqld]
bind-address = 127.0.0.1
lower_case_table_names = 1
local_infile = 1
max_connections = 200
innodb_buffer_pool_size = 2G
sql_mode = ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION
[client]
local_infile = 1
EOF
if [ ! -f /var/lib/mysql/.run3 ]; then
  rm -rf /var/lib/mysql && mkdir -p /var/lib/mysql && chown mysql:mysql /var/lib/mysql
  mysqld --initialize-insecure --user=mysql --lower-case-table-names=1
  touch /var/lib/mysql/.run3
fi
service mysql start
mysql -uroot -e "CREATE USER IF NOT EXISTS 'root'@'127.0.0.1'; \
  GRANT ALL ON *.* TO 'root'@'127.0.0.1' WITH GRANT OPTION;"

log "neo4j ${NEO4J_VERSION}"
if [ ! -d /opt/neo4j ]; then
  curl -fsSL "https://dist.neo4j.org/neo4j-community-${NEO4J_VERSION}-unix.tar.gz" | tar xz -C /opt
  mv "/opt/neo4j-community-${NEO4J_VERSION}" /opt/neo4j
  cat >> /opt/neo4j/conf/neo4j.conf <<'EOF'
dbms.security.auth_enabled=false
server.default_listen_address=127.0.0.1
server.memory.heap.max_size=4g
server.memory.pagecache.size=2g
EOF
fi
/opt/neo4j/bin/neo4j start || true

log "python packages and the two public repos"
pip install -q "duckdb==2.0.1" "psycopg[binary]" pymysql sqlglot  # duckdb as on the PC
[ -d "$REPO" ] || git clone -q https://github.com/RishiJain905/SQL-CYPHER-RL-ENVIRONMENT "$REPO"
[ -d "$TRAINER" ] || git clone -q https://github.com/RishiJain905/nfl-ask-trainer "$TRAINER"

log "data"
mkdir -p "$DATA"
BUNDLE=${BUNDLE:-/content/run3-bundle.tar.gz}
if [ -f "$BUNDLE" ] && [ ! -f "$DATA/.unpacked" ]; then
  tar xzf "$BUNDLE" -C "$DATA" && touch "$DATA/.unpacked"
fi
export RUN3_DATA="$DATA" RUN3_TRAINER_REPO="$TRAINER" PYTHONPATH="$REPO/src"
for i in $(seq 1 60); do curl -fs http://127.0.0.1:7474 >/dev/null && break; sleep 2; done

log "port every SQLite source into DuckDB, PostgreSQL, MySQL"
cd "$REPO"
python -m sqlrl.run3.porting setup
for src in $(ls "$DATA/dbs"); do
  ids=$(ls "$DATA/dbs/$src" | sed -n 's/\.sqlite$//p')
  args=$(for d in $ids; do printf -- "--db %s " "$d"; done)
  python -m sqlrl.run3.porting port --source "$src" $args > "$DATA/port-$src.jsonl"
done

log "check"
python -m sqlrl.run3.check_env --source "${CHECK_SOURCE:-birddev}" --db "${CHECK_DB:-superhero}"
log "done"
