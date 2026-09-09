#!/usr/bin/env bash

set -euo pipefail

eval "$(python3 /load_secrets.py)"

: "${CASSANDRA_HOSTS:?Не задан CASSANDRA_HOSTS}"
: "${CASSANDRA_KEYSPACE:?Не задан CASSANDRA_KEYSPACE}"
: "${CASSANDRA_USER:?Не задан CASSANDRA_USER}"
: "${CASSANDRA_PASSWORD:?Не задан CASSANDRA_PASSWORD}"
: "${CASSANDRA_BOOTSTRAP_USER:?Не задан CASSANDRA_BOOTSTRAP_USER}"
: "${CASSANDRA_BOOTSTRAP_PASSWORD:?Не задан CASSANDRA_BOOTSTRAP_PASSWORD}"

HOST="${CASSANDRA_HOSTS%%,*}"
PORT="${CASSANDRA_PORT:-9042}"
SCHEMA_PATH="${SCHEMA_PATH:-/schema.cql}"

if ! [[ "$CASSANDRA_USER" =~ ^[a-z_][a-z0-9_]*$ ]]; then
    echo "CASSANDRA_USER должен быть идентификатором [a-z_][a-z0-9_]*" >&2
    exit 1
fi

if [ "$CASSANDRA_USER" = "$CASSANDRA_BOOTSTRAP_USER" ]; then
    echo "CASSANDRA_USER совпадает с CASSANDRA_BOOTSTRAP_USER: заведите" >&2
    echo "для сервиса отдельную роль" >&2
    exit 1
fi

if [[ "$CASSANDRA_PASSWORD" == *"'"* ]]; then
    echo "CASSANDRA_PASSWORD не должен содержать одинарную кавычку" >&2
    exit 1
fi

cqlsh_super() {
    cqlsh "$HOST" "$PORT" \
        -u "$CASSANDRA_BOOTSTRAP_USER" \
        -p "$CASSANDRA_BOOTSTRAP_PASSWORD" \
        "$@"
}

for attempt in $(seq 1 60); do
    if cqlsh_super -e "DESCRIBE CLUSTER" >/dev/null 2>&1; then
        echo "Суперпользователь доступен (попытка $attempt)"
        break
    fi

    if [ "$attempt" -eq 60 ]; then
        echo "Не удалось залогиниться суперпользователем" >&2
        exit 1
    fi

    sleep 5
done

echo "Применяем схему $SCHEMA_PATH"
cqlsh_super -f "$SCHEMA_PATH"

echo "Заводим роль $CASSANDRA_USER и выдаём права на $CASSANDRA_KEYSPACE"

cqlsh_super -e "CREATE ROLE IF NOT EXISTS $CASSANDRA_USER WITH PASSWORD = '$CASSANDRA_PASSWORD' AND LOGIN = true;"
cqlsh_super -e "GRANT SELECT ON KEYSPACE $CASSANDRA_KEYSPACE TO $CASSANDRA_USER;"
cqlsh_super -e "GRANT MODIFY ON KEYSPACE $CASSANDRA_KEYSPACE TO $CASSANDRA_USER;"

cqlsh_super -e "LIST ALL PERMISSIONS OF $CASSANDRA_USER;"
