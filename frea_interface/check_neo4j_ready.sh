#!/bin/bash

# Ensure Neo4j is running locally and accessible
HOST=localhost
PORT=7687
MAX_RETRIES=10
RETRY_DELAY=3

echo "Checking if Neo4j is running on $HOST:$PORT..."
for ((i=1;i<=MAX_RETRIES;i++)); do
  nc -z $HOST $PORT && echo "Neo4j is up!" && exit 0
  echo "Attempt $i: Neo4j not available yet, retrying in $RETRY_DELAY seconds..."
  sleep $RETRY_DELAY
done

echo "ERROR: Neo4j is not running on $HOST:$PORT after $MAX_RETRIES attempts."
exit 1
