#!/usr/bin/env bash
# Start a Qdrant server (binary already on PATH) with storage on disk, then list the collections.
# Settings for running Qdrant on Colab or a Hugging Face Space.
QDRANT__STORAGE__STORAGE_PATH="${QDRANT_DATA:-./qdrant_data}" \
QDRANT__TELEMETRY_DISABLED=true \
QDRANT__STORAGE__PERFORMANCE__OPTIMIZERS_CPU_BUDGET=1 \
qdrant > /tmp/qdrant.log 2>&1 &
sleep 5
curl -s http://127.0.0.1:6333/collections
