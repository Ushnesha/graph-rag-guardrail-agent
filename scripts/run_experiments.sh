#!/bin/bash

#SBATCH --job-name=rag_llm_comparison
#SBATCH --partition=gpu                  # Adjust to your cluster's GPU partition name
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=8                # Request 8 CPU cores
#SBATCH --mem=64G                        # Request 64 GB system memory
#SBATCH --time=03:00:00                  # 3 hours max execution time
#SBATCH --output=results/job_%j.log       # Stdout log path (%j expands to job ID)
#SBATCH --error=results/job_%j.err        # Stderr log path

# Environment Python Path (direct resolution to bypass conda activation hook limitations)
ENV_PYTHON="/home/udaripa/projects/.conda/envs/ush_venv/bin/python"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/compare_models.py" ]; then
    COMPARE_SCRIPT="$SCRIPT_DIR/compare_models.py"
elif [ -f "$SCRIPT_DIR/../evaluation/compare_models.py" ]; then
    COMPARE_SCRIPT="$SCRIPT_DIR/../evaluation/compare_models.py"
elif [ -f "$SCRIPT_DIR/evaluation/compare_models.py" ]; then
    COMPARE_SCRIPT="$SCRIPT_DIR/evaluation/compare_models.py"
else
    COMPARE_SCRIPT="evaluation/compare_models.py"
fi
export HF_HUB_DISABLE_XET=1
JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"
JUDGE_PORT=11435
RAG_PORT=11434
HF_TOKEN=""
LIMIT=50
CONCURRENCY=5

# Models to test sequentially
MODELS=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "google/gemma-2-9b-it"
)

# Function to check if a port is in use and kill the process using it
cleanup_port() {
  local port=$1
  local pid=$(lsof -t -i:$port)
  if [ -n "$pid" ]; then
    echo "⚠️ Port $port is in use by PID $pid. Killing process..."
    kill -9 $pid
    sleep 2
  fi
}

# Function to wait for a port's /v1/models endpoint to return HTTP 200
wait_for_port() {
  local port=$1
  echo "⏳ Waiting for vLLM API on port $port to be ready..."
  while ! curl -s http://localhost:$port/v1/models | grep -q "data"; do
    sleep 5
  done
  echo "✅ Port $port is fully ready!"
}

# Ensure ports 11434 and 11435 are clean before starting
cleanup_port $RAG_PORT
cleanup_port $JUDGE_PORT

# 1. Start the Judge model (kept running throughout the entire experiment)
echo "🚀 Starting Judge Model: $JUDGE_MODEL on port $JUDGE_PORT..."
HF_TOKEN=$HF_TOKEN $ENV_PYTHON -m vllm.entrypoints.openai.api_server \
  --model $JUDGE_MODEL \
  --port $JUDGE_PORT \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.40 \
  --enable-chunked-prefill &
JUDGE_PID=$!

# Wait for the judge model to start
wait_for_port $JUDGE_PORT

# 2. Iterate and evaluate each model sequentially on port 11434
for MODEL in "${MODELS[@]}"; do
  echo "================================================================="
  echo "  🎯 STARTING EXPERIMENT FOR MODEL: $MODEL"
  echo "================================================================="

  # Clean up the port from previous run if any
  cleanup_port $RAG_PORT

  echo "🚀 Launching vLLM server for $MODEL on port $RAG_PORT..."
  # Allocating 0.45 VRAM allows plenty of KV cache space for a single model on an 80GB card
  HF_TOKEN=$HF_TOKEN $ENV_PYTHON -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $RAG_PORT \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.45 \
    --enable-chunked-prefill &
  RAG_PID=$!

  # Wait for this RAG model to initialize
  wait_for_port $RAG_PORT

  echo "📊 Running evaluation metrics and token tracking script..."
  $ENV_PYTHON "$COMPARE_SCRIPT" \
    --models "$MODEL:$RAG_PORT" \
    --limit $LIMIT \
    --concurrency $CONCURRENCY \
    --judge-model $JUDGE_MODEL \
    --judge-port $JUDGE_PORT

  echo "🛑 Stopping vLLM server for $MODEL (PID: $RAG_PID)..."
  kill $RAG_PID
  wait $RAG_PID 2>/dev/null
  
  echo "🧹 Cooling down GPU memory (10s)..."
  sleep 10
done

# 3. Clean up the Judge model
echo "🛑 Stopping Judge Model (PID: $JUDGE_PID)..."
kill $JUDGE_PID
wait $JUDGE_PID 2>/dev/null

echo "🎉 All sequential experiments completed! Consolidated report is saved at results/model_comparison_report.md"
