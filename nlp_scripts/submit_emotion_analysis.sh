#!/bin/bash
# Tries to get emotion_analysis.py onto a GPU (a100/h200/l40s/a40 -- p100/v100
# are excluded, this torch build has no kernels for their compute capability).
# If no GPU slot opens up within WAIT_SECS, falls back to the 8-core CPU job.
set -euo pipefail

NLP_DIR="/project/rise-phishing/kiva-prediction/nlp"
WAIT_SECS=180
POLL_INTERVAL=15

cd "$NLP_DIR"

gpu_job_id=$(qsub run_emotion_analysis_gpu.sh | grep -oE '[0-9]+' | head -1)
echo "Submitted GPU job $gpu_job_id, waiting up to ${WAIT_SECS}s for it to start running..."

elapsed=0
state=""
while [ "$elapsed" -lt "$WAIT_SECS" ]; do
    state=$(qstat -j "$gpu_job_id" >/dev/null 2>&1 && qstat -u "$USER" | awk -v id="$gpu_job_id" '$1==id {print $5}')
    if [ "$state" = "r" ]; then
        echo "GPU job $gpu_job_id is running. Done."
        exit 0
    fi
    if [ -z "$state" ]; then
        echo "GPU job $gpu_job_id is no longer queued (finished or errored) -- check emotion_analysis.qlog."
        exit 0
    fi
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
done

echo "No GPU slot freed up within ${WAIT_SECS}s (state=$state). Falling back to CPU."
qdel "$gpu_job_id" || true
cpu_job_id=$(qsub run_emotion_analysis_cpu.sh | grep -oE '[0-9]+' | head -1)
echo "Submitted CPU job $cpu_job_id."
