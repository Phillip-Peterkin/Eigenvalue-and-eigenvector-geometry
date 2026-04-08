#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Running analysis pipeline ==="
echo "Step 1: Intracranial criticality analysis (ds004752)"
python analysis_pipeline/scripts/run_all_subjects.py

echo "Step 2: Broadband comparison"
python analysis_pipeline/scripts/run_all_subjects_broadband.py

echo "Step 3: ds004752 replication"
python analysis_pipeline/scripts/run_ds004752.py

echo "Step 4: Main analysis pipeline"
python analysis_pipeline/scripts/run_pipeline.py

echo "=== Analysis scripts ==="
for script in analysis_pipeline/scripts/analysis/_*.py; do
    echo "Running: $script"
    python "$script"
done

echo "=== Pipeline complete ==="
echo "Results saved to: ../results/"
