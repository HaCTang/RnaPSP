#!/bin/bash

FOLDER_PATH="/home/thc/RnaPSP/RnaPSP/data_preparation"
cd "$FOLDER_PATH"

FILES_TO_RUN=(
    "wash_all_data.py"
    "base_ratio.py"
    "evenness.py"
    "self_correlation.py"
    "physics_based_pred.py"
    "rna_classification.py"
)

for FILE in "${FILES_TO_RUN[@]}"; do
    echo "Running $FILE..."
    python "$FILE"

    if [ $? -ne 0 ]; then
        echo "Error: $FILE failed. Stopping execution."
        exit 1
    fi
done

echo "All scripts have been run."
