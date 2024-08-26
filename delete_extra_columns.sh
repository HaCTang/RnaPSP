#!/bin/bash

# Define the directory path containing CSV files
directory_path="/home/thc/RnaPSP/RnaPSP/all data"

# Define the full path to the Python script
python_script_path="/home/thc/RnaPSP/RnaPSP/data_preparation/delete_extra_columns.py"

# Call the Python script with the directory path as an argument
python3 "$python_script_path" "$directory_path"
