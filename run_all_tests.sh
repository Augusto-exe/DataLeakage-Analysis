#!/bin/bash

source venv/bin/activate

if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Please install Python3 to proceed."
    exit 1
fi

# echo "Running normalization test..."
# python3 main.py normalization

# echo "Running feature selection test..."
# python3 main.py feature_selection

# echo "Running imputation test..."
# python3 main.py imputation

echo "Running tuning test..."
python3 -W ignore main.py tuning

# echo "Running SMOTE test..."
# python3 main.py smote

#echo "Running worst case leakage test..."
#python3 main.py worst_case

echo "All tests completed."