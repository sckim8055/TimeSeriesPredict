#!/bin/bash
cd "$(dirname "$0")"
python3 run_fixed.py --leak --pred_len 1 > exp/log_leak_pred1.txt 2>&1
python3 run_fixed.py --pred_len 1 > exp/log_fixed_pred1.txt 2>&1
python3 run_fixed.py --pred_len 3 > exp/log_fixed_pred3.txt 2>&1
python3 run_fixed.py --pred_len 6 > exp/log_fixed_pred6.txt 2>&1
echo ALLDONE > exp/ALLDONE
