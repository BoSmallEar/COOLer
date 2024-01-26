#!/bin/bash
# launching the experiment, only needs --config and  --exp_name
status=0
while [ $status -eq 0 ]
do
    python run_scripts/train.py $1 train ${@:2}
    status=$?
    echo $status
    if [[ $status -ne 0 ]]; then
        break
    fi
    python run_scripts/predict.py $1 predict ${@:2}
    status=$?
    echo $status
done

 