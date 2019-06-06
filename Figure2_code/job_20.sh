#!/bin/sh
max=11
for i in `seq 2 $max`
do
python3 system_control.py $i 20 200 10
done

