#!/bin/sh
max=11
python3 system_control.py 1 42 500 1
python3 system_control.py 2 42 500 1 
for i in `seq 3 $max`
do
	echo $i
	python3 system_control.py $i 42 500 10
done
