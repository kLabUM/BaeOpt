#!/bin/sh
max=11
for i in `seq 2 $max`
do	
	echo $i
	python3 system_control.py $i 42 20 10
done

