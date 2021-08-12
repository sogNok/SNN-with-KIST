#!/bin/bash

for (( i=40; i<100; i++ ))
do
    python GRecg.py --seed=$i
done

exit 0
