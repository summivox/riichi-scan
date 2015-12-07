#!/bin/bash -e
# File: run-batch.sh
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

while true; do
TIME=`date +"%d%M%S"`
./run.sh outputs/$TIME
sleep 0.5
done
