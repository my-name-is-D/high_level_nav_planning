#!/bin/bash

echo "Run models explo and goal in minigrid envs"

MODEL='ours_v4_2'
GOAL=0
X=5
Y=2
max_steps=35
n_runs=70

#AGENT ID 0
#test 1: goal

export agent_ID=$1
echo agent_ID: $agent_ID

for RUN in $(seq 40 $n_runs)
do
    if [ $RUN == 0 ]
    then
        max_step=25
    else
        max_step=$max_steps
    fi
    python3 navigation_testbench.py --env 'grid_cross_tunnels' --agent ${agent_ID} --run ${RUN} --model ${MODEL} --max_steps ${max_step} -p ${X} -p ${Y} --goal ${GOAL}
done #RUN
exit
