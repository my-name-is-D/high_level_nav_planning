#!/bin/bash

echo "Run models explo and goal in minigrid envs"

declare -a ENVS=("3x3_grid" "3x3_grid_alias" "4x4_grid" "4x4_grid_alias")
declare -a MODELS=("ours_v3" "cscg" "cscg_random_policy")
declare -a NAV_TYPES=("exploration" "goal")
declare -a GOALS=(4 3 11)
declare -a X_POSES=(0 1 2)
declare -a Y_POSES=(0 1 1)

#TEST 0 : exploration
#test 1: goal

export TEST=$1

echo Test $TEST

id=0

#ENVIRONMENTS
for env_setting in $(seq 0 3)
	do
	ENV=${ENVS[$env_setting]}
	echo $ENV
    mkdir -p results/${ENV}
    #MODELS
    for m_id in $(seq 0 2)
        do
        MODEL=${MODELS[$m_id]}
	    echo $MODEL
        #NAV TYPES
        for nav in $(seq 0 1)
            do
            NAV_TYPE=${NAV_TYPES[$nav]}
            echo $NAV_TYPE
            #START POSES
            for sp in $(seq 0 2)
            do 
                X=${X_POSES[$sp]}
                Y=${Y_POSES[$sp]}
                echo start pose: (${X},${Y})
                
                #EXPLO TYPE
                if [$NAV_TYPE=='goal']
                    then
                    stop_condition='goal_reached'
                    #SETTING GOALS
                    for g in $(seq 0 2)
                    do 
                        GOAL = ${GOALS[$g]}
                        file=results/${ENV}/${MODEL}_${NAV_TYPE}_SP:(${X},${Y})_${GOAL}.txt
                        echo saving logs in ${file}
                        python python navigation_testbench.py --env ${ENV} --model ${MODEL} --max_steps 300 -p ${X} -p ${X} --goal ${GOAL} --stop_condition ${stop_condition} >> ${file}
                    done #GOALS
                    else
                    stop_condition='explo_done'
                    file=results/${ENV}/${MODEL}_${NAV_TYPE}_SP:(${X},${Y}).txt
                    echo saving logs in ${file}
                    
                    python python navigation_testbench.py --env ${ENV} --model ${MODEL} --max_steps 300 -p ${X} -p ${X} --stop_condition ${stop_condition} >> ${file}
                fi
                # id=$((id + 1))
            done #NAv type
            
    done #models
            
done #envs
exit



