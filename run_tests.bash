#!/bin/bash

echo "Run models explo and goal in minigrid envs"

declare -a ENVS=('grid_3x3' "grid_3x3_alias" "grid_4x4" "grid_4x4_alias")
declare -a MODELS=('cscg_pose_ob' 'cscg_pose_ob_random_policy')
declare -a NAV_TYPES=("goal")
declare -a GOALS=(8 3 4 5)
declare -a X_POSES=(0 1 0 0)
declare -a Y_POSES=(0 2 1 3)

#TEST 0 : exploration
#test 1: goal

export TEST=$1

echo Test $TEST


#ENVIRONMENTS
for env_setting in $(seq 0 3)
do
    ENV=${ENVS[$env_setting]}
    echo $ENV
    mkdir -p results/${ENV}

    if [ "$ENV" == "grid_3x3" ] || [ "$ENV" == "grid_3x3_alias" ] 
    then
        max_steps=400
    else
        max_steps=600
    fi

    #MODELS
    for m_id in $(seq 0 1)
    do
        MODEL=${MODELS[$m_id]}
        echo $MODEL
        id=0
        #NAV TYPES
        for nav in $(seq 0 1)
        do
            NAV_TYPE=${NAV_TYPES[$nav]}
            echo $NAV_TYPE
            

            #START POSES
            for sp in $(seq 0 3)
            do 
                X=${X_POSES[$sp]}
                Y=${Y_POSES[$sp]}
                echo "start pose: (${X},${Y})"

                
                #EXPLO TYPE
                if [ $NAV_TYPE == 'goal' ]
                then
                    stop_condition='goal_reached'
                    
                    #SETTING GOALS
                    for g in $(seq 0 3)
                    do 
                        GOAL=${GOALS[$g]}
                        if [ "$MODEL" == "cscg_pose_ob" ] || [ "$MODEL" == "cscg_pose_ob_random_policy" ] 
                        then
                            # Use the pkl file from the last folder of 'results/cscg_exploration'
                            id=$(( $id + 1 ))
                            pkl_file=$(find results/${ENV}/cscg_exploration/${MODEL}/* -type d -exec ls -t {}/${MODEL}.pkl \; | tail -n$((id)) | head -n1)
                            echo loading pkl_file $pkl_file 
                            python navigation_testbench.py --env ${ENV} --model ${MODEL} --max_steps ${max_steps} -p ${X} -p ${Y} --load_model ${pkl_file} --goal ${GOAL} --stop_condition ${stop_condition} #>> ${file}
                        else
                            id=$(( $id + 1 ))
                            pkl_file=$(find results/${ENV}/ours_exploration/${MODEL}/* -type d -exec ls -t {}/${MODEL}.pkl \; | tail -n$((id)) | head -n1)
                            python navigation_testbench.py --load_model ${pkl_file} --env ${ENV} --model ${MODEL} --max_steps ${max_steps} -p ${X} -p ${Y} --goal ${GOAL} --stop_condition ${stop_condition} #>> ${file}
                        fi
                    done #GOALS
                else
                    stop_condition='explo_done'
                    #file=results/${ENV}/${MODEL}_${NAV_TYPE}_SP:"("${X},${Y}")".txt
                    #echo saving logs in ${file}
                    
                    python navigation_testbench.py --env ${ENV} --model ${MODEL} --max_steps ${max_steps} -p ${X} -p ${Y} --stop_condition ${stop_condition} #>> ${file}
                fi
                # id=$((id + 1))
            done #START POSES
        done #NAV TYPES
    done #MODELS
done #ENVIRONMENTS
exit
