#!/bin/bash

echo "Run models explo and goal in minigrid envs"

declare -a ENVS=("grid_3x3" "grid_3x3_alias" "grid_4x4" "grid_4x4_alias")
declare -a MODELS=("ours_v3" "cscg_pose_ob" "cscg_pose_ob_random_policy")
declare -a NAV_TYPES=("exploration" "goal")
declare -a GOALS=(4 3 11)
declare -a X_POSES=(0 1 2)
declare -a Y_POSES=(0 1 1)

#TEST 0 : exploration
#test 1: goal

export TEST=$1

echo Test $TEST

id=0
MODEL='cscg_pose_ob'
ENV='grid_3x3'

# pkl_file=$(find results/${ENV}/cscg_exploration/${MODEL}/* -type d -exec ls -t {}/${MODEL}.pkl \; | tail -n2 | head -n1)
# echo loading pkl_file $pkl_file

for idx in $(seq 1 2); do
    id=$(( $id + 1 ))
    pkl_file=$(find results/${ENV}/cscg_exploration/${MODEL}/* -type d -exec ls -t {}/${MODEL}.pkl \; | tail -n$((id)) | head -n1)
    #pkl_file=$(find results/${ENV}/cscg_exploration/${MODEL}/* -type d -exec ls -t {}/${MODEL}.pkl \; | tail -n+$((idx)) | head -n1)
    # Use pkl_file with idx here
    echo "Processing pkl_file with index $idx: $pkl_file"
done
# echo Processing pkl_file with index $idx: $pkl_file
# exit
