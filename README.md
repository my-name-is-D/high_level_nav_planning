# higher_level_nav



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Tests

python navigation_testbench.py --env grid_3x3 --model ours_v4_2 --max_steps 200 -p 0 -p 0 --load_model ${pkl_file} --goal ${GOAL} --stop_condition ${stop_condition} 


--env options: 'grid_3x3' "grid_3x3_alias" "grid_4x4" "grid_4x4_alias" 'grid_donut' 'grid_t_maze' 'grid_t_maze_alias'
#there was 2nd Tolman maze as well but it seems I lost the config, can be redone in env/minigrid.py based on the paper appendix though

--model options: ours_v4_2 ours_v4 ours_v3 cscg cscg_pose_ob 'cscg_pose_ob_random_policy'
#the diverse options for our models are explained below

--max_steps: max number of steps for the navigation 
-p -p : starting tile pose in (x,y)
--load_model: should we use a prior map?
--goal : goal colour (as a number)
--stop_condition: under which condition to stop, exploration stops when connection looks like GT


## Versionning

Ours_v3: VANILLA type navigation with state inferred on latest action, observation A and B

Ours_v4: either VANILLA or MMP type navigation (switchable whenever during run -tried from MMP to VANILLA, not the contrary-), state inferring improved.

Ours_v4_2: added linear increase of policies, policies of different lengths and the lookahead is a distance instead of the number of consecutive actions. Basically useless policies like 4x STAY or move stay move or left right have been removed, transforming an exponential increase of policies wth lookahead to a linear (or polinomial with STAY) problem. 