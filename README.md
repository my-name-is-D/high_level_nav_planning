# higher_level_nav



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Tests

 python navigation_testbench.py --load_model results/grid_donut/ours_exploration/ours_v4_2_MMP/ours_v4_2_MMP_2024-02-27-09-32-43/ours_v4_2_MMP.pkl --inf_algo 'MMP' --env grid_donut --model ours_v4_2_MMP --max_steps 15 -p 0 -p 0 --goal 12 --stop_condition goal_reached


 python navigation_testbench.py --inf_algo 'MMP' --env grid_donut --model ours_v4_2 --max_steps 2 -p 0 -p 0






## Versionning

Ours_v3: VANILLA type navigation with state inferred on latest action, observation A and B

Ours_v4: either VANILLA or MMP type navigation (switchable whenever during run -tried from MMP to VANILLA, not the contrary-), state inferring improved.

Ours_v4_2: added linear increase of policies, policies of different lengths and the lookahead is a distance instead of the number of consecutive actions. Basically useless policies like 4x STAY or move stay move or left right have been removed, transforming an exponential increase of policies wth lookahead to a linear (or polinomial with STAY) problem. 