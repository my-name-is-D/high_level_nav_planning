# higher_level_nav



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Tests



## Versionning

Ours_v3: VANILLA type navigation with state inferred on latest action, observation A and B

Ours_v4: either VANILLA or MMP type navigation (switchable whenever during run -tried from MMP to VANILLA, not the contrary-), state inferring improved.

Ours_v4_2: added linear increase of policies, policies of different lengths and the lookahead is a distance instead of the number of consecutive actions. Basically useless policies like 4x STAY or move stay move or left right have been removed, transforming an exponential increase of policies wth lookahead to a linear (or polinomial with STAY) problem. 