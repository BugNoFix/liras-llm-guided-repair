// Upon mission initialization, the robot shall transition from 'Idle' to an 'Active' mode.
Pr[<=1]([] user_guided_transport_r1_def.Init)

// Upon mission initialization, the human shall transition from 'Idle' to an 'Active' mode.
Pr[<=1]([] user_guided_transport_h1_def.Init)

// The human shall navigate from the initial location to the bedroom.
Pr[<=700] (<> pow(humanPositionX[0] - bedroom[0], 2) + pow(humanPositionY[0] - bedroom[1], 2) <= 900 )

// The robot shall navigate from the initial location to the bedroom.
Pr[<=700] (<> pow(robPositionX[0] - bedroom[0], 2) + pow(robPositionY[0] - bedroom[1], 2) <= 900 )

// The robot must not attempt to pick up the object before reaching the bedroom.
Pr[<=700] (<> !rp_1.idle && !(pow(robPositionX[0] - bedroom[0], 2) + pow(robPositionY[0] - bedroom[1], 2) <= 900) )

// The robot must not attempt to pick up the object before reaching the bedroom.
Pr[<=700] (<> user_guided_transport_r1_def.s2_start && !(pow(robPositionX[0] - bedroom[0], 2) + pow(robPositionY[0] - bedroom[1], 2) <= 900) )

// The robot must not attempt to pick up the object unless the human is also present in the bedroom.
Pr[<=700] (<> !rp_1.idle && !(pow(robPositionX[0] - bedroom[0], 2) + pow(robPositionY[0] - bedroom[1], 2) <= 900) )

// The robot must not attempt to pick up the object unless the human is also present in the bedroom.
Pr[<=700] (<> user_guided_transport_r1_def.s2_start && !(pow(robPositionX[0] - bedroom[0], 2) + pow(robPositionY[0] - bedroom[1], 2) <= 900) )

// The robot must successfully grasp exactly one object in the bedroom.
Pr[<=700](<> (pow(robPositionX[0] - bedroom
