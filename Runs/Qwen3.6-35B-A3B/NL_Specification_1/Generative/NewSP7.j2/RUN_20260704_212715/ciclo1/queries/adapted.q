// The robot shall navigate from the starting location to the coffee machine location upon mission initialization.
Pr[<=700] (<> pow(robPositionX[0] - coffee_room[0], 2) + pow(robPositionY[0] - coffee_room[1], 2) <= pow(30, 2))

// The robot shall navigate from the starting location to the coffee machine location upon mission initialization.
Pr[<=700] (<> pt_dist(robPositionX[0], location_X[2], robPositionY[0], location_Y[2]) <= 30)

// The robot must be able to reach the employee's locations from the coffee machine.
Pr[<=700] (<> Coffee_Delivery_robot1_def.A2_a3_deposit_complete)

// The robot must be able to reach the employee's locations from the coffee machine.
Pr[<=700] (<> Coffee_Delivery_robot1_def.A2_a3_deposit_complete)

// The robot must navigate without entering forbidden areas inside the university building.
Pr[<=700] (<> !isValidPosition(robPositionX[0], robPositionY[0]))

// The robot must perform exactly one cycle of the mission for each delivery request (no repetitions, no missing steps).
Pr[<=700] (<> currP > 2)

// The robot must not continue performing actions indefinitely once the delivery is completed.
Pr[<=700]([] (Coffee_Delivery.end imply (rm_1.idle && rf_1.idle && rp_1.idle && rd_1.idle)))

// Upon receiving a coffee request, the robot shall transition from Idle to an Active mode.
Pr[<=1]([] Coffee_Delivery_robot1_def.init_robot1)

// The robot must deliver the coffee to the intended employee.
Pr[<=700] (<> (pow(robPositionX[0] - office001[0], 2) + pow(robPositionY[0] - office001[1], 2) <= 900) imply Coffee_Delivery_robot1_def.A2_a3_deposit_complete)

// The robot must deliver the coffee to the intended employee.
Pr[<=700] (<> (pow(robPositionX[0] - office004[0], 2) + pow(robPositionY[0] - office004[1], 2) <= 900) imply Coffee_Delivery_robot1_def.A2_a3_deposit_complete)

// The robot must reach the coffee machine before picking up the coffee.
Pr[<=700] (<> (pow(robPositionX[0] - coffee_room[0], 2) + pow
