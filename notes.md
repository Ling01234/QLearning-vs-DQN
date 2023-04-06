# AC
- plotting total rewards a lot of squiggles, even if it goes upwards over time

# SAC
- get robust and stable learning in continuous action space environments
- more smooth upward trend, since it maximizes not only the total reward over time, but also the stochasticity/entropy of how the agent behaves

# Deep Q Learning
- handles discrete action spaces well
- faces difficulty when it comes to continuous action space, such as the inverted pendulum

## reasons
- number of actions is very large or even infinite
- this makes it computationally infeasible to represente and store qvalues for every action
- results in non convergence or very slow convergence, which is impractical


# Future work
- Modification of deep q learning agorithms such as Deep deterministic policy gradient (ddpg) which combines q learning with policy gradients to learn a deterministic policy directly.