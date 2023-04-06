# Methods
- deep q learning in two games, discrete and continuous. and show the limitations of deep qlearning, as well as possible modifications to the algorithm for better performance in continuous action space


# AC
- plotting total rewards a lot of squiggles, even if it goes upwards over time

# SAC
- get robust and stable learning in continuous action space environments
- more smooth upward trend, since it maximizes not only the total reward over time, but also the stochasticity/entropy of how the agent behaves

# Deep Q Learning
- handles discrete action spaces well
- faces difficulty when it comes to continuous action space, such as the inverted pendulum
- use Huber loss: acts like the mean squaraed error when error is small, acts like mean absolute error when its large
- Huber loss makes it more robust to outliers when the estimates of Q are very noisy.

## reasons
- number of actions is very large or even infinite
- this makes it computationally infeasible to represente and store qvalues for every action
- results in non convergence or very slow convergence, which is impractical


# Future work
- Modification of deep q learning agorithms such as Deep deterministic policy gradient (ddpg) which combines q learning with policy gradients to learn a deterministic policy directly.



