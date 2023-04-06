# Methods
- deep q learning in two games, discrete and continuous. and show the limitations of deep qlearning, as well as possible modifications to the algorithm for better performance in continuous action space


# Q learning
- model free algorithm that learns to approximate the optimal q value function for a given env
- agent updates its qvalue esimates using bellman's equation. q learnign is a tabular method
-



# Deep Q Learning
- uses experience replay
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



