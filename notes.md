# Methods
- deep q learning in two games, discrete and continuous. and show the limitations of deep qlearning, as well as possible modifications to the algorithm for better performance in continuous action space

I am implementing two reinforcement learning models, q learning and deep q learning and testing the two models on two games from the open ai gym environment. the two games that are being tested on are cartpole and lunar lander. i need you to begin by writing me a professional introduction for a report paper. in the introduction, you can talk about the background of q learning, as well as the background or deep q learning. you can also talk about the difficulties of each algorithm, and how one came about deep q learning from q learning. this can be phrased in a way where you talk about the motivation of comparing these two methods. you may also add whatever you see fit in the introduction of a report paper.


# Q learning
- model free algorithm that learns to approximate the optimal q value function for a given env
- agent updates its qvalue esimates using bellman's equation. q learnign is a tabular method
- for cartpole, alpha = 0.25 is best





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
- learning rate might be too high -> wanders off of local minima, where the results is quite decent
- doesnt have a target network

# Future work
- Modification of deep q learning agorithms such as Deep deterministic policy gradient (ddpg) which combines q learning with policy gradients to learn a deterministic policy directly.



# To do
- add references all over the paper
- correct paper syntaxically
