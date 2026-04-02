# This Website is Learning

This is a demo site meant to showcase a general-purpose reinforcement learning agent, implemented in JavaScript. The environment rewards the agent for being close to a small circle indicator, and needs to learn to navigate through applying force to a physics-governed circle. The implementation uses various techniques, including PPO, GAE+Bootstrapping, and PopArt. For more information on how this can be applied to multi-agent or research contexts, please see [CellTRIP](https://github.com/Oafish1/cellTRIP), which utilizes the similar methodologies to coordinate cells in a learned environment for in-silico simulation of feature perturbation and spatiotemporal imputation.

## How to Use

Simply [open the webpage](https://oafish1.github.io/ThisWebsiteIsLearning/) and allow it to run for ~10 iterations (~1 hour), at which time the circle, or agent, will have learned to approach the target dot. For a more challenging simulation, you may move the target during training. You may also adjust simulation speed in the top-left of the page.

## Features

- **Interactive Learning Visualization**: Watch the agent learn in real time as it discovers optimal navigation strategies.
- **Model Versioning and Reward**: Text on the circle indicates the version and current reward of the model.
- **Training Details and Monitorint**: Debug menu provides textual annotations during training, allowing observation of PopArt and model convergence.
- **Full JavaScript Implementation**: The site uses only static content written in JS using Tensorflow, allowing it to run without an API.
- **Consistent Physics**: Regardless of browser refresh rates or variable inter-frame delta times, the simulation clamps to tunable defined delta ranges to avoid large jumps.

## Design Decisions

- **Failure Case**: Upon exceeding the edge of the webpage, the episode is terminated and a negative reward is applied. Several other methodologies were tested, including bouncing off walls and never terminating.
- **Training Parameters**: The model updates with a batch size of `512` and an epoch size of `5120`, spanning `5` epochs per update iteration, balancing visible improvement between updates with optimal performance.
- **Rewards**: The agent is rewarded based on two criterion, namely, decreasing velocity and approaching the target dot.

## Potential Future Improvements

- **Vectorized Environment**: Parallelizing the environment would increase training speed dramatically, which is especially important for PPO - which is relatively sample inefficient.
- **Trainable Entropy**: Currently, the model outputs are drawn from a distribution with a standard deviation equivalent to half of the maximal model output by magnitude. Making this entropy trainable by predicting standard deviation or adding an entropy loss would allow the model to refine its movements more effectively.
- **Tunable Rewards and Training Parameters**: Allowing users to change training parameters could lead to more interesting interactions and education about fine-tuning.
- **Detach Rendering and Training**: Detaching simulation from rendering would allow the simulation to speed up beyond the browser refresh rate, better utilizing full GPU capabilities. In practice, however, running the model in FP32 slows computation beyond the point where this would be useful.
- **Background Training**: Currently, the model training freezes all interactable elements on the webpage, this is unideal for an interactive application.
- **Reward Visualization Plots**: Visualization of episode rewards over time, quantifying the training of the model.
- **General UI Improvements**: The UI is quite barebones at the moment, and could use some additional design.

## Demo Video


