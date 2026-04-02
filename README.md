# This Website is Learning

This is a demo site meant to showcase a general-purpose reinforcement learning agent, implemented in JavaScript. The environment rewards the agent for being close to a small circle indicator, and needs to learn to navigate through applying force to a physics-governed circle. The implementation uses various techniques, including PPO, GAE+Bootstrapping, and PopArt. For seeing how this can be applied to multi-agent or research contexts, please see [CellTRIP](https://github.com/Oafish1/cellTRIP), which utilizes the same methodologies to coordinate cells in a learned environment for in-silico simulation of feature perturbation and spatiotemporal imputation.

## How to Use

Simply [open the webpage](https://oafish1.github.io/ThisWebsiteIsLearning/) and allow it to run for ~200 iterations (~1 hour), at which time the circle, or agent, will have learned to follow the dot - which is movable using your cursor. You may also adjust simulation speed in the top-left of the page.

## Features

- **Interactive Learning Visualization**: Watch the agent learn in real-time as it discovers optimal navigation strategies
- **Model Versioning and Reward**: Text on the circle indicates the version and current reward of the model
- **Training Details and Monitorint**: Debug menu provides textual annotations during training, allowing observation of PopArt and model convergence
- **Full JavaScript Implementation**: The site uses only static content written in JS using Tensorflow, allowing it to run without an API
- **Consistent Physics**: Regardless of browser refresh rates or variable inter-frame delta times, the simulation clamps to tunable defined delta ranges to avoid large jumps

## Design Decisions

- **Failure Case**: Upon exceeding the edge of the webpage, the episode is terminated and a negative reward is applied. Several other methodologies were tested, including bouncing off walls and never terminating
- **Training Parameters**: The model updates with a batch size of `64` and an epoch size of `320`, spanning `5` epochs per update iteration, balancing visible improvement between updates with optimal performance

## Potential Future Improvements

- **Detach Rendering and Training**: Detaching simulation from rendering would allow the simulation to speed up beyond the browser refresh rate, better utilizing full GPU capabilities. In practice, however, running the model in FP32 slows computation beyond the point where this would be useful.
- **Vectorized Environment**: Parallelizing the environment would increase training speed dramatically, which is especially important for PPO - which is relatively sample inefficient
- **Reward Visualization Plots**: Visualization of episode rewards over time, quantifying the training of the model

## Demo Video


