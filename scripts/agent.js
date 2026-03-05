// Wait for backend
await tf.ready();

// Global variables
var std = 0.1;

// Log probability calculation for Gaussian distribution: https://stats.stackexchange.com/questions/7440/likelihood-function-of-normal-distribution/7441#7441
function gaussianLogProb(action, mu, sigma) {
    return tf.sum(tf.sub(- .5 * Math.log(2 * Math.PI * std * std), tf.square(tf.sub(action, mu)).div(2 * sigma * sigma)), 1);
}

// Public function to get force from model
function _getForce(x, y, vx, vy, target_x, target_y, delta_time) {
    // Plug into model
    let input_tensor = tf.tensor2d([[x, y, vx, vy, target_x, target_y]]);
    let action_means = actor_model.predict(input_tensor);

    // Sample action from Gaussian distribution with mean from model and fixed std
    let action = tf.randomNormal([2], 0, std).reshape(action_means.shape).add(action_means);

    // Compute previous reward
    let reward_prerequisites = computeRewardPrerequisites(x, y, vx, vy, target_x, target_y);
    updatePreviousReward(reward_prerequisites, delta_time);

    // Record experience for training, including critic evaluation and log likelihood
    let action_cpu = action.dataSync();
    memory.push({
        state: [x, y, vx, vy, target_x, target_y],
        reward_prerequisites: reward_prerequisites,
        reward: 0,
        action: action_cpu, 
        log_prob: gaussianLogProb(action, action_means, std).dataSync()[0],
        value: critic_model.predict(input_tensor).dataSync()[0] * return_std + return_mean,
        terminal: false,
        truncated: false
    });

    // Return
    return { x: action_cpu[0], y: action_cpu[1] };
}
export const getForce = (...args) => tf.tidy(() => _getForce(...args));  // Wrap in tidy to prevent memory leaks

// Get previous reward
export function getPreviousReward() {
    if (memory.length > 1) {
        return memory[memory.length - 2].reward;
    }
    return 0;
}

// Reward function prerequisites
function computeRewardPrerequisites(x, y, vx, vy, target_x, target_y) {
    let distance_to_target = Math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2);
    // let velocity_mag = Math.sqrt(vx * vx + vy * vy);  // Proper
    let velocity_mag = (Math.abs(vx) + Math.abs(vy)) / 2;  // L1 norm
    return { distance: distance_to_target, velocity: velocity_mag };
}

// Reward function based on distance to target and low velocity
function updatePreviousReward(new_reward_prerequisites, delta_time) {
    if (memory.length > 0 && !memory[memory.length - 1].terminal && !memory[memory.length - 1].truncated) {
        // Get old reward prerequisites and initialize
        let action = memory[memory.length - 1].action;
        let old_reward_prerequisites = memory[memory.length - 1].reward_prerequisites;

        // Distance reward
        let dist_change = new_reward_prerequisites.distance - old_reward_prerequisites.distance;
        let dist_reward = - dist_change * 4e0 / delta_time;

        // Velocity penalty
        let square_velocity_change = new_reward_prerequisites.velocity ** 2 - old_reward_prerequisites.velocity ** 2;
        let vel_reward = - square_velocity_change * 1e2 / delta_time;

        // Action penalty
        let action_reward = - Math.sqrt(action[0] * action[0] + action[1] * action[1]) * 1e0;

        // Record
        let reward = 10 * dist_reward + 0 * vel_reward + 0 * action_reward;  // 10 2 0
        memory[memory.length - 1].reward += 1e-2 * reward;  // Scale down reward to speed up initial convergence by the critic

        // console.log("Reward: " + memory[memory.length - 1].reward.toFixed(3) + " (Dist: " + dist_reward.toFixed(3) + ", Vel: " + vel_reward.toFixed(3) + ", Act: " + action_reward.toFixed(3) + ")");
    }
}

// Modify agent's reward by a certain amount (e.g. for hitting walls)
// Doesn't matter too much if it's on this or the next reward
export function modifyReward(amount, terminal=false, truncated=false) {
    if (memory.length > 0) {
        memory[memory.length - 1].reward += amount;
        memory[memory.length - 1].terminal = terminal;
        memory[memory.length - 1].truncated = truncated;
    }
}

// Synchronous boolean masking
// https://github.com/tensorflow/tfjs/issues/380#issuecomment-3474357825
function _booleanMask(tensor, mask) {
    let count = tf.sum(mask).arraySync();
    let indices = tf.range(0, tensor.shape[0]);
    let top = tf.topk(tf.where(mask, indices, tf.scalar(-1)), count);
    return tf.gather(tensor, top.indices);
}
export const booleanMask = (...args) => tf.tidy(() => _booleanMask(...args));

// Train the model using PPO
function _trainModel() {
    // Parameters
    // 5 | 512 | 10 - Works well, a bit slow
    // 5 | 64 | 10 - Works, avoids walls around V106
    let epochs = 5;
    let batch_size = 512;
    let min_memories = 10 * batch_size;

    // Prepare training data from memory
    let states = memory.map(exp => exp.state);
    let rewards = memory.map(exp => exp.reward);
    let actions = memory.map(exp => exp.action);
    let log_probs = memory.map(exp => exp.log_prob);
    let values = memory.map(exp => exp.value);
    let terminals = memory.map(exp => exp.terminal);
    let truncateds = memory.map(exp => exp.truncated);

    // Omit last state and action since they don't have a subsequent reward
    // Could also consider computing one more state, but omit for less frontent calling
    let last_state = states.pop();
    rewards.pop();
    actions.pop();
    log_probs.pop();
    values.pop();
    terminals.pop();
    truncateds.pop();

    // Apply GAE to rewards using critic model
    let advantages = [];
    let usable_mask = [];
    let next_advantage = 0;
    // Bootstrap in case truncated
    let next_state_value = critic_model.predict(tf.tensor2d([last_state])).arraySync()[0][0] * return_std + return_mean;  // Unnormalize value for GAE calculation
    let gamma = 0.99;  // Discount factor
    let lambda = 0.95;
    let terminates = false;
    for (let i = rewards.length - 1; i >= 0; i--) {
        // Set next state value to 0 if terminal
        if (terminals[i]) {
            next_advantage = 0;
            next_state_value = 0;
            terminates = true;
        }
        // If next is truncated, use state to bootstrap
        else if (i+1 < truncateds.length && truncateds[i+1]) {
            next_advantage = 0;
            next_state_value = critic_model.predict(tf.tensor2d([states[i]])).arraySync()[0][0] * return_std + return_mean;
            terminates = false;

            advantages.unshift(next_advantage);
            usable_mask.unshift(false);
            continue;  // Don't use truncated states for training, but do use for bootstrapping next state value
        }
        let delta = rewards[i] + gamma * next_state_value - values[i];
        next_advantage = delta + gamma * lambda * next_advantage;
        next_state_value = values[i];
        advantages.unshift(next_advantage);
        // Only use if not truncated or last, and if terminates eventually
        // usable_mask.unshift(terminates);
        // usable_mask.unshift(true); 
        usable_mask.unshift(true);
    }

    // If there are no terminated episodes, just use truncated
    // if (!usable_mask[0]) {
    //     usable_mask = Array(usable_mask.length).fill(true);  // Could also consider truncateds
    //     console.log("No terminated episodes, using all memories for training.");
    // }

    // Convert to tensors
    states = tf.tensor2d(states);
    rewards = tf.tensor1d(rewards);
    actions = tf.tensor2d(actions);
    log_probs = tf.tensor1d(log_probs);
    advantages = tf.tensor1d(advantages);
    values = tf.tensor1d(values);

    // Check if number of usable memories is sufficient for training
    // TODO: Could potentially lock up, if there is a terminal episode then an indefinite one. FIX
    let num_memories = usable_mask.reduce((sum, usable) => sum + (usable ? 1 : 0), 0);
    if (num_memories <= min_memories) {
        console.log("Not enough usable memories to train (" + num_memories + "), skipping");
        return false;
    }

    // Filter out unterminated memories
    usable_mask = tf.tensor1d(usable_mask, 'bool');
    states = booleanMask(states, usable_mask);
    rewards = booleanMask(rewards, usable_mask);
    actions = booleanMask(actions, usable_mask);
    log_probs = booleanMask(log_probs, usable_mask);
    advantages = booleanMask(advantages, usable_mask);
    values = booleanMask(values, usable_mask);

    // Iterate epochs and train on batches
    console.log("Training on " + states.shape[0] + " states");
    for (let epoch = 0; epoch < epochs; epoch++) {
        // Shuffle indices at the start of the epoch
        let indices = tf.range(0, states.shape[0]).arraySync();
        tf.util.shuffle(indices);
        indices = tf.tensor1d(indices, 'int32');

        for (let i = 0; i < states.shape[0]; i += batch_size) {
            // Subset detached data for batch
            let max_index = Math.min(i + batch_size, states.shape[0]);
            let batch_indices = indices.slice(i, max_index-i);
            let batch_states = tf.gather(states, batch_indices);
            // let batch_rewards = tf.gather(rewards, batch_indices);
            let batch_actions = tf.gather(actions, batch_indices);
            let batch_log_probs = tf.gather(log_probs, batch_indices);
            let batch_advantages = tf.gather(advantages, batch_indices);
            let batch_values = tf.gather(values, batch_indices);

            // Normalize advantages
            let adv_mean = tf.mean(batch_advantages);
            let adv_std = tf.sqrt(tf.mean(tf.square(batch_advantages.sub(adv_mean))));
            let normalized_batch_advantages = batch_advantages.sub(adv_mean).div(adv_std.add(1e-8));
            
            // Train actor model using continuous PPO loss
            let actor_loss = actor_optimizer.minimize(() => {
                let action_means = actor_model.predict(batch_states);
                // let action_diffs = tf.sub(batch_actions, action_means);
                let log_probs = gaussianLogProb(batch_actions, action_means, std);
                let ratios = tf.exp(tf.sub(log_probs, batch_log_probs));
                let surrogate1 = tf.mul(ratios, normalized_batch_advantages);
                let eps = .2;
                let surrogate2 = tf.mul(tf.clipByValue(ratios, 1 - eps, 1 + eps), normalized_batch_advantages);
                let actor_loss = tf.neg(tf.mean(tf.minimum(surrogate1, surrogate2)));
                return actor_loss;
            }, true);  // , actor_model.trainableVariables

            // Compute returns by adding batch_advantages to batch_values
            let batch_returns = batch_advantages.add(batch_values);
            let normalized_batch_returns = batch_returns.sub(return_mean).div(return_std + 1e-8);

            // Train critic model using MSE
            let critic_loss = critic_optimizer.minimize(() => {
                let values = critic_model.predict(batch_states).reshape([-1]);
                let critic_loss = tf.losses.meanSquaredError(normalized_batch_returns, values);
                return critic_loss;
            }, true);  // , critic_model.trainableVariables

            // Print training losses
            // console.log("Epoch " + epoch + ", Batch " + (i / batch_size) + ": Actor Loss = " + actor_loss.arraySync().toFixed(3) + ", Critic Loss = " + critic_loss.arraySync().toFixed(3));
            
            // Update running mean and std
            let beta = 3e-3;
            let old_return_mean = return_mean;
            let old_return_std = return_std;
            return_mean = (1 - beta) * return_mean + beta * tf.mean(batch_returns).arraySync();  // ArraySync prevents NaN from combining tensor and scalar
            return_square_mean = (1 - beta) * return_square_mean + beta * tf.mean(tf.square(batch_returns)).arraySync();
            return_std = Math.sqrt(return_square_mean - return_mean ** 2);

            // Update last layer of critic model according to PopArt normalization
            let last_layer_weights = critic_model.layers[critic_model.layers.length - 1].getWeights();
            let new_weights = last_layer_weights[0].mul(old_return_std).div(return_std);
            let new_bias = last_layer_weights[1].mul(old_return_std).add(old_return_mean).sub(return_mean).div(return_std);
            critic_model.layers[critic_model.layers.length - 1].setWeights([new_weights, new_bias]);
        }
    }

    // Increment model iteration
    model_iteration++;

    // Clear used memories after training, i.e. those in usable mask
    // IMPORTANT: Assumes that ending memories are contiguous
    memory = memory.filter((_, index) => !usable_mask.arraySync()[index]);  // Could optimize
    console.log("Leaving " + memory.length + " memories for next training iteration");

    // Debugging
    console.log("Return mean (" + return_mean.toFixed(3) + "), Return std (" + return_std.toFixed(3) + ")");
    // actor_model.layers[0].getWeights()[0].print();
    // critic_model.layers[0].getWeights()[0].print();

    return true;
}
export const trainModel = (...args) => tf.tidy(() => _trainModel(...args));

export function getModelVersion() {
    return model_iteration;
}

// Create model
var model_iteration = 0;
const actor_model = tf.sequential();
actor_model.add(tf.layers.dense({ inputShape: [6], units: 32 }));
actor_model.add(tf.layers.activation({ activation: "relu" }));
actor_model.add(tf.layers.dense({ units: 32 }));
actor_model.add(tf.layers.activation({ activation: "relu" }));
actor_model.add(tf.layers.dense({ units: 2 }));
actor_model.add(tf.layers.activation({ activation: "tanh" }));

// Create critic model
const critic_model = tf.sequential();
critic_model.add(tf.layers.dense({ inputShape: [6], units: 32 }));
critic_model.add(tf.layers.activation({ activation: "relu" }));
critic_model.add(tf.layers.dense({ units: 32 }));
critic_model.add(tf.layers.activation({ activation: "relu" }));
critic_model.add(tf.layers.dense({ units: 1 }));

// Store running mean and std for reward normalization in critic
var return_mean = 0;
var return_square_mean = 1;
var return_std = 1;

// Create optimizers with weight decay
const actor_optimizer = tf.train.adam(3e-4, undefined, undefined, 1e-5);
const critic_optimizer = tf.train.adam(3e-4, undefined, undefined, 1e-5);

// Initialize memory buffer for training
let memory = [];
