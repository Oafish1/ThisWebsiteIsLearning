// Global variables
var std = 1;

// Public function to get force from model
export function getForce(x, y, vx, vy, target_x, target_y) {
    // Plug into model
    let input_tensor = tf.tensor2d([[x, y, vx, vy, target_x, target_y]]);
    let output_tensor = model.predict(input_tensor);
    let output_array = output_tensor.arraySync()[0];

    // Assess with critic
    let value = critic_model.predict(input_tensor).arraySync()[0][0];
    value = value * return_std + return_mean;

    // Sample from Gaussian distribution with mean from model
    output_array = output_array.map(val => val + tf.randomNormal([1], 0, std).arraySync()[0]);
    let log_prob = -0.5 * output_array.reduce((sum, val) => sum + (val * val) / (std * std), 0);

    // Get reward
    let reward = getReward(x, y, vx, vy, target_x, target_y);

    // Record experience for training, including critic evaluation and log likelihood
    memory.push({ state: [x, y, vx, vy, target_x, target_y], reward: reward, reward_mod: 0, action: output_array, log_prob: log_prob, value: value });

    // Return
    return { x: output_array[0], y: output_array[1], reward: reward };
}

// Reward function based on distance to target and low velocity
function getReward(x, y, vx, vy, target_x, target_y) {
    let distance_to_target = Math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2);
    let velocity_mag = Math.sqrt(vx * vx + vy * vy);
    return -distance_to_target // - .1 * velocity_mag;
}

// Modify agent's reward by a certain amount (e.g. for hitting walls)
// Doesn't matter too much if it's on this or the next reward
export function modifyReward(amount) {
    if (memory.length > 0) {
        memory[memory.length - 1].reward_mod += amount;
    }
}

// Train the model using PPO
export function trainModel() {
    // Prepare training data from memory
    let states = memory.map(exp => exp.state);
    let rewards = memory.map(exp => exp.reward);
    let actions = memory.map(exp => exp.action);
    let log_probs = memory.map(exp => exp.log_prob);
    let values = memory.map(exp => exp.value);

    // Get differences between subsequent rewards and add reward modifications
    let reward_diffs = [];
    for (let i = 1; i < rewards.length; i++) {
        reward_diffs.push(rewards[i] - rewards[i - 1] + memory[i].reward_mod);
    }

    // Omit last state and action since they don't have a subsequent reward
    // Could also consider using the reward of the last state, but for less frontent calling
    let last_state = states.pop();
    rewards = reward_diffs;
    actions.pop();
    log_probs.pop();
    values.pop();

    // Apply GAE to rewards using critic model
    values = tf.tensor1d(values);  // Cast values to tensor
    let advantages = [];
    let gae = critic_model.predict(tf.tensor2d([last_state])).arraySync()[0][0];  // Bootstrap last state value
    values = values.mul(return_std).add(return_mean);  // Unnormalize values for GAE calculation
    let gamma = 0.99;
    let lambda = 0.95;
    for (let i = rewards.length - 1; i >= 0; i--) {
        let delta = rewards[i] + gamma * (i < rewards.length - 1 ? values.arraySync()[i + 1] : 0) - values.arraySync()[i];
        gae = delta + gamma * lambda * gae;
        advantages.unshift(gae);
    }

    // Iterate epochs and train on batches
    let epochs = 10;
    let batch_size = 64;
    console.log("Training on " + states.length + " samples");
    for (let epoch = 0; epoch < epochs; epoch++) {
        for (let i = 0; i < states.length; i += batch_size) {
            // Subset detached data for batch
            let batch_states = states.slice(i, i + batch_size);
            let batch_rewards = rewards.slice(i, i + batch_size);
            let batch_actions = actions.slice(i, i + batch_size);
            let batch_log_probs = log_probs.slice(i, i + batch_size);
            let batch_advantages = advantages.slice(i, i + batch_size);
            let batch_values = values.arraySync().slice(i, i + batch_size);

            // Normalize advantages
            let adv_mean = tf.mean(tf.tensor1d(batch_advantages)).arraySync();
            let adv_std = tf.sqrt(tf.mean(tf.square(tf.tensor1d(batch_advantages).sub(adv_mean)))).arraySync();
            batch_advantages = batch_advantages.map(adv => (adv - adv_mean) / (adv_std + 1e-8));

            // Train actor model using continuous PPO loss, ensuring to not use arraySync()
            actor_optimizer.minimize(() => {
                let action_means = model.predict(tf.tensor2d(batch_states));
                let action_diffs = tf.sub(tf.tensor2d(batch_actions), action_means);
                // Compute log probabilities assuming Gaussian distribution
                let log_probs = tf.mul(-0.5 / (std * std), tf.sum(tf.square(action_diffs), 1));
                let ratios = tf.exp(tf.sub(log_probs, tf.tensor1d(batch_log_probs)));
                let surrogate1 = tf.mul(ratios, tf.tensor1d(batch_advantages));
                let surrogate2 = tf.mul(tf.clipByValue(ratios, 0.8, 1.2), tf.tensor1d(batch_advantages));
                let actor_loss = tf.neg(tf.mean(tf.minimum(surrogate1, surrogate2)));
                return actor_loss;
            });

            // Compute returns by adding batch_advantages to batch_values
            let batch_returns = tf.tensor1d(batch_advantages).add(tf.tensor1d(batch_values)).arraySync();

            // Compute return mean and std
            let batch_return_mean = tf.mean(tf.tensor1d(batch_returns)).arraySync();
            let batch_return_std = tf.sqrt(tf.mean(tf.square(tf.tensor1d(batch_returns).sub(batch_return_mean)))).arraySync();
            
            // Update running mean and std
            let beta = .05;
            return_mean = (1 - beta) * return_mean + beta * batch_return_mean;
            return_std = (1 - beta) * return_std + beta * batch_return_std;

            // Finish normalization of returns
            batch_returns = batch_returns.map(r => (r - return_mean) / (return_std + 1e-8));

            // Train critic model using MSE
            critic_optimizer.minimize(() => {
                let values = critic_model.predict(tf.tensor2d(batch_states)).reshape([-1]);
                let critic_loss = tf.losses.meanSquaredError(tf.tensor1d(batch_returns), values);
                return critic_loss;
            });
        }
    }

    // Increment model iteration
    model_iteration++;

    // Clear memory after training
    memory = [];
}

export function getModelVersion() {
    return model_iteration;
}

// Create model
var model_iteration = 0;
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [6], units: 16, activation: "relu" }));
model.add(tf.layers.layerNormalization());
model.add(tf.layers.dense({ units: 16, activation: "relu" }));
model.add(tf.layers.layerNormalization());
model.add(tf.layers.dense({ units: 2, activation: "tanh" }));

// Create critic model
const critic_model = tf.sequential();
critic_model.add(tf.layers.dense({ inputShape: [6], units: 16, activation: "relu" }));
model.add(tf.layers.layerNormalization());
critic_model.add(tf.layers.dense({ units: 16, activation: "relu" }));
model.add(tf.layers.layerNormalization());
critic_model.add(tf.layers.dense({ units: 1 }));

// Store running mean and std for reward normalization in critic
var return_mean = 0;
var return_std = 1;

// Create optimizers
const actor_optimizer = tf.train.adam(0.0003);
const critic_optimizer = tf.train.adam(0.0003);

// Initialize memory buffer for training
let memory = [];
