// Set backend
tf.setBackend("webgpu").then(() => {
    console.log("Backend set to WebGPU");
}).catch((error) => {
    console.error("Error setting WebGPU backend:", error);
});

// Import
import { getForce, trainModel, getModelVersion, modifyReward, getPreviousReward } from "./agent.js";

// Wait for backend
await tf.ready();

// Main vars
const do_warn = false;

// Find canvas and context
const canvas = document.getElementById("main-window");
const ctx = canvas.getContext("2d");
const speed_scale_slider = document.getElementById("simulation-speed-slider");

// Handle resizing
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();  // Trigger once to use whole window
window.addEventListener("resize", resizeCanvas);

// Get mouse position while mouse button is held down
const pointer = {
    x: canvas.width / 2,
    y: canvas.height / 2
}
function updatePointerPosition(event) {
    pointer.x = event.clientX;
    if (pointer.x < 0) {
        pointer.x = 0;
    } else if (pointer.x > canvas.width) {
        pointer.x = canvas.width;
    }
    pointer.y = event.clientY;
    if (pointer.y < 0) {
        pointer.y = 0;
    } else if (pointer.y > canvas.height) {
        pointer.y = canvas.height;
    }
}
window.addEventListener("mousedown", function(event) {
    updatePointerPosition(event);
    window.addEventListener("mousemove", updatePointerPosition);
});
window.addEventListener("mouseup", function(event) {
    window.removeEventListener("mousemove", updatePointerPosition);
});

// Configuration
const circle = {
    // Main
    x: canvas.width / 2,
    y: canvas.height / 2,
    vx: 0,
    vy: 0,
    // Display
    r: 25,
    inference_color: "#2b95db",
    training_color: "#ff0000",
    text_color: "#ffffff",
    // Flags
    training: false,
    // iteration_flag: false,
    // Pointer
    pointer_r: 5,
    pointer_color: "#000600"
};

const env = {
    speed_scale: 20,
    velocity_limit: .2,
    wall_force_scale: 2,
    draw_max_delta: .5,
    update_min_delta: .1,
    update_max_delta: .1  // Normally .2
}

// Set speed scale
function setSpeedScale(new_scale) {
    env.speed_scale = new_scale;
}

var prev_draw_time = 0;
var update_delta_time = 0;

// Main drawing function
function draw(new_time) {
    // Compute delta time
    let draw_delta_time = new_time - prev_draw_time;
    draw_delta_time = draw_delta_time * env.speed_scale;  // Scale by speed scale factor
    draw_delta_time = draw_delta_time / 1000;  // Convert to seconds
    
    // Clamp draw delta time to avoid large jumps
    // Change accent color of slider to red if too fast, otherwise green
    speed_scale_slider.style.accentColor = "#00ff08";
    if (draw_delta_time > env.draw_max_delta) {
        if (do_warn) {
            console.warn("Draw delta time of " + draw_delta_time.toFixed(3) + " s exceeds max delta, clamping to " + env.draw_max_delta + " s");
        }
        draw_delta_time = env.draw_max_delta;
        speed_scale_slider.style.accentColor = "#ff0000";
    }
    prev_draw_time = new_time;

    // Update update delta time according to draw delta time
    update_delta_time += draw_delta_time;
    if (update_delta_time > env.update_max_delta) {
        if (do_warn) {
            console.warn("Update delta time of " + update_delta_time.toFixed(3) + " s exceeds max delta, clamping update and draw times to " + env.update_max_delta + " s");
        }
        draw_delta_time += env.update_max_delta - update_delta_time;  // Reduce draw delta time accordingly
        update_delta_time = env.update_max_delta;
        speed_scale_slider.style.accentColor = "#ff0000";
    }

    // Get new velocity if above minimum delta time
    if (update_delta_time >= env.update_min_delta) {
        // Compute force
        let force = getForce(
            circle.x / canvas.width,
            circle.y / canvas.height,
            circle.vx / canvas.width,
            circle.vy / canvas.height,
            pointer.x / canvas.width,
            pointer.y / canvas.height,
            update_delta_time);
        force.x = force.x * env.velocity_limit * canvas.width;
        force.y = force.y * env.velocity_limit * canvas.height;
        circle.vx += force.x * update_delta_time;
        circle.vy += force.y * update_delta_time;

        // Reset delta time
        update_delta_time = 0;
    }

    // Clamp velocity properly
    // let velocity_mag = Math.sqrt(circle.vx * circle.vx + circle.vy * circle.vy);
    // if (velocity_mag > env.velocity_limit) {
    //     circle.vx = (circle.vx / velocity_mag) * env.velocity_limit;
    //     circle.vy = (circle.vy / velocity_mag) * env.velocity_limit;
    // }

    // Clamp velocity on each axis
    // This makes the physics more understandable to the agent
    if (circle.vx > env.velocity_limit * canvas.width) {
        circle.vx = env.velocity_limit * canvas.width;
    } else if (circle.vx < -env.velocity_limit * canvas.width) {
        circle.vx = -env.velocity_limit * canvas.width;
    }
    if (circle.vy > env.velocity_limit * canvas.height) {
        circle.vy = env.velocity_limit * canvas.height;
    } else if (circle.vy < -env.velocity_limit * canvas.height) {
        circle.vy = -env.velocity_limit * canvas.height;
    }

    // Update position
    circle.x += circle.vx * draw_delta_time;
    circle.y += circle.vy * draw_delta_time;

    // Force back into frame
    // if (circle.x + circle.r > canvas.width) {
    //     circle.vx -= env.wall_force_scale * env.velocity_limit * canvas.width * draw_delta_time;
    //     // modifyReward(-1);
    // } else if (circle.x - circle.r < 0) {
    //     circle.vx += env.wall_force_scale * env.velocity_limit * canvas.width * draw_delta_time;
    //     // modifyReward(-1);
    // }
    // if (circle.y + circle.r > canvas.height) {
    //     circle.vy -= env.wall_force_scale * env.velocity_limit * canvas.height * draw_delta_time;
    //     // modifyReward(-1);
    // } else if (circle.y - circle.r < 0) {
    //     circle.vy += env.wall_force_scale * env.velocity_limit * canvas.height * draw_delta_time;
    //     // modifyReward(-1);
    // }

    // Kill velocity on walls
    // let wallhit_penalty = -1;
    // if (circle.x + circle.r > canvas.width) {
    //     circle.vx = 0;
    //     circle.x = canvas.width - circle.r;
    //     // modifyReward(wallhit_penalty);
    // } else if (circle.x - circle.r < 0) {
    //     circle.vx = 0;
    //     circle.x = circle.r;
    //     // modifyReward(wallhit_penalty);
    // }
    // if (circle.y + circle.r > canvas.height) {
    //     circle.vy = 0;
    //     circle.y = canvas.height - circle.r;
    //     // modifyReward(wallhit_penalty);
    // } else if (circle.y - circle.r < 0) {
    //     circle.vy = 0;
    //     circle.y = circle.r;
    //     // modifyReward(wallhit_penalty);
    // }

    // Reset and terminate if out of bounds
    if ((circle.x + circle.r < 0) || (circle.x - circle.r > canvas.width) || (circle.y + circle.r < 0) || (circle.y - circle.r > canvas.height)) {
        circle.x = canvas.width / 2;
        circle.y = canvas.height / 2;
        circle.vx = 0;
        circle.vy = 0;
        modifyReward(-100, true);  // Terminate
        // modifyReward(0, true);  // Terminate with no penalty
        // modifyReward(0, false, true);  // Truncate
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw circle
    ctx.beginPath();
    ctx.arc(circle.x, circle.y, circle.r, 0, 2 * Math.PI);
    ctx.fillStyle = circle.training ? circle.training_color : circle.inference_color;
    ctx.fill();
    ctx.closePath();

    // Display reward and version as two-line text within circle
    // One tick delayed, but it's fine
    ctx.fillStyle = circle.text_color;
    ctx.font = "8px Arial";
    ctx.textAlign = "center";
    ctx.fillText("RWD " + getPreviousReward().toFixed(2), circle.x, circle.y + 10);
    ctx.fillText("V" + getModelVersion(), circle.x, circle.y - 6);

    // Draw pointer
    ctx.beginPath();
    ctx.arc(pointer.x, pointer.y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = circle.pointer_color;
    ctx.fill();
    ctx.closePath();

    window.requestAnimationFrame(draw);
}

// Loop animation
window.requestAnimationFrame(draw);

// Add event listener for speed scale slider and update speed-scale-text when it changes
function updateSpeedScaleSlider() {
    setSpeedScale(parseFloat(speed_scale_slider.value));
    var speed_scale_text = document.getElementById("simulation-speed-text");
    speed_scale_text.textContent = speed_scale_slider.value + "x Speed";
}
speed_scale_slider.addEventListener("input", updateSpeedScaleSlider);
updateSpeedScaleSlider();  // Trigger once to initialize text

// Check if we can train model every second, turning the circle red while training
async function trainLoop() {
    circle.training = true;
    // Wait 100ms to give the renderer a chance to update the circle color before starting training
    setTimeout(() => {
        // Begin training
        let begin_time = performance.now();
        trainModel().then((trained) => {
            if ( trained ) {
                console.log("Model trained (V" + getModelVersion() + "), training took " + (performance.now() - begin_time) / 1000 + " s");
                console.log();
            }
        });
        circle.training = false;
        setTimeout(trainLoop, 1000);  // Schedule next training loop, don't use setInterval to avoid cutting into inference time if training takes too long
    }, 100);
}
setTimeout(trainLoop, 1000);

// TODO
// Add touch support for mobile
// Add checkbox to toggle if we should train
// Make slider immune to changing pointer when dragging
// Plot graph of mean reward
// Add eyes to circle based on where facing
// Make parameters user-adjustable
