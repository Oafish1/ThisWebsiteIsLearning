import { getForce, trainModel, getModelVersion, modifyReward } from "./agent.js";

// Find canvas and context
var canvas = document.getElementById("main-window");
var ctx = canvas.getContext("2d");
var speed_scale_slider = document.getElementById("simulation-speed-slider");

// Handle resizing
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();  // Trigger once to use whole window
window.addEventListener("resize", resizeCanvas);

// Get mouse position while mouse button is held down
var pointer = {
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
var circle = {
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
    // Pointer
    pointer_r: 5,
    pointer_color: "#000600"
};

var env = {
    speed_scale: 20,
    position_scale: 2000,
    velocity_scale: 750,
    wall_force_scale: 2,
    min_delta: .1,
    max_delta: .4,
    training_interval: 200
}

// Set speed scale
function setSpeedScale(new_scale) {
    env.speed_scale = new_scale;
}

var prev_draw_time = 0;
var prev_update_time = 0;
var reward = 0;

// Main drawing function
function draw(new_time) {
    // Compute delta time
    let draw_delta_time = new_time - prev_draw_time;
    let update_delta_time = new_time - prev_update_time;
    // Scale by speed scale factor
    draw_delta_time = draw_delta_time * env.speed_scale;
    update_delta_time = update_delta_time * env.speed_scale;
    // Convert to seconds
    draw_delta_time = draw_delta_time / 1000;
    update_delta_time = update_delta_time / 1000;
    // Clamp delta time to avoid large jumps
    // Change accent color of slider to red if too fast, otherwise green
        speed_scale_slider.style.accentColor = "#00ff08";
    if (draw_delta_time > env.max_delta) {
        console.warn("Draw delta time of " + draw_delta_time.toFixed(3) + " s exceeds max delta, clamping to " + env.max_delta + " s");
        draw_delta_time = env.max_delta;
        speed_scale_slider.style.accentColor = "#ff0000";
    }
    if (update_delta_time > env.max_delta) {
        console.warn("Update delta time of " + update_delta_time.toFixed(3) + " s exceeds max delta, clamping to " + env.max_delta + " s");
        update_delta_time = env.max_delta;
        speed_scale_slider.style.accentColor = "#ff0000";
    }
    // Update draw time
    prev_draw_time = new_time;

    // Get new velocity if above minimum delta time
    if (update_delta_time > env.min_delta) {
        // Compute force
        let force = getForce(circle.x / env.position_scale, circle.y / env.position_scale, circle.vx / env.velocity_scale, circle.vy / env.velocity_scale, pointer.x / env.position_scale, pointer.y / env.position_scale);
        force.x = force.x * env.velocity_scale;
        force.y = force.y * env.velocity_scale;
        circle.vx += force.x * update_delta_time;
        circle.vy += force.y * update_delta_time;
        reward = force.reward;

        // Update update time
        prev_update_time = new_time;
    }

    // Clamp velocity
    let velocity_mag = Math.sqrt(circle.vx * circle.vx + circle.vy * circle.vy);
    if (velocity_mag > 1 * env.velocity_scale) {
        circle.vx = (circle.vx / velocity_mag) * env.velocity_scale;
        circle.vy = (circle.vy / velocity_mag) * env.velocity_scale;
    }

    // Update position
    circle.x += circle.vx * draw_delta_time;
    circle.y += circle.vy * draw_delta_time;

    // Bounce off walls
    // if (circle.x + circle.r > canvas.width) {
    //     circle.vx -= env.wall_force_scale * env.velocity_scale * draw_delta_time;
    //     modifyReward(-1);
    // } else if (circle.x - circle.r < 0) {
    //     circle.vx += env.wall_force_scale * env.velocity_scale * draw_delta_time;
    //     modifyReward(-1);
    // }
    // if (circle.y + circle.r > canvas.height) {
    //     circle.vy -= env.wall_force_scale * env.velocity_scale * draw_delta_time;
    //     modifyReward(-1);
    // } else if (circle.y - circle.r < 0) {
    //     circle.vy += env.wall_force_scale * env.velocity_scale * draw_delta_time;
    //     modifyReward(-1);
    // }

    // Kill velocity on walls
    if (circle.x + circle.r > canvas.width) {
        circle.vx = 0;
        circle.x = canvas.width - circle.r;
        modifyReward(-1);
    } else if (circle.x - circle.r < 0) {
        circle.vx = 0;
        circle.x = circle.r;
        modifyReward(-1);
    }
    if (circle.y + circle.r > canvas.height) {
        circle.vy = 0;
        circle.y = canvas.height - circle.r;
        modifyReward(-1);
    } else if (circle.y - circle.r < 0) {
        circle.vy = 0;
        circle.y = circle.r;
        modifyReward(-1);
    }

    // Teleport if out of bounds
    // if ((circle.x + circle.r < 0) || (circle.x - circle.r > canvas.width) || (circle.y + circle.r < 0) || (circle.y - circle.r > canvas.height)) {
    //     circle.x = canvas.width / 2;
    //     circle.y = canvas.height / 2;
    //     modifyReward(-10);
    // }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw pointer
    ctx.beginPath();
    ctx.arc(pointer.x, pointer.y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = circle.pointer_color;
    ctx.fill();
    ctx.closePath();

    // Basic drawing
    ctx.beginPath();
    ctx.arc(circle.x, circle.y, circle.r, 0, 2 * Math.PI);
    ctx.fillStyle = circle.training ? circle.training_color : circle.inference_color;
    ctx.fill();
    ctx.closePath();

    // Display reward and version as two-line text within circle
    ctx.fillStyle = circle.text_color;
    ctx.font = "8px Arial";
    ctx.textAlign = "center";
    ctx.fillText("RWD " + reward.toFixed(2), circle.x, circle.y + 10);
    ctx.fillText("V" + getModelVersion(), circle.x, circle.y - 6);

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

// Train model every 10 seconds, turning the circle red while training
function trainLoop() {
    circle.training = true;
    // Wait 100ms to give the renderer a chance to update the circle color before starting training
    setTimeout(() => {
        // Begin training
        let begin_time = performance.now();
        trainModel();
        circle.training = false;
        console.log("Model trained (V" + getModelVersion() + "), training took " + (performance.now() - begin_time) / 1000 + " s");
        setTimeout(trainLoop, 1000 * env.training_interval / env.speed_scale);  // Schedule next training loop, don't use setInterval to avoid cutting into inference time if training takes too long
    }, 100);
}
setTimeout(trainLoop, 1000 * env.training_interval / env.speed_scale);
