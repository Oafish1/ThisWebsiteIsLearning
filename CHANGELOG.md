### 2026.4.1.1
- Adjust `README` to be more clear and add more details/future work
- Readjust entropy and environment speed to promote effective training

### 2026.4.1
- Clean up `README`
- Try action holding
- Revise training parameters for visualization and training efficiency

### 2026.3.4
- Add random start positioning
- Add random velocity limit
- Clear WebGL cache regularly
- Fix action sampling
- Fix log likelihood calculation
- Fix remaining bugs, allowing the model to learn
- Massively increase default speed limit
- Remove asynchronous functions from training
- Use `tf.tidy()` when training to prevent memory leaks

### 2026.3.3
- Add action penalty
- Add delta time reward scaling
- Add `README.md`
- Add separate draw and update time constraints and independent handling
- Change backend to use GPU if available
- Clean UI z-ordering
- Complicate model structure
- Fix bug with normalizing advantages before return calculation
- Fix optimizer bug causing non-updates
- Fix slicing bug causing over-indexing
- Make reward components fully isolated for step-to-step change measurement
- Miscellaneous QOL changes
- More verbose console messages
- Normalize model inputs, rather than having the model figure them out
- Optional warning mute
- Stop using incomplete memories
- Tune hyperparameters, including std
- Update shown reward to be adjusted by penalties
- Use tf-native APIs where possible

### 2026.3.2.1
- Add memory shuffling at the beginning of each epoch
- Complicate the model
- Fix slicing bug with batch sampling causing only first memories to be used for training
- Readd velocity penalty

### 2026.3.2
- Initial styling, controls, and PPO implementation with GAE and bootstrapping