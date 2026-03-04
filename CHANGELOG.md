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