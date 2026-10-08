





# What is the difference between the different version of self play

```
PURE_SELF_PLAY:  Train against current policy (both players identical).
FICTITIOUS_PLAY: Sample historical checkpoints uniformly as opponents.
DOUBLE_ORACLE:   Sample checkpoints based on Nash equilibrium distribution.
```



# Todo

- [ ] Does a frozen agent make the environment station or non stationary
- [x] What is piecewise stationary
- [ ] "explainthis to mde "`env.py`: for `PURE_SELF_PLAY` the two-seat PettingZoo env is flattened with SuperSuit so **one PPO model steps both seats** (that's why `train.py` halves `n_steps` to `3072 // (2*num_envs)` — every battle yields two trajectories)."
