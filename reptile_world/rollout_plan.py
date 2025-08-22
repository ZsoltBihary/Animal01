# This is the plan for stateful rollout + buffering + training

# === In the buffer we need:
#   --- buffer_row = (prev_map, prev_act, obs, act, targ_q, targ_r, targ_obs)
# === The forward method, used in training has the signature:
#   --- (prev_map, prev_act, obs, act) -> (pred_q, pred_r, pred_obs)
# === The loss function has the structure:
#   --- loss = loss_q(pred_q, targ_q) + lam_r * loss_r(pred_r, targ_r) + lam_obs * loss_obs(pred_obs, targ_obs)

# === The model method, used in rollout / simulation has the signature:
#   --- (prev_map, prev_act, obs) -> (map, q_a)

# === The model method that is most general, has the signature:
#   --- (prev_map, prev_act, obs) -> (map, q_a, r_a, obs_a)
# *** But it may not be the most efficient implementation to base everything on this

# ===== ROLLOUT PLAN =====

# We have to align the buffer data row from quantities that are calculated
# at different time steps of the rollout cycle.
# Also, we want to be able to pause and then restart data collection at where we left off,
#     as if we had one long rollout simulation.
# For these reasons, rollout should be structured as a class that can hold the necessary
#     variables as attributes.

# for step in range(num_steps)
#     act = select_action(self.q_a)
#     next_rew, next_obs = resolve(act)
#     next_q_a, next_map = online_model(self.map, act, next_obs)
#
#     *** THIS IS WHERE THE TARGET Q IS CALCULATED ...
#     next_q_target_a = target_model(self.map, act, next_obs)
#     next_best_act = argmax(next_q_a)
#     next_best_q = next_q_target_a[next_best_act]
#     targ_q = (1.0-self.gamma) * next_rew + self.gamma * next_best_q
#
#     *** THIS IS WHERE DATA IS SAVED TO BUFFER ...
#     buffer.save(prev_map = self.prev_map, prev_act = self.prev_act, obs = self.obs, act = act,
#                 targ_q = targ_q, targ_r = next_rew, targ_obs = next_obs)
#
#     self.obs = next_obs
#     self.prev_act = act
#     self.q_a = next_q_a
#     self.prev_map = map
#     self.map = next_map
#
@torch.no_grad()
def rollout(self, num_steps: int):
    # startup must set: self.map, self.prev_map, self.prev_act, self.obs, maybe self.q_a
    # and ensure world.last_action matches prev_act if needed

    for t in range(num_steps):
        # 1) select action from current policy (use q_a computed from previous step or recompute)
        #    ensure q_a is available for the current map/obs; otherwise compute using online model:
        # q_a = self.online_model.predict_q(self.map, self.prev_act, self.obs)  # optional
        action = self.animal.select_action(self.q_a)  # shape (B,)

        # 2) step the environment -> get immediate reward and next observation
        next_rew, next_obs = self.world.resolve(action)  # returns batched tensors (B,) and (B, C, K, K) etc.

        # 3) compute online next q and next_map (no in-place modification)
        next_map, next_q_a = self.online_model(self.map, action, next_obs)   # returns next_map, next_q_a

        # 4) compute target q-values for same inputs (target model must not mutate any state)
        #    If target_model returns map too, ignore returned map (it shouldn't mutate global state)
        next_q_target_a = self.target_model(self.map, action, next_obs)  # shape (B, A)
        # 5) double-dqn selection/eval:
        best_action = next_q_a.argmax(dim=1)
        best_q = next_q_target_a.gather(1, best_action.unsqueeze(1)).squeeze(1)  # (B,)

        # 6) compute TD target (choose canonical or your normalized version)
        targ_q = next_rew + self.gamma * best_q            # canonical
        # or: targ_q = (1.0 - self.gamma) * next_rew + self.gamma * best_q   # if intentional

        # 7) save a full row to buffer (all tensors detached and moved to CPU inside buffer)
        #    row = (prev_map, prev_act, obs, act, targ_q, targ_r=next_rew, targ_obs=next_obs)
        self.buffer.add_row(
            prev_map=self.prev_map,         # must be batched (B, L, S, K, K)
            prev_act=self.prev_act,         # (B,)
            obs=self.obs,                   # (B, E, K, K) or original obs
            act=action,                     # (B,)
            targ_q=targ_q,                  # (B,)
            targ_r=next_rew,                # (B,)
            targ_obs=next_obs               # (B, C, K, K) or your obs format
        )

        # 8) advance local variables for next iteration
        self.prev_map = self.map       # assign current map to prev_map
        self.map = next_map            # update map to next_map (no in-place)
        self.prev_act = action
        self.obs = next_obs
        self.q_a = next_q_a
