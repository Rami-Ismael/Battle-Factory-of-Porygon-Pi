"""Actor-only inference and exact-input logit caching; sample on every request."""
from collections import OrderedDict
import torch
from stable_baselines3.common.preprocessing import preprocess_obs


def actor_logits(policy, observations):
    processed = preprocess_obs(observations,policy.observation_space,normalize_images=policy.normalize_images)
    features = policy.pi_features_extractor(processed)
    latent = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent)


def sample_actions(policy,logits,mask,deterministic=False):
    # Match the reference's two sampling calls, including its initially discarded
    # second action. Removing those draws would change seeded policy trajectories.
    dist = policy.get_dist_from_logits(logits,mask)
    actions = dist.get_actions(deterministic=deterministic)
    dist2 = policy.get_dist_from_logits(logits,mask,actions[:,:1])
    actions2 = dist2.get_actions(deterministic=deterministic)
    actions[:,1] = actions2[:,1]
    return actions.reshape((-1,*policy.action_space.shape))


class LogitCache:
    def __init__(self,limit=4096):
        self.entries = OrderedDict()
        self.limit = limit
        self.hits = 0
        self.misses = 0

    def compute(self,policy,observations,obs_array,mask_array):
        # The cache belongs to one immutable loaded policy in one worker. Include
        # the full input bytes (including masks), never a partial board signature.
        keys = [(row.tobytes(),mask.tobytes()) for row,mask in zip(obs_array,mask_array)]
        outputs = [None]*len(keys)
        missing = []
        for i,key in enumerate(keys):
            if key in self.entries:
                self.hits += 1
                self.entries.move_to_end(key)
                outputs[i] = self.entries[key]
            else:
                self.misses += 1
                missing.append(i)
        if missing:
            subset = {k:v[missing] for k,v in observations.items()}
            logits = actor_logits(policy,subset)
            for i,row in zip(missing,logits):
                outputs[i] = row
                self.entries[keys[i]] = row.detach().clone()
                self.entries.move_to_end(keys[i])
                while len(self.entries)>self.limit:
                    self.entries.popitem(last=False)
        return torch.stack(outputs)
