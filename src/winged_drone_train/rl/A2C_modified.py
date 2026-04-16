import torch
import torch.nn.functional as F
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import Memory
from torch.distributions import Normal
import math

class ActorCriticTanh(ActorCriticRecurrent):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=128,
        critic_rnn_hidden_size=None,
        rnn_num_layers=1,
        init_noise_std=1.0,
        max_servo=0.34906585,
        max_throttle=1.0,
        **kw,
    ):
        actor_rnn_hidden  = rnn_hidden_size
        critic_rnn_hidden = critic_rnn_hidden_size if critic_rnn_hidden_size is not None else rnn_hidden_size

        # Bypass ActorCriticRecurrent.__init__ so actor/critic can have
        # independent LSTM hidden sizes. Call ActorCritic directly with the
        # correct MLP input dims (= each RNN's hidden size).
        ActorCritic.__init__(
            self,
            num_actor_obs=actor_rnn_hidden,
            num_critic_obs=critic_rnn_hidden,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        self.memory_a = Memory(num_actor_obs,  type=rnn_type, num_layers=rnn_num_layers, hidden_size=actor_rnn_hidden)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=critic_rnn_hidden)


        self.max_servo    = max_servo
        self.max_throttle = max_throttle
        self._LOG2        = math.log(2.)
        self.recurrency   = True
    # ------------------------------------------------ helper
    def _scale(self, a):
        thr = 0.5 * (a[..., :1] + 1) * self.max_throttle
        srv = a[..., 1:] * self.max_servo
        return torch.cat([thr, srv], -1)

    def _inverse_scale(self, act):
        thr, srv = act[..., :1], act[..., 1:]
        max_thr = self.max_throttle if self.max_throttle > 1e-6 else 1e-6
        max_srv = self.max_servo if self.max_servo > 1e-6 else 1e-6
        a_thr = thr / max_thr * 2 - 1
        a_srv = srv / max_srv
        return torch.cat([a_thr, a_srv], -1).clamp(-0.999999, 0.999999)

    # ------------------------------------------------ overrides
    def act(self, obs, deterministic=False, masks=None, hidden_states=None):
        if self.recurrency:
            # === Comportamento originale (ricorrente) ===
            inp = self.memory_a(obs, masks, hidden_states)
        else:
            # === Feed-forward puro (no RNN) ===
            inp = obs          # MLP dell’attore

        self.update_distribution(inp.squeeze(0))
        z = self.distribution.mean if deterministic else self.distribution.rsample()
        a = torch.tanh(z)                         # (-1,1)
        act = self._scale(a)

        # log-prob stabile
        logp_corr = 2 * (self._LOG2 - z - F.softplus(-2*z))
        self._last_logp = (self.distribution.log_prob(z) + logp_corr).sum(-1, keepdim=True)

        return act

    def act_inference(self, observations):
        if self.recurrency:
            inp = self.memory_a(observations)
        else:
            inp = observations
        actions_mean = self.actor(inp.squeeze(0) if inp.dim() == 3 else inp)
        a = torch.tanh(actions_mean)
        return self._scale(a)

    def get_actions_log_prob(self, act):
        a  = self._inverse_scale(act)
        # atanh in forma stabile
        z  = 0.5 * (torch.log1p(a) - torch.log1p(-a))
        logp_corr = 2 * (self._LOG2 - z - F.softplus(-2*z))
        return (self.distribution.log_prob(z) + logp_corr).sum(-1)
