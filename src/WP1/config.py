"""
Unified RunConfig dataclass — single source of truth for a training run.
========================================================================

Motivation
----------
Prior to WP1 every tuneable knob lived inside two functions —
``get_train_cfg()`` and ``get_cfgs()`` in ``winged_drone_train.train`` —
as deeply-nested, hardcoded Python dicts.  This made it impossible to:

* compare two runs by diffing their configs,
* reproduce a run from its artefacts alone,
* sweep hyperparameters without editing source code.

``RunConfig`` replaces both functions with a single, flat-ish dataclass
hierarchy that can be:

1. **serialised to / deserialised from YAML** (``to_yaml`` / ``from_yaml``),
2. **overridden from the CLI** with ``--cfg.section.key value`` syntax,
3. **converted back** to the legacy dict format via ``to_legacy_cfgs()``
   so that existing ``WingedDroneEnv``, ``Gen_Env``, and ``OnPolicyRunner``
   code does not need any modification.

Hierarchy
---------
::

    RunConfig
    ├── exp_name          str             experiment tag (log folder name)
    ├── ppo               PPOConfig       clip, KL, entropy, gamma, lr, ...
    ├── policy            PolicyConfig    hidden dims, LSTM, activation, action limits
    ├── training          TrainingConfig  num_envs, max_iters, save/eval intervals
    ├── env               EnvConfig       termination thresholds, forest, aero noise
    ├── obs               ObsConfig       num_obs, noise stds, genome flag
    ├── reward            RewardConfig    per-component scale factors
    ├── command           CommandConfig   command dimensionality
    └── catalog           CatalogConfig   URDF catalog size, dir, seed

Usage
-----
.. code-block:: python

    from WP1.config import RunConfig

    # Defaults (matches current hardcoded values)
    cfg = RunConfig()

    # From YAML
    cfg = RunConfig.from_yaml("configs/foundation.yaml")

    # CLI overrides
    cfg.apply_cli_overrides(["--cfg.ppo.learning_rate", "3e-4"])

    # Convert to legacy dicts for existing code
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = cfg.to_legacy_cfgs()

    # Snapshot for reproducibility
    cfg.to_yaml("logs/runs/.../config.yaml")
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ============================================================================
#  Sub-configs
# ============================================================================

@dataclass
class PPOConfig:
    """Proximal Policy Optimisation hyperparameters.

    These map directly to the ``train_cfg["algorithm"]`` dict consumed by
    RSL-RL's ``PPO`` class.

    Attributes
    ----------
    clip_param : float
        Probability-ratio clipping threshold (epsilon).  Typical range
        0.1 – 0.3.  Default 0.15.
    desired_kl : float
        Target approximate KL divergence for the adaptive learning-rate
        schedule.  The LR is halved / doubled to track this value.
    entropy_coef : float
        Coefficient for the entropy bonus added to the surrogate loss.
        Encourages exploration; set to 0 to disable.
    gamma : float
        Discount factor for future rewards.  0.99 corresponds to a
        ~100-step effective horizon.
    lam : float
        GAE-lambda for advantage estimation.  Lower values reduce
        variance at the cost of bias.
    learning_rate : float
        Initial learning rate for Adam.  Adjusted at runtime when
        ``schedule="adaptive"``.
    max_grad_norm : float
        Maximum L2 norm for gradient clipping.  Prevents catastrophic
        updates.
    num_learning_epochs : int
        Number of passes over the rollout buffer per PPO update.
    num_mini_batches : int
        Number of mini-batches the rollout buffer is split into.
    schedule : str
        Learning-rate schedule.  ``"adaptive"`` adjusts LR to track
        ``desired_kl``; ``"fixed"`` keeps it constant.
    use_clipped_value_loss : bool
        Whether to clip the value-function loss (PPO2-style).
    value_loss_coef : float
        Weight of the value-function loss relative to the policy loss.
    normalize_advantage_per_mini_batch : bool
        Re-normalise advantages within each mini-batch (zero-mean, unit-std).
    """

    clip_param: float = 0.15
    desired_kl: float = 0.006
    entropy_coef: float = 0.002
    gamma: float = 0.99
    lam: float = 0.9
    learning_rate: float = 1e-4
    max_grad_norm: float = 0.5
    num_learning_epochs: int = 2
    num_mini_batches: int = 32
    schedule: str = "adaptive"
    use_clipped_value_loss: bool = True
    value_loss_coef: float = 0.3
    normalize_advantage_per_mini_batch: bool = True


@dataclass
class PolicyConfig:
    """Actor-critic network architecture.

    Defines the MLP hidden-layer sizes, recurrent backbone, and action-space
    scaling for the ``ActorCriticTanh`` policy used by RSL-RL.

    Attributes
    ----------
    actor_hidden_dims : List[int]
        Sizes of the actor MLP hidden layers (applied *before* the LSTM).
    critic_hidden_dims : List[int]
        Sizes of the critic MLP hidden layers.
    activation : str
        Activation function name (``"elu"``, ``"relu"``, ``"tanh"``).
    init_noise_std : float
        Initial standard deviation for the diagonal Gaussian action
        distribution.
    rnn_type : str
        Recurrent cell type: ``"lstm"`` or ``"gru"``.
    rnn_hidden_size : int
        Number of units in the recurrent cell.
    rnn_num_layers : int
        Number of stacked recurrent layers.
    max_servo : float
        Maximum servo deflection (radians) after tanh scaling.
        1.0 rad ~= 57 deg.
    max_throttle : float
        Maximum throttle command after tanh scaling (dimensionless, [0, 1]).
    """

    actor_hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "elu"
    init_noise_std: float = 0.3
    rnn_type: str = "lstm"
    rnn_hidden_size: int = 64
    rnn_num_layers: int = 1
    max_servo: float = 1.0
    max_throttle: float = 1.0


@dataclass
class TrainingConfig:
    """Training loop / runner settings.

    Attributes
    ----------
    num_envs : int
        Number of parallel Genesis environments.  Must be >= number of
        URDFs in the catalog for mixture mode.
    max_iterations : int
        Total number of PPO updates (outer loop).
    num_steps_per_env : int
        Rollout length *T* — steps collected per environment per iteration
        before each PPO update.  Total batch size = num_envs * T.
    save_interval : int
        Save a model checkpoint every *N* iterations.
    eval_interval : int
        Run held-out evaluation every *N* iterations (WP3).
    seed : int
        Global random seed for reproducibility (numpy, torch, genesis).
    empirical_normalization : bool
        Whether RSL-RL applies running-mean observation normalisation.
    device : str
        Torch device string (``"cuda:0"``, ``"cpu"``).
    """

    num_envs: int = 16384
    max_iterations: int = 850
    num_steps_per_env: int = 25
    save_interval: int = 50
    eval_interval: int = 200
    seed: int = 42
    empirical_normalization: bool = True
    device: str = "cuda:0"


@dataclass
class EnvConfig:
    """WingedDroneEnv parameters — physics, termination, forest geometry.

    Maps one-to-one to the ``env_cfg`` dict consumed by ``WingedDroneEnv``.

    Attributes
    ----------
    num_actions : int
        Action-space dimension: 1 throttle + 4 servos = 5.
    dt : float
        Nominal time-step (not used directly; kept for reference).
    drone : str
        Drone model identifier (``"morphing_drone"``).
    naca : str
        NACA aerofoil profile code for the wings (e.g. ``"3416"``).
    termination_if_close_to_ground : float
        Ground-proximity termination threshold in metres.
    termination_if_y_greater_than : float
        Lateral-wall termination threshold (absolute Y, metres).
    termination_if_z_greater_than : float
        Altitude ceiling termination threshold (metres).
    base_init_pos : List[float]
        Initial drone position ``[x, y, z]`` in metres.
    base_init_quat : List[float]
        Initial drone orientation quaternion ``[w, x, y, z]``.
    episode_length_s : float
        Maximum episode duration in seconds.
    simulate_action_latency : bool
        Whether to simulate one-step action latency via a FIFO buffer.
    action_latency_min_steps, action_latency_max_steps : int
        Min / max latency in physics steps (randomised per env on reset).
    clip_actions : float
        Hard clamp on normalised actions before scaling.
    tree_radius : float
        Radius of forest-cylinder obstacles (metres).
    tree_height : float
        Height of forest-cylinder obstacles (metres).
    y_lower, y_upper : float
        Lateral corridor bounds (metres).
    forest_x_limit, x_upper : float
        Forward distance for the success condition (metres).
    aero_noise : bool
        Enable aerodynamic force / parameter noise.
    aero_noise_sigma0 : float
        Base std for force-magnitude and force-direction noise.
    noise_sigma_param : float
        Std for aerodynamic parameter randomisation on reset.
    """

    num_actions: int = 5
    dt: float = 0.01
    drone: str = "morphing_drone"
    naca: str = "3416"
    termination_if_close_to_ground: float = 0.1
    termination_if_y_greater_than: float = 50.0
    termination_if_z_greater_than: float = 50.0
    base_init_pos: List[float] = field(default_factory=lambda: [-30.0, 0.0, 15.0])
    base_init_quat: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    episode_length_s: float = 100.0
    at_target_threshold: float = 0.1
    resampling_time_s: float = 3.0
    simulate_action_latency: bool = True
    action_latency_min_steps: int = 0
    action_latency_max_steps: int = 1
    action_latency_random_per_step: bool = False
    clip_actions: float = 1.0
    visualize_target: bool = False
    visualize_camera: bool = True
    max_visualize_FPS: int = 15
    tree_radius: float = 0.75
    tree_height: float = 100.0
    y_lower: float = -50.0
    y_upper: float = 50.0
    forest_x_limit: float = 150.0
    x_upper: float = 150.0
    aero_noise: bool = True
    aero_noise_sigma0: float = 0.05
    noise_sigma_param: float = 0.15


@dataclass
class ObsConfig:
    """Observation-space configuration.

    Controls which features are included in the actor / critic observation
    vectors and the per-feature Gaussian noise standard deviations applied
    during training.

    The observation layout (without genome) is::

        [z_norm(1), quat(4), lin_vel(3), depth(20), last_actions(5), v_cmd(1)]
        = 34 elements  (36 when depth has 22 sectors — see env)

    Attributes
    ----------
    num_obs : int
        Total observation dimension.  Overwritten by the environment after
        construction, but must be set here to match the expected default.
    add_noise : bool
        Whether to inject Gaussian noise into the actor observation.
    add_genome_obs_actor : bool
        Whether to append the normalised genome vector to actor observations.
        **Must be False** for the morphology-blind actor (WP2).
    add_genome_obs_critic : bool
        Whether to append the normalised genome vector to critic observations.
    noise_std_* : float
        Per-feature noise standard deviations.  ``0.0`` means no noise for
        that feature.
    """

    num_obs: int = 36
    add_noise: bool = True
    add_genome_obs_actor: bool = False
    add_genome_obs_critic: bool = True
    noise_std_z: float = 0.01
    noise_std_quat: float = 0.01
    noise_std_vel: float = 0.02
    noise_std_depth: float = 0.05
    noise_std_last_thr: float = 0.0
    noise_std_last_jnts: float = 0.0
    noise_std_v_tgt: float = 0.0
    noise_std_genome: float = 0.05


@dataclass
class RewardConfig:
    """Per-component reward scaling weights.

    Each attribute multiplies the corresponding reward term computed by
    ``WingedDroneEnv._reward_<name>()``.  Negative values penalise;
    positive values encourage.  Set to ``0.0`` to disable a component.

    Attributes
    ----------
    smooth : float
        Penalty for large action *changes* between consecutive steps
        (encourages smooth control).
    angular : float
        Penalty for large angular velocities.
    crash : float
        One-time penalty applied on any crash termination.
    obstacle : float
        Continuous penalty that grows exponentially as the drone
        approaches obstacles.
    energy : float
        Penalty proportional to instantaneous power consumption.
    progress : float
        Gaussian reward tracking the commanded forward speed.
    height : float
        Penalty for deviating from the reference altitude.
    success : float
        Bonus for reaching the end of the forest corridor.
    cosmetic : float
        Penalty for asymmetric servo usage (visual quality of flight).
    stability : float
        Penalty for oscillatory behaviour (currently unused, set to 0).
    """

    smooth: float = -1e-1
    angular: float = -5e-3
    crash: float = -10.0
    obstacle: float = -0.1
    energy: float = -2e-3
    progress: float = 5e-1
    height: float = -5e-3
    success: float = 0.0
    cosmetic: float = -1.0
    stability: float = 0.0


@dataclass
class CommandConfig:
    """High-level command configuration.

    Attributes
    ----------
    num_commands : int
        Dimensionality of the command vector.  Currently 1 (forward-speed
        target only).
    """

    num_commands: int = 1


@dataclass
class CatalogConfig:
    """URDF catalog settings for multi-morphology training (WP3).

    When ``n_urdf`` is set, a catalog of that many random morphologies is
    built before training begins (using ``Chromosome_Drone`` + ``UrdfMaker``).
    The catalog directory is then passed to ``Gen_Env`` which creates one
    ``WingedDroneEnv`` per URDF and distributes ``num_envs`` across them.

    Attributes
    ----------
    n_urdf : Optional[int]
        Number of URDFs to generate.  ``None`` or ``0`` disables catalog
        building and falls back to single-morphology training.
    catalog_dir : Optional[str]
        Directory where URDFs and ``catalog.txt`` are stored.  Defaults to
        ``urdf_generated`` when ``n_urdf`` is set.
    urdf_seed : int
        Random seed for reproducible catalog generation.
    """

    n_urdf: Optional[int] = None
    catalog_dir: Optional[str] = None
    urdf_seed: int = 0


# ============================================================================
#  Top-level RunConfig
# ============================================================================

@dataclass
class RunConfig:
    """Single top-level configuration object for a complete training run.

    ``RunConfig`` is the central dataclass that replaces the legacy
    ``get_train_cfg()`` + ``get_cfgs()`` functions.  It aggregates all
    sub-configs and provides three key capabilities:

    1. **YAML serialisation** — ``to_yaml()`` / ``from_yaml()`` for saving
       and loading complete configuration snapshots.
    2. **CLI overrides** — ``apply_cli_overrides()`` parses
       ``--cfg.section.key value`` arguments for quick experiments.
    3. **Legacy conversion** — ``to_legacy_cfgs()`` produces the exact
       ``(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)`` tuple
       expected by the existing ``WingedDroneEnv`` and ``OnPolicyRunner``.

    Attributes
    ----------
    exp_name : str
        Experiment name used for the run-folder suffix and TensorBoard tags.
    ppo : PPOConfig
        PPO algorithm hyperparameters.
    policy : PolicyConfig
        Actor-critic network architecture.
    training : TrainingConfig
        Runner and training-loop settings.
    env : EnvConfig
        WingedDroneEnv physics, termination, and forest parameters.
    obs : ObsConfig
        Observation-space flags and noise scales.
    reward : RewardConfig
        Per-component reward scaling weights.
    command : CommandConfig
        Command vector configuration.
    catalog : CatalogConfig
        URDF catalog settings for multi-morphology training.
    """

    exp_name: str = "drone-forest"

    ppo: PPOConfig = field(default_factory=PPOConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)

    # ------------------------------------------------------------------ #
    # Conversion helpers — produce the legacy dict format expected by
    # existing WingedDroneEnv, Gen_Env, and RSL-RL runner
    # ------------------------------------------------------------------ #

    def to_train_cfg(self) -> Dict[str, Any]:
        """Build the ``train_cfg`` dict consumed by RSL-RL's ``OnPolicyRunner``.

        The returned dict has the exact nested structure that RSL-RL expects::

            {
                "algorithm": { ... PPO knobs ... },
                "policy":    { ... network arch ... },
                "runner":    { ... logging, checkpointing ... },
                ...
            }

        Returns
        -------
        Dict[str, Any]
            Fully-populated training configuration dictionary.
        """
        return {
            "num_steps_per_env": self.training.num_steps_per_env,
            "save_interval": self.training.save_interval,
            "runner_class_name": "OnPolicyRunner",
            "empirical_normalization": self.training.empirical_normalization,
            "seed": self.training.seed,
            "logger": "tensorboard",
            "algorithm": {
                "normalize_advantage_per_mini_batch": self.ppo.normalize_advantage_per_mini_batch,
                "class_name": "PPO",
                "clip_param": self.ppo.clip_param,
                "desired_kl": self.ppo.desired_kl,
                "entropy_coef": self.ppo.entropy_coef,
                "gamma": self.ppo.gamma,
                "lam": self.ppo.lam,
                "learning_rate": self.ppo.learning_rate,
                "max_grad_norm": self.ppo.max_grad_norm,
                "num_learning_epochs": self.ppo.num_learning_epochs,
                "num_mini_batches": self.ppo.num_mini_batches,
                "schedule": self.ppo.schedule,
                "use_clipped_value_loss": self.ppo.use_clipped_value_loss,
                "value_loss_coef": self.ppo.value_loss_coef,
            },
            "init_member_classes": {},
            "policy": {
                "class_name": "ActorCriticTanh",
                "activation": self.policy.activation,
                "actor_hidden_dims": list(self.policy.actor_hidden_dims),
                "critic_hidden_dims": list(self.policy.critic_hidden_dims),
                "init_noise_std": self.policy.init_noise_std,
                "rnn_type": self.policy.rnn_type,
                "rnn_hidden_size": self.policy.rnn_hidden_size,
                "rnn_num_layers": self.policy.rnn_num_layers,
                "max_servo": self.policy.max_servo,
                "max_throttle": self.policy.max_throttle,
            },
            "runner": {
                "algorithm_class_name": "PPO",
                "checkpoint": -1,
                "experiment_name": self.exp_name,
                "load_run": -1,
                "log_interval": 1,
                "max_iterations": self.training.max_iterations,
                "policy_class_name": "ActorCriticTanh",
                "record_interval": -1,
                "resume": False,
                "resume_path": None,
                "run_name": "",
                "runner_class_name": "OnPolicyRunner",
            },
        }

    def to_env_cfg(self) -> Dict[str, Any]:
        """Build the ``env_cfg`` dict consumed by ``WingedDroneEnv``.

        Returns
        -------
        Dict[str, Any]
            Flat dictionary with all environment parameters.
        """
        return asdict(self.env)

    def to_obs_cfg(self) -> Dict[str, Any]:
        """Build the ``obs_cfg`` dict consumed by ``ObservationBuilder``.

        Re-nests the flat ``noise_std_*`` fields into the
        ``{"noise_std": {...}}`` sub-dict that the builder expects.

        Returns
        -------
        Dict[str, Any]
            Observation configuration with nested noise-std mapping.
        """
        return {
            "num_obs": self.obs.num_obs,
            "add_noise": self.obs.add_noise,
            "add_genome_obs_actor": self.obs.add_genome_obs_actor,
            "add_genome_obs_critic": self.obs.add_genome_obs_critic,
            "noise_std": {
                "z": self.obs.noise_std_z,
                "quat": self.obs.noise_std_quat,
                "vel": self.obs.noise_std_vel,
                "depth": self.obs.noise_std_depth,
                "last_thr": self.obs.noise_std_last_thr,
                "last_jnts": self.obs.noise_std_last_jnts,
                "v_tgt": self.obs.noise_std_v_tgt,
                "genome": self.obs.noise_std_genome,
            },
        }

    def to_reward_cfg(self) -> Dict[str, Any]:
        """Build the ``reward_cfg`` dict consumed by ``WingedDroneEnv``.

        Returns
        -------
        Dict[str, Any]
            ``{"reward_scales": {name: weight, ...}}``.
        """
        return {"reward_scales": asdict(self.reward)}

    def to_command_cfg(self) -> Dict[str, Any]:
        """Build the ``command_cfg`` dict consumed by ``WingedDroneEnv``.

        Returns
        -------
        Dict[str, Any]
            Command configuration (currently just ``num_commands``).
        """
        return asdict(self.command)

    def to_legacy_cfgs(self) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Convert this config into the five legacy dicts.

        Returns
        -------
        Tuple[Dict, Dict, Dict, Dict, Dict]
            ``(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg)`` — the
            exact format expected by ``WingedDroneEnv`` and ``OnPolicyRunner``.
        """
        return (
            self.to_env_cfg(),
            self.to_obs_cfg(),
            self.to_reward_cfg(),
            self.to_command_cfg(),
            self.to_train_cfg(),
        )

    # ------------------------------------------------------------------ #
    # YAML serialization
    # ------------------------------------------------------------------ #

    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Serialise this config to a YAML string.

        Parameters
        ----------
        path : Path, optional
            If given, the YAML is also written to this file (parent dirs
            are created automatically).

        Returns
        -------
        str
            The YAML representation of the full config.
        """
        data = asdict(self)
        text = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        """Deserialise a ``RunConfig`` from a YAML file.

        Missing keys fall back to their dataclass defaults, so a YAML file
        only needs to specify the values that differ from the defaults.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        RunConfig
            Fully-populated configuration object.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        """Recursively construct a ``RunConfig`` from a nested dict.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary (typically from ``yaml.safe_load``).  Keys that do
            not correspond to a known field are silently ignored.

        Returns
        -------
        RunConfig
            Populated config with defaults for any missing keys.
        """
        cfg = cls()
        sub_map = {
            "ppo": PPOConfig,
            "policy": PolicyConfig,
            "training": TrainingConfig,
            "env": EnvConfig,
            "obs": ObsConfig,
            "reward": RewardConfig,
            "command": CommandConfig,
            "catalog": CatalogConfig,
        }
        for key, val in data.items():
            if key in sub_map and isinstance(val, dict):
                sub = sub_map[key]()
                for sk, sv in val.items():
                    if hasattr(sub, sk):
                        setattr(sub, sk, sv)
                setattr(cfg, key, sub)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    # ------------------------------------------------------------------ #
    # CLI override support
    # ------------------------------------------------------------------ #

    def apply_cli_overrides(self, argv: Optional[List[str]] = None) -> None:
        """Apply ``--cfg.section.key value`` overrides from the command line.

        Scans *argv* for arguments of the form ``--cfg.<section>.<key>``
        followed by a value token.  The value is automatically cast to the
        type of the existing field (bool, int, float, list, or str).

        Top-level fields (e.g. ``exp_name``) can be set with a single dot:
        ``--cfg.exp_name my-experiment``.

        Parameters
        ----------
        argv : List[str], optional
            Argument list to scan.  Defaults to ``sys.argv[1:]``.

        Examples
        --------
        .. code-block:: bash

            --cfg.ppo.learning_rate 3e-4
            --cfg.training.num_envs 4096
            --cfg.reward.crash -20.0
            --cfg.policy.actor_hidden_dims [128,128]
            --cfg.exp_name my-experiment
        """
        if argv is None:
            argv = sys.argv[1:]
        i = 0
        while i < len(argv):
            arg = argv[i]
            if arg.startswith("--cfg."):
                parts = arg[len("--cfg."):].split(".")
                if len(parts) == 2 and i + 1 < len(argv):
                    section, key = parts
                    val_str = argv[i + 1]
                    sub = getattr(self, section, None)
                    if sub is not None and hasattr(sub, key):
                        current = getattr(sub, key)
                        setattr(sub, key, _cast(val_str, current))
                        i += 2
                        continue
                elif len(parts) == 1 and i + 1 < len(argv):
                    key = parts[0]
                    if hasattr(self, key):
                        current = getattr(self, key)
                        setattr(self, key, _cast(argv[i + 1], current))
                        i += 2
                        continue
            i += 1


def _cast(val_str: str, reference: Any) -> Any:
    """Cast a CLI string value to match the type of *reference*.

    Handles ``bool``, ``int``, ``float``, ``list`` (comma-separated),
    and falls back to ``str``.

    Parameters
    ----------
    val_str : str
        Raw string from the command line.
    reference : Any
        The current value of the field — its type determines the cast.

    Returns
    -------
    Any
        The value cast to the appropriate Python type.
    """
    if isinstance(reference, bool):
        return val_str.lower() in ("true", "1", "yes")
    if isinstance(reference, int):
        return int(val_str)
    if isinstance(reference, float):
        return float(val_str)
    if isinstance(reference, list):
        # e.g. "[64,128]"
        val_str = val_str.strip("[]")
        items = [s.strip() for s in val_str.split(",")]
        if reference and isinstance(reference[0], int):
            return [int(x) for x in items]
        return [float(x) for x in items]
    return val_str
