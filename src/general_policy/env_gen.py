from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import genesis as gs

from winged_drone_train.env import WingedDroneEnv
from winged_drone_train.train import configure_solver_noise


def _split_even(total: int, k: int) -> List[int]:
    """
    Split `total` as evenly as possible into `k` integer parts.

    Example:
        total = 10, k = 3 → [4, 3, 3]
    """
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


class Gen_Env:
    """
    Multi-URDF vectorized environment on top of `WingedDroneEnv`.

    Design
    ------
    - We always use **all** URDFs found in the catalog (or passed via `urdf_list`).
    - For each URDF we create **one** Genesis scene (one `WingedDroneEnv`).
    - The total number of parallel envs `num_envs` is split evenly across all URDFs.

    This means:
        K = number of URDFs
        sizes = split_even(num_envs, K)
        sum(sizes) = num_envs

    Requirement: num_envs >= K
        (each URDF must have at least one environment).
    """

    def __init__(
        self,
        num_envs: int,
        env_cfg: Dict,
        obs_cfg: Dict,
        reward_cfg: Dict,
        command_cfg: Dict,
        catalog_dir: Optional[str] = None,
        urdf_list: Optional[List[str]] = None,
        max_scenes: Optional[int] = None,
        show_viewer: bool = False,
        eval: bool = False,
        device: str = "cuda",
        # Debug options
        debug: Optional[bool] = None,
        debug_every: Optional[int] = None,
        peek_envs: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        # ------------------------------------------------------------------ #
        # Basic configuration                                               #
        # ------------------------------------------------------------------ #
        self.device = torch.device(device)
        self.eval_mode = bool(eval)

        # Debug controls
        self._dbg_on = True if debug is None else bool(debug)
        self._dbg_every = int(
            os.getenv("FP_DEBUG_EVERY", "50") if debug_every is None else debug_every
        )
        self._dbg_peek = int(
            os.getenv("FP_PEEK_ENVS", "0") if peek_envs is None else peek_envs
        )
        self._progress_callback = progress_callback
        self._t = 0
        self._tic = time.time()

        # ------------------------------------------------------------------ #
        # 1) Resolve URDF list                                              #
        # ------------------------------------------------------------------ #
        urdf_list = self._resolve_urdf_list(catalog_dir=catalog_dir, urdf_list=urdf_list)

        if not urdf_list:
            raise RuntimeError(
                "[Gen_Env] Empty URDF catalog. "
                "Either provide `urdf_list` or set URDF_CATALOG_DIR / catalog_dir."
            )

        # Use *all* URDFs we have
        K = len(urdf_list)

        # Each URDF must be used by at least one env
        if num_envs < K:
            raise RuntimeError(
                f"[Gen_Env] num_envs={num_envs} is smaller than the number of URDFs "
                f"({K}). Each URDF must have at least one environment. "
                "Increase num_envs or reduce the number of URDFs in the catalog."
            )

        # Split num_envs evenly across K URDFs
        sizes = _split_even(num_envs, K)
        self.num_envs = int(num_envs)

        # If user gives a restrictive max_scenes, we override it to satisfy:
        #   one scene per URDF.
        if max_scenes is not None and max_scenes < K:
            print(
                f"[Gen_Env] Warning: requested max_scenes={max_scenes} "
                f"but we have {K} URDFs. Overriding max_scenes to {K} "
                "so that all URDFs are used."
            )
            max_scenes = K

        if max_scenes is None:
            max_scenes = K

        # Optionally shuffle URDF order to decorrelate scene layout
        #random.shuffle(urdf_list)
        chosen_urdfs = urdf_list  # K entries, one per scene

        print(f"[Gen_Env] URDF scenes (K) = {K}, sizes per scene = {sizes}")
        print(
            "[Gen_Env] URDFs: "
            f"{[Path(p).name for p in chosen_urdfs[:min(K, 8)]]}"
            + (" ..." if K > 8 else "")
        )

        # ------------------------------------------------------------------ #
        # 2) Create sub-environments                                        #
        # ------------------------------------------------------------------ #
        self._subs: List[WingedDroneEnv] = []
        self._slices: List[slice] = []
        self.scene_init_total_s = 0.0
        self.scene_build_total_s = 0.0
        self.slowest_scene_init_s = 0.0
        self.slowest_scene_urdf = ""

        build_tic = time.time()
        gs.max_scenes = max_scenes  # cap in Genesis (>= K)

        start = 0
        for i, (count_i, urdf_i) in enumerate(zip(sizes, chosen_urdfs)):
            if count_i <= 0:
                continue

            stop = start + count_i
            sl = slice(start, stop)
            self._slices.append(sl)

            print(
                f"[Gen_Env] Building sub-env #{i:02d} with {count_i} envs "
                f"from URDF '{Path(urdf_i).name}'"
            )

            sub_env_cfg = dict(env_cfg)
            sub_obs_cfg = dict(obs_cfg)
            sub_reward_cfg = dict(reward_cfg)
            sub_command_cfg = dict(command_cfg)

            sub_init_start = time.perf_counter()
            sub = WingedDroneEnv(
                num_envs=count_i,
                env_cfg=sub_env_cfg,
                obs_cfg=sub_obs_cfg,
                reward_cfg=sub_reward_cfg,
                command_cfg=sub_command_cfg,
                urdf_file=urdf_i,
                show_viewer=(show_viewer and i == 0),
                eval=self.eval_mode,
                device=self._torch_device_str,
            )

            configure_solver_noise(sub, env_cfg)

            sub_init_elapsed = time.perf_counter() - sub_init_start
            per_env = sub_init_elapsed / max(1, count_i)
            print(
                f"[Gen_Env] sub-env #{i:02d} init time: "
                f"{sub_init_elapsed:.3f}s total, {per_env:.6f}s per env "
                f"(count={count_i}, urdf='{Path(urdf_i).name}')"
            )
            scene_init_s = float(getattr(sub, "scene_init_elapsed", 0.0))
            scene_build_s = float(getattr(sub, "scene_build_elapsed", 0.0))
            self.scene_init_total_s += scene_init_s
            self.scene_build_total_s += scene_build_s
            if scene_init_s >= self.slowest_scene_init_s:
                self.slowest_scene_init_s = scene_init_s
                self.slowest_scene_urdf = Path(urdf_i).name
            self._subs.append(sub)
            # Defer cache clearing to reduce GPU stall overhead
            start = stop

        # Clear cache once after all sub-envs are built
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"[Gen_Env] Created {len(self._subs)} sub-envs in "
            f"{time.time() - build_tic:.2f} s"
        )
        if self._subs:
            print(
                "[Gen_Env] scene summary: "
                f"scene_init_total_s={self.scene_init_total_s:.3f} "
                f"scene_build_total_s={self.scene_build_total_s:.3f} "
                f"slowest_scene_s={self.slowest_scene_init_s:.3f} "
                f"slowest_scene_urdf='{self.slowest_scene_urdf}'"
            )

        if not self._subs:
            raise RuntimeError("[Gen_Env] No sub-environments were created.")

        # ------------------------------------------------------------------ #
        # 3) Sanity checks and shared meta-data                             #
        # ------------------------------------------------------------------ #
        n_obs = {s.num_obs for s in self._subs}
        n_act = {s.num_actions for s in self._subs}
        n_priv = {s.num_privileged_obs for s in self._subs}
        if not (len(n_obs) == len(n_act) == len(n_priv) == 1):
            raise RuntimeError(
                "[Gen_Env] Incompatible sub-envs: "
                f"num_obs={n_obs}, num_actions={n_act}, num_priv={n_priv}"
            )

        self.num_obs = self._subs[0].num_obs
        self.num_actions = self._subs[0].num_actions
        self.num_privileged_obs = self._subs[0].num_privileged_obs
        self.dt = float(self._subs[0].dt)
        self.max_episode_length = int(self._subs[0].max_episode_length)

        self.max_episode_length_per_env = torch.cat(
            [s.max_episode_length_per_env.to(self.device) for s in self._subs],
            dim=0,
        )

        # ------------------------------------------------------------------ #
        # 4) Allocate unified buffers                                      #
        # ------------------------------------------------------------------ #
        B, O = self.num_envs, self.num_obs

        self.obs_buf = torch.zeros((B, O), device=self.device, dtype=torch.float32)
        self.privileged_obs_buf = torch.zeros(
            (B, self.num_privileged_obs),
            device=self.device,
            dtype=torch.float32,
        )
        self.rew_buf = torch.zeros((B,), device=self.device, dtype=torch.float32)
        self.reset_buf = torch.ones((B,), device=self.device, dtype=torch.int64)

        ep_dtype = self._subs[0].episode_length_buf.dtype
        self.episode_length_buf = torch.zeros(
            (B,), device=self.device, dtype=ep_dtype
        )

        # Extras dictionary, RSL-RL-style. Keep persistent buffers to avoid
        # per-step GPU allocations.
        self.extras: Dict = {
            "observations": {"critic": self.privileged_obs_buf},
            "time_outs": torch.zeros((B,), device=self.device, dtype=torch.float32),
        }

        print(
            f"[Gen_Env] B={self.num_envs}  dt={self.dt:.4f}  "
            f"num_obs={self.num_obs}  num_priv={self.num_privileged_obs}  "
            f"num_actions={self.num_actions}"
        )

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the first sub-environment."""
        # Avoid infinite recursion for attributes accessed during __init__
        if name.startswith("_"):
            raise AttributeError(name)
        subs = self.__dict__.get("_subs")
        if subs:
            return getattr(subs[0], name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @property
    def _torch_device_str(self) -> str:
        """Return device string (`'cuda:0'`, `'cpu'`, ...) for WingedDroneEnv."""
        if self.device.index is None:
            return self.device.type
        return f"{self.device.type}:{self.device.index}"

    def _resolve_urdf_list(
        self,
        catalog_dir: Optional[str],
        urdf_list: Optional[List[str]],
    ) -> List[str]:
        """
        Resolve the list of URDF files.

        Priority:
          1. If `urdf_list` is provided, use it directly.
          2. Else, search `catalog_dir` or $URDF_CATALOG_DIR:
             - if 'catalog.txt' exists, read URDF names/paths from there;
             - otherwise, use all '*.urdf' files in that folder.
        """
        if urdf_list is not None:
            return [str(p) for p in urdf_list]

        base_dir = Path(
            catalog_dir or os.getenv("URDF_CATALOG_DIR", "")
        ).expanduser()

        if not base_dir or not base_dir.exists():
            return []

        catalog_file = base_dir / "catalog.txt"
        if catalog_file.exists():
            lines = [s.strip() for s in catalog_file.read_text().splitlines() if s.strip()]
            resolved: List[str] = []
            for s in lines:
                # If path is relative, assume it is under catalog_dir
                p = Path(s)
                if not p.is_absolute():
                    p = base_dir / p
                resolved.append(str(p))
            return resolved

        # Fallback: all URDFs in the folder
        return sorted(str(p) for p in base_dir.glob("*.urdf"))

    def _prepare_extras_for_step(self) -> None:
        """Reset reusable extras buffers before reset/step aggregation."""
        self.extras["time_outs"].zero_()
        if "episode" in self.extras:
            del self.extras["episode"]

    def _store_critic_obs(self, critic_obs: Optional[torch.Tensor], sl: slice) -> None:
        """Store critic observations for a sub-slice into `self.extras`."""
        critic_buf = self.extras["observations"]["critic"]
        if critic_obs is None:
            critic_buf[sl].zero_()
            return
        critic_buf[sl] = critic_obs.to(self.device)

    def _copy_reset_from_sub(
        self,
        sub: WingedDroneEnv,
        sl: slice,
        obs: torch.Tensor,
        info: Dict,
    ) -> None:
        """Copy reset outputs from a sub-env into unified buffers."""
        self.obs_buf[sl] = obs.to(self.device)
        self.episode_length_buf[sl] = sub.episode_length_buf.to(self.device)

        critic_obs = info.get("observations", {}).get("critic")
        self._store_critic_obs(critic_obs, sl)

        if "time_outs" in info:
            self.extras["time_outs"][sl] = info["time_outs"].to(self.device).float()
        else:
            self.extras["time_outs"][sl].zero_()

    def _copy_step_from_sub(
        self,
        sub: WingedDroneEnv,
        sl: slice,
        obs_sub: torch.Tensor,
        rew_sub: torch.Tensor,
        done_sub: torch.Tensor,
        info_sub: Dict,
    ) -> None:
        """Copy step outputs from a sub-env into unified buffers."""
        self.obs_buf[sl] = obs_sub.to(self.device)
        self.rew_buf[sl] = rew_sub.to(self.device)
        self.reset_buf[sl] = done_sub.to(self.device)
        self.episode_length_buf[sl] = sub.episode_length_buf.to(self.device)

        critic_obs = info_sub.get("observations", {}).get("critic")
        self._store_critic_obs(critic_obs, sl)

        time_outs = info_sub.get("time_outs")
        if time_outs is not None:
            self.extras["time_outs"][sl] = time_outs.to(self.device).float()
        else:
            self.extras["time_outs"][sl].zero_()

    # ------------------------------------------------------------------ #
    # Debug utilities                                                    #
    # ------------------------------------------------------------------ #

    def _ms(self, t: torch.Tensor, n: int = 6) -> str:
        """Return compact 'mean ± std' string for a tensor."""
        if not torch.is_tensor(t) or t.numel() == 0:
            return "-"
        if t.ndim == 2:
            t = t[:, : min(n, t.shape[1])]
        m = torch.mean(t, dim=0)
        s = torch.std(t, dim=0)
        return " | ".join(f"{float(mi):+.3f}±{float(si):.3f}" for mi, si in zip(m, s))

    def _peek_ids(self, sl: slice, k: int) -> List[int]:
        """
        Return up to `k` approximately evenly spaced global env indices
        belonging to the given slice.
        """
        idx = list(range(sl.start, sl.stop))
        if len(idx) <= k:
            return idx
        step = max(1, (len(idx) - 1) // max(1, k))
        return [idx[i] for i in range(0, len(idx), step)][:k]

    def _print_debug(self, actions: torch.Tensor) -> None:
        """
        Print aggregated statistics for each sub-env every `self._dbg_every`
        calls to `step()`.
        """
        if not self._dbg_on:
            return
        if (self._t % max(1, self._dbg_every)) != 0:
            return

        elapsed = time.time() - self._tic
        fps = self._t / max(elapsed, 1e-6)

        print(
            f"[Gen_Env: debug] t={self._t}  elapsed={elapsed:.1f}s  "
            f"approx FPS={fps:.1f}"
        )

        for k, (sub, sl) in enumerate(zip(self._subs, self._slices)):
            r_slice = self.rew_buf[sl]
            r_mean = float(r_slice.mean()) if r_slice.numel() else float("nan")
            r_std = float(r_slice.std()) if r_slice.numel() > 1 else 0.0

            resets = int(self.reset_buf[sl].sum().item())

            ep = getattr(sub, "extras", {}).get("episode", {}) if hasattr(sub, "extras") else {}
            final_x = float(ep.get("final_x", float("nan")))
            final_z = float(ep.get("final_z", float("nan")))
            num_crashed = float(
                ep.get("num_wall_crashed", 0.0) + ep.get("num_angle_crashed", 0.0)
            )
            num_success = float(ep.get("num_success", 0.0))
            num_collision = float(ep.get("num_collision", 0.0))

            print(
                f"  sub={k:02d} | r={r_mean:+.3f}±{r_std:.3f} | "
                f"resets={resets:4d} | final_x={final_x:+.3f} "
                f"final_z={final_z:+.3f} | "
                f"crash={num_crashed:.0f} success={num_success:.0f} "
                f"collision={num_collision:.0f}"
            )

        if self._dbg_peek > 0:
            peek_idx = self._peek_ids(slice(0, self.num_envs), self._dbg_peek)
            print(
                f"  actions (first {self._dbg_peek} envs): "
                f"{self._ms(actions[peek_idx])}"
            )

    # ------------------------------------------------------------------ #
    # VecEnv-like API                                                    #
    # ------------------------------------------------------------------ #

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """
        Reset all sub-environments and return initial observations and extras.
        """
        self._prepare_extras_for_step()

        for sub, sl in zip(self._subs, self._slices):
            obs, info = sub.reset()
            self._copy_reset_from_sub(sub, sl, obs, info)

        self.reset_buf.fill_(1)
        self._t = 0

        print("[Gen_Env] Reset completed. Buffers initialized.")
        return self.obs_buf, self.extras

    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step all sub-environments with a single batch of actions.
        """
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"[Gen_Env.step] Expected actions of shape "
                f"({self.num_envs}, {self.num_actions}), got {tuple(actions.shape)}"
            )

        actions = actions.to(self.device)

        self._prepare_extras_for_step()

        episodes_list: List[Tuple[Dict, int]] = []

        for sub, sl in zip(self._subs, self._slices):
            obs_sub, rew_sub, done_sub, info_sub = sub.step(actions[sl])
            self._copy_step_from_sub(sub, sl, obs_sub, rew_sub, done_sub, info_sub)

            ep = info_sub.get("episode")
            if ep:
                try:
                    count = int(done_sub.sum().item())
                except Exception:
                    count = 1
                episodes_list.append((ep, max(count, 1)))

        if episodes_list:
            aggregated: Dict[str, float] = {}
            total_weight = 0
            for ep_dict, weight in episodes_list:
                total_weight += weight
                for k, v in ep_dict.items():
                    try:
                        aggregated[k] = aggregated.get(k, 0.0) + float(v) * weight
                    except Exception:
                        pass

            if total_weight > 0:
                for k in list(aggregated.keys()):
                    aggregated[k] /= float(total_weight)
                self.extras["episode"] = aggregated

        self._t += 1
        self._print_debug(actions)

        # Per-step empty_cache() forces a CUDA driver round-trip and stalls the
        # caching allocator; PyTorch reuses memory automatically. Removed: it was
        # costing ~30 ms/step (~25% slowdown vs single-WingedDroneEnv path).
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------ #
    # Convenience getters                                                #
    # ------------------------------------------------------------------ #

    def get_observations(self) -> Tuple[torch.Tensor, Dict]:
        """Return current actor observations and a shallow copy of extras."""
        return self.obs_buf, dict(self.extras)

    def get_privileged_observations(self) -> Tuple[torch.Tensor, Dict]:
        """
        Construct privileged observations from actor + extra critic features.
        """
        critic = self.extras.get("observations", {}).get("critic")
        if critic is not None and critic.shape[1] > self.num_obs:
            extra = critic[:, self.num_obs :]
            return torch.cat([self.obs_buf, extra], dim=1), {}
        return self.obs_buf, {}

    def close(self) -> None:
        """Explicitly delete sub-environments to help Genesis clean up scenes."""
        try:
            for sub in self._subs:
                del sub
        except Exception:
            pass
