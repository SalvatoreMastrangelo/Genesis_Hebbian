from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


_TB_ADD_SCALAR_PATCHED = False
_TB_ORIG_ADD_SCALAR = None
_TB_LOGGER_BY_DIR: Dict[str, "RLTrainingLogger"] = {}


class RLTrainingLogger:
    """Attach robust PPO diagnostics logging/printing to an RSL-RL runner."""

    def __init__(self, runner: Any, log_dir: Path, max_iterations: Optional[int] = None) -> None:
        self.runner = runner
        self.log_dir = Path(log_dir)
        self.max_iterations = int(max_iterations) if max_iterations is not None else 0

        self.writer = None
        self.owns_writer = False

        self.alg = getattr(self.runner, "alg", None)
        self.original_update = getattr(self.alg, "update", None) if self.alg is not None else None
        self.update_step = {"i": 0}
        self.cuda_mem_log_every = max(1, int(os.getenv("PPO_CUDA_MEM_LOG_EVERY", "5") or 5))
        self.csv_path = self.log_dir / "tensorboard_1pct.csv"
        self.current_scalars: Dict[str, float] = {}
        self.pending_csv_step: Optional[int] = None
        self.next_csv_percent = 1
        self.csv_rows: list[Dict[str, float]] = []
        self.csv_columns: list[str] = ["progress_pct", "step"]
        self.csv_tag_columns: Dict[str, str] = {}
        self.csv_column_counts: Dict[str, int] = {}

        self.internal_capture: Dict[str, Any] = {
            "enabled": False,
            "pending_old_logp": [],
            "ratio_sum": 0.0,
            "ratio_count": 0,
            "kl_sum": 0.0,
            "kl_count": 0,
            "clip_sum": 0.0,
            "clip_count": 0,
            "entropy_sum": 0.0,
            "entropy_count": 0,
        }

        self.wrapped_gen_methods: Dict[str, Any] = {}
        self.wrapped_get_actions_log_prob = None
        self.actor_critic = getattr(self.alg, "actor_critic", None) if self.alg is not None else None
        self.storage_obj = getattr(self.alg, "storage", None) if self.alg is not None else None
        if self.storage_obj is None and self.alg is not None:
            self.storage_obj = getattr(self.alg, "rollout_storage", None)

    def attach(self) -> None:
        self._setup_writer()
        self._patch_storage_generators()
        self._patch_get_actions_log_prob()
        self._patch_alg_update()

    def close(self) -> None:
        try:
            if self.alg is not None and callable(self.original_update):
                self.alg.update = self.original_update
        except Exception:
            pass
        try:
            self._flush_pending_csv_step()
        except Exception:
            pass
        try:
            if self.actor_critic is not None and self.wrapped_get_actions_log_prob is not None:
                self.actor_critic.get_actions_log_prob = self.wrapped_get_actions_log_prob
        except Exception:
            pass
        try:
            if self.storage_obj is not None:
                for method_name, method_orig in self.wrapped_gen_methods.items():
                    setattr(self.storage_obj, method_name, method_orig)
        except Exception:
            pass
        _TB_LOGGER_BY_DIR.pop(str(self.log_dir.resolve()), None)
        try:
            if self.owns_writer and self.writer is not None:
                self.writer.close()
        except Exception:
            pass

    def _setup_writer(self) -> None:
        try:
            logger = getattr(self.runner, "logger", None)
            if logger is not None and hasattr(logger, "writer"):
                self.writer = logger.writer
            if self.writer is None:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                self.owns_writer = True
        except Exception:
            self.writer = None
            self.owns_writer = False
        self._register_tensorboard_hook()

    def _register_tensorboard_hook(self) -> None:
        global _TB_ADD_SCALAR_PATCHED, _TB_ORIG_ADD_SCALAR
        try:
            resolved_log_dir = str(self.log_dir.resolve())
        except Exception:
            resolved_log_dir = str(self.log_dir)
        _TB_LOGGER_BY_DIR[resolved_log_dir] = self
        if _TB_ADD_SCALAR_PATCHED:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            _TB_ORIG_ADD_SCALAR = SummaryWriter.add_scalar

            def _wrapped_add_scalar(writer_self: Any, tag: str, scalar_value: Any, global_step: Optional[int] = None, *args: Any, **kwargs: Any) -> Any:
                out = _TB_ORIG_ADD_SCALAR(writer_self, tag, scalar_value, global_step, *args, **kwargs)
                try:
                    writer_log_dir = getattr(writer_self, "log_dir", None)
                    if writer_log_dir is not None:
                        logger = _TB_LOGGER_BY_DIR.get(str(Path(writer_log_dir).resolve()))
                        if logger is not None:
                            logger._track_scalar_for_csv(tag, scalar_value, global_step)
                except Exception:
                    pass
                return out

            SummaryWriter.add_scalar = _wrapped_add_scalar
            _TB_ADD_SCALAR_PATCHED = True
        except Exception:
            pass

    def _ensure_csv_file_parent(self) -> bool:
        if self.max_iterations <= 0:
            return False
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    def _simplify_tag_name(self, tag: str) -> str:
        txt = str(tag).strip().replace("\\", "/")
        replacements = (
            ("Episode/", "ep_"),
            ("Train/", "train_"),
            ("Loss/", "loss_"),
            ("Policy/", "policy_"),
            ("PPO/", "ppo_"),
            ("Opt/", "opt_"),
            ("Adv/", "adv_"),
            ("CUDA/", "cuda_"),
            ("Critic/", "critic_"),
        )
        for prefix, repl in replacements:
            if txt.startswith(prefix):
                txt = repl + txt[len(prefix):]
                break
        txt = txt.replace("/", "_").replace(".", "_").replace("-", "_").replace(" ", "_")
        txt = "".join(ch.lower() if ch.isalnum() or ch == "_" else "_" for ch in txt)
        while "__" in txt:
            txt = txt.replace("__", "_")
        txt = txt.strip("_") or "metric"
        return txt

    def _column_name_for_tag(self, tag: str) -> str:
        existing = self.csv_tag_columns.get(tag)
        if existing is not None:
            return existing
        base = self._simplify_tag_name(tag)
        count = self.csv_column_counts.get(base, 0)
        col = base if count == 0 else f"{base}_{count + 1}"
        self.csv_column_counts[base] = count + 1
        self.csv_tag_columns[tag] = col
        if col not in self.csv_columns:
            self.csv_columns.append(col)
        return col

    def _write_csv_snapshot_table(self) -> None:
        if not self._ensure_csv_file_parent():
            return
        try:
            with self.csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns, extrasaction="ignore")
                writer.writeheader()
                for row in self.csv_rows:
                    writer.writerow(row)
        except Exception:
            pass

    def _track_scalar_for_csv(self, tag: Any, scalar_value: Any, global_step: Optional[int]) -> None:
        if self.max_iterations <= 0 or global_step is None:
            return
        try:
            step = int(global_step)
        except Exception:
            return

        scalar = self._to_float(scalar_value)
        if scalar is None:
            return

        if self.pending_csv_step is not None and step != self.pending_csv_step:
            self._flush_pending_csv_step()

        self.pending_csv_step = step
        self.current_scalars[str(tag)] = scalar

    def _flush_pending_csv_step(self) -> None:
        if self.pending_csv_step is None or self.max_iterations <= 0:
            return

        completed_pct = int(((self.pending_csv_step + 1) * 100) // max(1, self.max_iterations))
        if completed_pct <= 0:
            return

        while self.next_csv_percent <= min(100, completed_pct):
            row: Dict[str, float] = {
                "progress_pct": float(self.next_csv_percent),
                "step": float(self.pending_csv_step),
            }
            for tag in sorted(self.current_scalars):
                row[self._column_name_for_tag(tag)] = self.current_scalars[tag]
            self.csv_rows.append(row)
            self.next_csv_percent += 1
        self._write_csv_snapshot_table()

    def _to_float(self, val: Any) -> Optional[float]:
        try:
            if val is None:
                return None
            if isinstance(val, (int, float)):
                out = float(val)
            elif torch.is_tensor(val):
                if val.numel() != 1:
                    return None
                out = float(val.detach().cpu().item())
            else:
                out = float(val)
            if out != out or out == float("inf") or out == float("-inf"):
                return None
            return out
        except Exception:
            return None

    def _mean_tensor(self, val: Any) -> Optional[float]:
        try:
            if val is None:
                return None
            if torch.is_tensor(val):
                if val.numel() == 0:
                    return None
                return float(val.detach().float().mean().cpu().item())
            return self._to_float(val)
        except Exception:
            return None

    def _safe_add_scalar(self, tag: str, value: Any, step: int) -> None:
        if self.writer is None:
            return
        scalar = self._to_float(value)
        if scalar is None:
            return
        try:
            self.writer.add_scalar(tag, scalar, step)
        except Exception:
            pass

    def _collect_named_scalars(self, obj: Any, prefix: str = "") -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            if obj is None:
                return out
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = f"{prefix}.{k}" if prefix else str(k)
                    out.update(self._collect_named_scalars(v, key))
                return out
            if isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    key = f"{prefix}.{i}" if prefix else str(i)
                    out.update(self._collect_named_scalars(v, key))
                return out
            scalar = self._to_float(obj)
            if scalar is not None and prefix:
                out[prefix] = scalar
        except Exception:
            return out
        return out

    def _pick_metric(self, scalars: Dict[str, float], aliases: Tuple[str, ...]) -> Optional[float]:
        if not scalars:
            return None
        lowered = {k.lower(): v for k, v in scalars.items()}
        for alias in aliases:
            a = alias.lower()
            for k, v in lowered.items():
                if k == a or k.endswith(f".{a}") or a in k:
                    return v
        return None

    def _collect_storage_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        try:
            storage = getattr(self.alg, "storage", None)
            if storage is None:
                storage = getattr(self.alg, "rollout_storage", None)
            if storage is None:
                return stats

            advantages = getattr(storage, "advantages", None)
            if advantages is None:
                advantages = getattr(storage, "advantage", None)
            if torch.is_tensor(advantages) and advantages.numel() > 0:
                adv = advantages.detach().float()
                stats["Adv/mean"] = float(adv.mean().cpu().item())
                stats["Adv/std"] = float(adv.std(unbiased=False).cpu().item())
                stats["Adv/max_abs"] = float(adv.abs().max().cpu().item())

            returns = getattr(storage, "returns", None)
            values = getattr(storage, "values", None)
            if torch.is_tensor(returns) and torch.is_tensor(values):
                r = returns.detach().float()
                v = values.detach().float()
                if r.shape == v.shape and r.numel() > 1:
                    var_r = torch.var(r, unbiased=False)
                    if float(var_r.cpu().item()) > 1e-12:
                        ev = 1.0 - torch.var(r - v, unbiased=False) / var_r
                        stats["Critic/explained_variance"] = float(ev.cpu().item())
        except Exception:
            return stats
        return stats

    def _collect_action_std(self) -> Optional[float]:
        try:
            actor_critic = getattr(self.alg, "actor_critic", None)
            if actor_critic is None:
                return None
            for name in ("action_std", "std", "sigma", "action_sigma"):
                if hasattr(actor_critic, name):
                    val = getattr(actor_critic, name)
                    if callable(val):
                        try:
                            val = val()
                        except Exception:
                            continue
                    m = self._mean_tensor(val)
                    if m is not None:
                        return m
            if hasattr(actor_critic, "log_std"):
                log_std = getattr(actor_critic, "log_std")
                if callable(log_std):
                    try:
                        log_std = log_std()
                    except Exception:
                        log_std = None
                if torch.is_tensor(log_std):
                    return self._mean_tensor(torch.exp(log_std.detach().float()))
        except Exception:
            return None
        return None

    def _collect_opt_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        try:
            optimizer = getattr(self.alg, "optimizer", None)
            if optimizer is not None:
                try:
                    if optimizer.param_groups:
                        lr_val = optimizer.param_groups[0].get("lr", None)
                        if lr_val is not None:
                            stats["Opt/lr"] = float(lr_val)
                except Exception:
                    pass

                try:
                    total = None
                    for group in optimizer.param_groups:
                        for p in group.get("params", []):
                            g = getattr(p, "grad", None)
                            if g is None:
                                continue
                            gn2 = g.detach().float().norm(2) ** 2
                            total = gn2 if total is None else (total + gn2)
                    if total is not None:
                        stats["Opt/grad_norm"] = float(torch.sqrt(total).cpu().item())
                except Exception:
                    pass

            for attr, tag in (
                ("grad_norm", "Opt/grad_norm"),
                ("gradient_norm", "Opt/grad_norm"),
                ("grad_norm_before_clip", "Opt/grad_norm_before_clip"),
                ("grad_norm_after_clip", "Opt/grad_norm_after_clip"),
                ("last_grad_norm", "Opt/grad_norm"),
                ("last_grad_norm_before_clip", "Opt/grad_norm_before_clip"),
                ("last_grad_norm_after_clip", "Opt/grad_norm_after_clip"),
                ("num_early_stop", "PPO/early_stop_count"),
                ("early_stop_count", "PPO/early_stop_count"),
                ("kl_stop_count", "PPO/early_stop_count"),
                ("update_skipped", "PPO/update_skipped"),
                ("num_updates_skipped", "PPO/update_skipped"),
            ):
                if hasattr(self.alg, attr):
                    val = self._to_float(getattr(self.alg, attr))
                    if val is not None:
                        stats[tag] = val
        except Exception:
            return stats
        return stats

    def _collect_cuda_mem_stats(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        try:
            if not torch.cuda.is_available():
                return stats
            alloc = float(torch.cuda.memory_allocated() / (1024.0 ** 3))
            reserved = float(torch.cuda.memory_reserved() / (1024.0 ** 3))
            max_alloc = float(torch.cuda.max_memory_allocated() / (1024.0 ** 3))
            stats["CUDA/memory_allocated_gb"] = alloc
            stats["CUDA/memory_reserved_gb"] = reserved
            stats["CUDA/max_memory_allocated_gb"] = max_alloc
            if reserved > 1e-9:
                stats["CUDA/alloc_reserved_ratio"] = alloc / reserved
        except Exception:
            return stats
        return stats

    def _flatten_batch_tensor(self, x: Any) -> Optional[torch.Tensor]:
        try:
            if not torch.is_tensor(x):
                return None
            if x.numel() == 0:
                return None
            if x.dim() >= 2:
                return x.reshape(-1, *x.shape[2:]) if x.dim() > 2 else x.reshape(-1)
            return x.reshape(-1)
        except Exception:
            return None

    def _flatten_vector_tensor(self, x: Any) -> Optional[torch.Tensor]:
        try:
            if not torch.is_tensor(x) or x.numel() == 0:
                return None
            return x.reshape(-1)
        except Exception:
            return None

    def _compute_policy_diagnostics_from_minibatch(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            storage = getattr(self.alg, "storage", None)
            if storage is None:
                storage = getattr(self.alg, "rollout_storage", None)
            actor_critic = getattr(self.alg, "actor_critic", None)
            if storage is None or actor_critic is None:
                return out

            gen_fn = None
            for gname in ("recurrent_mini_batch_generator", "reccurent_mini_batch_generator"):
                if hasattr(storage, gname):
                    gen_fn = getattr(storage, gname)
                    break
            if gen_fn is None and hasattr(storage, "mini_batch_generator"):
                gen_fn = getattr(storage, "mini_batch_generator")
            if gen_fn is None:
                return out

            try:
                gen = gen_fn(num_mini_batches=1, num_epochs=1)
            except TypeError:
                try:
                    gen = gen_fn(1, 1)
                except Exception:
                    return out

            try:
                batch = next(iter(gen))
            except Exception:
                return out
            if not isinstance(batch, (tuple, list)) or len(batch) < 7:
                return out

            obs_batch = batch[0]
            actions_batch = batch[2]
            old_logp_batch = batch[6]
            hid_states_batch = batch[9] if len(batch) > 9 else None
            masks_batch = batch[10] if len(batch) > 10 else None

            if not (torch.is_tensor(obs_batch) and torch.is_tensor(actions_batch) and torch.is_tensor(old_logp_batch)):
                return out

            with torch.no_grad():
                new_logp_batch = None
                try:
                    hidden_for_act = hid_states_batch
                    if isinstance(hid_states_batch, (tuple, list)) and len(hid_states_batch) > 0:
                        hidden_for_act = hid_states_batch[0]
                    _ = actor_critic.act(
                        obs_batch,
                        masks=masks_batch if torch.is_tensor(masks_batch) else None,
                        hidden_states=hidden_for_act,
                    )
                    new_logp_batch = actor_critic.get_actions_log_prob(actions_batch)
                except Exception:
                    try:
                        _ = actor_critic.act(obs_batch)
                        new_logp_batch = actor_critic.get_actions_log_prob(actions_batch)
                    except Exception:
                        return out

                old_logp = self._flatten_vector_tensor(old_logp_batch)
                new_logp = self._flatten_vector_tensor(new_logp_batch)
                if old_logp is None or new_logp is None:
                    return out
                m = min(old_logp.shape[0], new_logp.shape[0])
                if m <= 0:
                    return out
                old_logp = old_logp[:m].float()
                new_logp = new_logp[:m].float()

                finite_mask = torch.isfinite(old_logp) & torch.isfinite(new_logp)
                if torch.count_nonzero(finite_mask).item() <= 0:
                    return out
                old_logp = old_logp[finite_mask]
                new_logp = new_logp[finite_mask]

                log_ratio = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)
                finite_ratio = torch.isfinite(ratio) & torch.isfinite(log_ratio)
                if torch.count_nonzero(finite_ratio).item() <= 0:
                    return out
                ratio = ratio[finite_ratio]
                log_ratio = log_ratio[finite_ratio]

                out["PPO/ratio_mean"] = float(ratio.mean().cpu().item())
                approx_kl = torch.mean((ratio - 1.0) - log_ratio)
                kl_val = float(approx_kl.cpu().item())
                if kl_val == kl_val and kl_val != float("inf") and kl_val != float("-inf"):
                    out["PPO/kl"] = kl_val

                clip_param = getattr(self.alg, "clip_param", None)
                if clip_param is None:
                    try:
                        clip_param = getattr(getattr(self.alg, "cfg", None), "clip_param", None)
                    except Exception:
                        clip_param = None
                clip_v = self._to_float(clip_param)
                if clip_v is not None and clip_v > 0.0:
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_v).float())
                    clip_val = float(clip_frac.cpu().item())
                    if clip_val == clip_val and clip_val != float("inf") and clip_val != float("-inf"):
                        out["PPO/clip_fraction"] = clip_val
        except Exception:
            return out
        return out

    def _fmt_metric(self, v: Optional[float]) -> str:
        s = self._to_float(v)
        if s is None:
            return "n/a"
        try:
            return f"{float(s):.6g}"
        except Exception:
            return "n/a"

    def _extract_old_log_prob_from_batch(self, batch: Any) -> Optional[torch.Tensor]:
        try:
            if not isinstance(batch, (tuple, list)) or len(batch) == 0:
                return None
            if len(batch) > 6 and torch.is_tensor(batch[6]):
                return self._flatten_vector_tensor(batch[6])

            action_batch = batch[2] if len(batch) > 2 and torch.is_tensor(batch[2]) else None
            bsz = int(action_batch.shape[0]) if torch.is_tensor(action_batch) and action_batch.dim() > 0 else None
            candidates = []
            for item in batch:
                if not torch.is_tensor(item):
                    continue
                vec = self._flatten_vector_tensor(item)
                if vec is None:
                    continue
                if bsz is not None and vec.shape[0] != bsz:
                    continue
                if vec.dtype.is_floating_point:
                    candidates.append(vec)
            if not candidates:
                return None

            def _score(v: torch.Tensor) -> float:
                try:
                    m = float(torch.nanmean(v).cpu().item())
                    s = float(torch.nanstd(v).cpu().item())
                    neg_bias = 0.0 if m < 0.0 else 10.0
                    mag_penalty = abs(m) * 0.1 + s * 0.01
                    return neg_bias + mag_penalty
                except Exception:
                    return 1e9

            candidates.sort(key=_score)
            return candidates[0]
        except Exception:
            return None

    def _patch_storage_generators(self) -> None:
        def _wrap_generator_method(method_name: str) -> None:
            try:
                if self.storage_obj is None or not hasattr(self.storage_obj, method_name):
                    return
                orig = getattr(self.storage_obj, method_name)
                if not callable(orig):
                    return
                self.wrapped_gen_methods[method_name] = orig

                def _wrapped_gen(*g_args: Any, **g_kwargs: Any) -> Any:
                    gen = orig(*g_args, **g_kwargs)
                    for batch in gen:
                        if self.internal_capture["enabled"]:
                            old_lp = self._extract_old_log_prob_from_batch(batch)
                            if old_lp is not None:
                                self.internal_capture["pending_old_logp"].append(old_lp.detach())
                        yield batch

                setattr(self.storage_obj, method_name, _wrapped_gen)
            except Exception:
                pass

        for gen_name in ("recurrent_mini_batch_generator", "reccurent_mini_batch_generator", "mini_batch_generator"):
            _wrap_generator_method(gen_name)

    def _patch_get_actions_log_prob(self) -> None:
        if self.actor_critic is None or not hasattr(self.actor_critic, "get_actions_log_prob"):
            return
        try:
            orig_get_actions_log_prob = self.actor_critic.get_actions_log_prob

            def _wrapped_get_actions_log_prob(*ga_args: Any, **ga_kwargs: Any) -> Any:
                new_lp_out = orig_get_actions_log_prob(*ga_args, **ga_kwargs)
                if self.internal_capture["enabled"]:
                    try:
                        dist = getattr(self.actor_critic, "distribution", None)
                        if dist is not None and hasattr(dist, "entropy"):
                            ent = dist.entropy()
                            if torch.is_tensor(ent) and ent.numel() > 0:
                                ent_v = ent.detach().float()
                                ent_m = float(ent_v.mean().cpu().item())
                                if ent_m == ent_m and ent_m != float("inf") and ent_m != float("-inf"):
                                    self.internal_capture["entropy_sum"] += ent_m * int(ent_v.numel())
                                    self.internal_capture["entropy_count"] += int(ent_v.numel())

                        if self.internal_capture["pending_old_logp"]:
                            old_lp = self.internal_capture["pending_old_logp"].pop(0)
                            new_lp = self._flatten_vector_tensor(new_lp_out)
                            if new_lp is not None and old_lp is not None:
                                m = min(new_lp.shape[0], old_lp.shape[0])
                                if m > 0:
                                    new_lp = new_lp[:m].detach().float()
                                    old_lp = old_lp[:m].detach().float()
                                    finite = torch.isfinite(new_lp) & torch.isfinite(old_lp)
                                    if torch.count_nonzero(finite).item() > 0:
                                        new_lp = new_lp[finite]
                                        old_lp = old_lp[finite]
                                        log_ratio = torch.clamp(new_lp - old_lp, min=-20.0, max=20.0)
                                        ratio = torch.exp(log_ratio)
                                        finite2 = torch.isfinite(log_ratio) & torch.isfinite(ratio)
                                        if torch.count_nonzero(finite2).item() > 0:
                                            log_ratio = log_ratio[finite2]
                                            ratio = ratio[finite2]
                                            n = int(ratio.numel())
                                            ratio_mean = float(ratio.mean().cpu().item())
                                            kl_mean = float(torch.mean((ratio - 1.0) - log_ratio).cpu().item())
                                            if ratio_mean == ratio_mean and ratio_mean != float("inf") and ratio_mean != float("-inf"):
                                                self.internal_capture["ratio_sum"] += ratio_mean * n
                                                self.internal_capture["ratio_count"] += n
                                            if kl_mean == kl_mean and kl_mean != float("inf") and kl_mean != float("-inf"):
                                                self.internal_capture["kl_sum"] += kl_mean * n
                                                self.internal_capture["kl_count"] += n
                                            clip_v = self._to_float(getattr(self.alg, "clip_param", None))
                                            if clip_v is not None and clip_v > 0.0:
                                                clip_frac = float(torch.mean((torch.abs(ratio - 1.0) > clip_v).float()).cpu().item())
                                                if clip_frac == clip_frac and clip_frac != float("inf") and clip_frac != float("-inf"):
                                                    self.internal_capture["clip_sum"] += clip_frac * n
                                                    self.internal_capture["clip_count"] += n
                    except Exception:
                        pass
                return new_lp_out

            self.actor_critic.get_actions_log_prob = _wrapped_get_actions_log_prob
            self.wrapped_get_actions_log_prob = orig_get_actions_log_prob
        except Exception:
            self.wrapped_get_actions_log_prob = None

    def _patch_alg_update(self) -> None:
        if not callable(self.original_update):
            return

        def _wrapped_update(*u_args: Any, **u_kwargs: Any) -> Any:
            self.internal_capture["enabled"] = True
            self.internal_capture["pending_old_logp"] = []
            self.internal_capture["ratio_sum"] = 0.0
            self.internal_capture["ratio_count"] = 0
            self.internal_capture["kl_sum"] = 0.0
            self.internal_capture["kl_count"] = 0
            self.internal_capture["clip_sum"] = 0.0
            self.internal_capture["clip_count"] = 0
            self.internal_capture["entropy_sum"] = 0.0
            self.internal_capture["entropy_count"] = 0

            result = self.original_update(*u_args, **u_kwargs)
            self.internal_capture["enabled"] = False
            step = self.update_step["i"]
            console_metrics: Dict[str, Optional[float]] = {}

            try:
                raw_scalars = self._collect_named_scalars(result)
                if isinstance(result, tuple):
                    if len(result) >= 1:
                        v0 = self._to_float(result[0])
                        if v0 is not None:
                            self._safe_add_scalar("Loss/value", v0, step)
                        console_metrics["value"] = v0
                    if len(result) >= 2:
                        v1 = self._to_float(result[1])
                        if v1 is not None:
                            self._safe_add_scalar("Loss/surrogate", v1, step)
                        console_metrics["surrogate"] = v1
                elif isinstance(result, dict):
                    raw_scalars.update(self._collect_named_scalars(result))

                metric_specs: Dict[str, Tuple[str, ...]] = {
                    "PPO/kl": ("approx_kl", "kl", "mean_kl", "kl_mean", "kl_divergence"),
                    "PPO/clip_fraction": ("clip_fraction", "clipfrac", "clipped_fraction"),
                    "PPO/ratio_mean": ("ratio_mean", "mean_ratio", "policy_ratio", "ratio"),
                    "Loss/surrogate": ("surrogate_loss", "policy_loss", "actor_loss", "loss_pi"),
                    "Loss/value": ("value_loss", "critic_loss", "vf_loss", "loss_v"),
                    "Policy/entropy": ("entropy", "entropy_loss"),
                    "PPO/early_stop_count": ("early_stop", "early_stop_count", "kl_stop_count"),
                    "PPO/update_skipped": ("update_skipped", "num_updates_skipped", "skipped"),
                    "Opt/grad_norm": ("grad_norm", "gradient_norm", "last_grad_norm"),
                    "Opt/grad_norm_before_clip": ("grad_norm_before_clip", "last_grad_norm_before_clip"),
                    "Opt/grad_norm_after_clip": ("grad_norm_after_clip", "last_grad_norm_after_clip"),
                    "Opt/lr": ("lr", "learning_rate"),
                }

                for tag, aliases in metric_specs.items():
                    val = self._to_float(self._pick_metric(raw_scalars, aliases))
                    if val is not None:
                        self._safe_add_scalar(tag, val, step)
                    if tag == "PPO/kl":
                        console_metrics["kl"] = val
                    elif tag == "PPO/clip_fraction":
                        console_metrics["clip_frac"] = val
                    elif tag == "PPO/ratio_mean":
                        console_metrics["ratio"] = val
                    elif tag == "Loss/surrogate":
                        console_metrics["surrogate"] = val if val is not None else console_metrics.get("surrogate")
                    elif tag == "Loss/value":
                        console_metrics["value"] = val if val is not None else console_metrics.get("value")
                    elif tag == "Policy/entropy":
                        console_metrics["entropy"] = val
                    elif tag == "Opt/grad_norm":
                        console_metrics["grad"] = val
                    elif tag == "Opt/lr":
                        console_metrics["lr"] = val

                direct_policy_stats = self._compute_policy_diagnostics_from_minibatch()
                for tag, val in direct_policy_stats.items():
                    self._safe_add_scalar(tag, val, step)
                if "PPO/kl" in direct_policy_stats:
                    console_metrics["kl"] = direct_policy_stats["PPO/kl"]
                if "PPO/clip_fraction" in direct_policy_stats:
                    console_metrics["clip_frac"] = direct_policy_stats["PPO/clip_fraction"]
                if "PPO/ratio_mean" in direct_policy_stats:
                    console_metrics["ratio"] = direct_policy_stats["PPO/ratio_mean"]

                if self.internal_capture["ratio_count"] > 0:
                    r = self.internal_capture["ratio_sum"] / float(self.internal_capture["ratio_count"])
                    self._safe_add_scalar("PPO/ratio_mean", r, step)
                    console_metrics["ratio"] = r
                if self.internal_capture["kl_count"] > 0:
                    k = self.internal_capture["kl_sum"] / float(self.internal_capture["kl_count"])
                    self._safe_add_scalar("PPO/kl", k, step)
                    console_metrics["kl"] = k
                if self.internal_capture["clip_count"] > 0:
                    c = self.internal_capture["clip_sum"] / float(self.internal_capture["clip_count"])
                    self._safe_add_scalar("PPO/clip_fraction", c, step)
                    console_metrics["clip_frac"] = c
                if self.internal_capture["entropy_count"] > 0 and console_metrics.get("entropy") is None:
                    e = self.internal_capture["entropy_sum"] / float(self.internal_capture["entropy_count"])
                    self._safe_add_scalar("Policy/entropy", e, step)
                    console_metrics["entropy"] = e

                storage_stats = self._collect_storage_stats()
                for tag, val in storage_stats.items():
                    self._safe_add_scalar(tag, val, step)
                console_metrics["adv_mean"] = storage_stats.get("Adv/mean")
                console_metrics["adv_std"] = storage_stats.get("Adv/std")
                console_metrics["adv_max"] = storage_stats.get("Adv/max_abs")
                console_metrics["ev"] = storage_stats.get("Critic/explained_variance")

                action_std_mean = self._collect_action_std()
                if action_std_mean is not None:
                    self._safe_add_scalar("Policy/action_std_mean", action_std_mean, step)
                console_metrics["act_std"] = action_std_mean

                opt_stats = self._collect_opt_stats()
                for tag, val in opt_stats.items():
                    self._safe_add_scalar(tag, val, step)
                console_metrics["lr"] = opt_stats.get("Opt/lr", console_metrics.get("lr"))
                console_metrics["grad"] = opt_stats.get("Opt/grad_norm", console_metrics.get("grad"))
                console_metrics["early_stop"] = opt_stats.get("PPO/early_stop_count")
                console_metrics["skipped"] = opt_stats.get("PPO/update_skipped")

                if torch.cuda.is_available() and (step % self.cuda_mem_log_every == 0):
                    cuda_stats = self._collect_cuda_mem_stats()
                    for tag, val in cuda_stats.items():
                        self._safe_add_scalar(tag, val, step)
                    console_metrics["cuda_alloc_gb"] = cuda_stats.get("CUDA/memory_allocated_gb")
                    console_metrics["cuda_reserved_gb"] = cuda_stats.get("CUDA/memory_reserved_gb")
                    console_metrics["cuda_max_alloc_gb"] = cuda_stats.get("CUDA/max_memory_allocated_gb")
                    console_metrics["cuda_alloc_res"] = cuda_stats.get("CUDA/alloc_reserved_ratio")

                for k in list(console_metrics.keys()):
                    console_metrics[k] = self._to_float(console_metrics.get(k))

                print(
                    "[ppo-extra] "
                    f"it={step} "
                    f"kl={self._fmt_metric(console_metrics.get('kl'))} "
                    f"clip={self._fmt_metric(console_metrics.get('clip_frac'))} "
                    f"ratio={self._fmt_metric(console_metrics.get('ratio'))} "
                    f"sur={self._fmt_metric(console_metrics.get('surrogate'))} "
                    f"val={self._fmt_metric(console_metrics.get('value'))} "
                    f"ev={self._fmt_metric(console_metrics.get('ev'))} "
                    f"adv_m={self._fmt_metric(console_metrics.get('adv_mean'))} "
                    f"adv_s={self._fmt_metric(console_metrics.get('adv_std'))} "
                    f"adv_max={self._fmt_metric(console_metrics.get('adv_max'))} "
                    f"ent={self._fmt_metric(console_metrics.get('entropy'))} "
                    f"act_std={self._fmt_metric(console_metrics.get('act_std'))} "
                    f"grad={self._fmt_metric(console_metrics.get('grad'))} "
                    f"lr={self._fmt_metric(console_metrics.get('lr'))} "
                    f"early_stop={self._fmt_metric(console_metrics.get('early_stop'))} "
                    f"skipped={self._fmt_metric(console_metrics.get('skipped'))} "
                    f"cuda_alloc_gb={self._fmt_metric(console_metrics.get('cuda_alloc_gb'))} "
                    f"cuda_reserved_gb={self._fmt_metric(console_metrics.get('cuda_reserved_gb'))} "
                    f"cuda_max_alloc_gb={self._fmt_metric(console_metrics.get('cuda_max_alloc_gb'))} "
                    f"cuda_alloc_res={self._fmt_metric(console_metrics.get('cuda_alloc_res'))}"
                )
            except Exception:
                pass
            finally:
                self.update_step["i"] = step + 1

            return result

        try:
            self.alg.update = _wrapped_update
        except Exception:
            pass
