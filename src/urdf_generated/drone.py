import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import math
import re
from typing import Dict, Iterable, List, Tuple, Optional, Sequence
import csv
############################################################
# ENUM: aerodynamic surface type
############################################################
class SurfaceKind(str, Enum):
    FUSELAGE = "fuselage"
    WING = "wing"
    ELEVATOR = "elevator"
    RUDDER = "rudder"
    PROPELLER = "propeller"


############################################################
# ONE AERODYNAMIC SURFACE
############################################################
@dataclass
class AeroSurface:
    frame_name: str
    kind: SurfaceKind
    S: float
    AR: float
    chord: float
    span: float
    params: dict


############################################################
# ACTUATOR INFORMATION
############################################################
@dataclass
class ActuatorInfo:
    frame_name: str
    joint_name: str | None
    actuator: str | None         # e.g. "X08", "P08"; primary actuator
    limits: tuple                # (lower, upper)
    yaw_actuator: str | None = None
    pitch_actuator: str | None = None


############################################################
# DRONE MODEL
############################################################
class DroneModel:
    REQUIRED_GLOBAL_KEYS = ("rho", "force_cap")
    REQUIRED_TYPE_KEYS = {
        SurfaceKind.FUSELAGE: ("cd0", "k_slip_fus", "cp_start", "cp_end", "cg_to_chord"),
        SurfaceKind.WING: (
            "cl_alpha_2d",
            "alpha0_2d",
            "cd0",
            "alpha_stall_deg",
            "m_smooth",
            "cp_start",
            "cp_end",
            "cg_to_chord",
            "k_slip_wing",
        ),
        SurfaceKind.ELEVATOR: (
            "cl_alpha_2d",
            "alpha0_2d",
            "cd0",
            "alpha_stall_deg",
            "m_smooth",
            "cp_start",
            "cp_end",
            "cg_to_chord",
            "k_slip_tail",
            "k_eps_tail",
        ),
        SurfaceKind.RUDDER: (
            "cl_alpha_2d",
            "alpha0_2d",
            "cd0",
            "alpha_stall_deg",
            "m_smooth",
            "cp_start",
            "cp_end",
            "cg_to_chord",
        ),
        SurfaceKind.PROPELLER: (),
    }

    def __init__(self, urdf_path: str, config_override: dict | None = None):
        self.urdf_path = Path(urdf_path)
        self._config_override = copy.deepcopy(config_override) if config_override is not None else None

        self.surfaces: list[AeroSurface] = []
        self.actuators: dict[str, ActuatorInfo] = {}

        self.global_params = {}
        self.noise_params = {}
        self._debug_dims = {}

        self._load_all()

    def _require_keys(self, section: str, mapping: dict, required: Iterable[str]):
        missing = [k for k in required if k not in mapping]
        if missing:
            raise ValueError(f"Missing keys {missing} in aero configuration section '{section}'")


    ############################################################
    # MASTER LOAD FUNCTION
    ############################################################
    def _load_all(self):
        root = ET.parse(self.urdf_path).getroot()
        cfg = self._load_yaml()

        if not isinstance(cfg, dict):
            raise ValueError("aero configuration must define a mapping at the top level.")

        global_cfg = cfg.get("global")
        type_cfg = cfg.get("types")
        link_cfg = cfg.get("links")
        if not isinstance(global_cfg, dict) or not global_cfg:
            raise ValueError("aero configuration must define a non-empty 'global' section.")
        if not isinstance(type_cfg, dict) or not type_cfg:
            raise ValueError("aero configuration must define a non-empty 'types' section.")
        if not isinstance(link_cfg, dict) or not link_cfg:
            raise ValueError("aero configuration must define a non-empty 'links' section.")

        self._require_keys("global", global_cfg, self.REQUIRED_GLOBAL_KEYS)

        self.global_params = global_cfg
        self.noise_params = cfg.get("noise", {})

        aero_frames = self._discover_aero_frames(root, link_cfg)

        # Build aerodynamic surfaces
        for frame in aero_frames:
            kind = self._infer_kind(frame, link_cfg)
            S, AR, chord, span = self._compute_geom(root, frame, kind)
            params = self._resolve_params(frame, kind, type_cfg, link_cfg)

            # --- Seff ratios (relative to S0) ---
            def _as_ratio(x, default=1.0):
                if x is None:
                    return float(default)
                if isinstance(x, str) and x.strip().lower() in ("none", "null", "~", ""):
                    return float(default)
                r = float(x)
                return 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)

            # Effective-area ratios at maximum deflection of the associated joint.
            # Only ratios explicitly provided in `links:` should affect the solver.
            params["s_folded"] = _as_ratio(params.get("s_folded", 1.0), 1.0)
            params["s_folded_yaw"] = _as_ratio(params.get("s_folded_yaw", 1.0), 1.0)
            params["s_folded_pitch"] = _as_ratio(params.get("s_folded_pitch", 1.0), 1.0)

            self.surfaces.append(AeroSurface(frame, kind, S, AR, chord, span, params))

        self._wingspan = self._wingspan_from_surfaces(self.surfaces)
        prop_frames = [s.frame_name for s in self.surfaces if s.kind == SurfaceKind.PROPELLER]
        if not prop_frames:
            raise ValueError("No propeller surface defined in aero configuration/URDF.")
        prop_radius = self._extract_prop_radius(root, prop_frames[0])
        slip_factor = prop_radius / self._wingspan
        for surf in self.surfaces:
            if surf.kind == SurfaceKind.WING and "k_slip_wing" in surf.params:
                surf.params["k_slip_wing"] *= slip_factor

        # Build actuator data
        self._parse_actuators(root, link_cfg)
        catalog = self._load_actuator_catalog()
        self._validate_actuator_catalog_usage(catalog)
        self._apply_actuator_catalog(catalog)
        self._validate_prop_parameters()
        # Optional debug data
        self._compute_debug_dims(root)
        for s in self.surfaces:
            if s.kind == SurfaceKind.PROPELLER:
                print("[PROP PARAMS]", s.frame_name, s.params.get("max_thrust"), s.params.get("prop_cutoff_hz"), s.params.get("kappa_prop"))



    ############################################################
    # LOAD AERO CONFIG
    ############################################################
    def _load_yaml(self):
        if self._config_override is None:
            raise ValueError("Aero configuration override is required for DroneModel.")
        return copy.deepcopy(self._config_override)

    def _load_actuator_catalog(self) -> dict[str, dict]:
        """
        Load actuators.csv placed next to the URDF (same folder).
        Expected columns like:
        name,type,mass,max_thrust,kappa_prop,prop_cutoff_hz,R,kV,kI,prop_voltage_nominal,prop_ct0,prop_ct1,prop_ct2,kp,kv
        """
        csv_path = self.urdf_path.parent / "actuators.csv"
        if not csv_path.exists():
            return {}

        catalog: dict[str, dict] = {}
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("name") or "").strip()
                if not name:
                    continue

                def fkey(k: str):
                    v = (row.get(k) or "").strip()
                    return float(v) if v != "" else None

                catalog[name] = {
                    "type": (row.get("type") or "").strip().lower(),
                    "mass": fkey("mass"),
                    "max_thrust": fkey("max_thrust"),
                    "kappa_prop": fkey("kappa_prop"),
                    "prop_cutoff_hz": fkey("prop_cutoff_hz"),
                    "R": fkey("R"),
                    "kV": fkey("kV"),
                    "kI": fkey("kI"),
                    "prop_voltage_nominal": fkey("prop_voltage_nominal"),
                    "prop_ct0": fkey("prop_ct0"),
                    "prop_ct1": fkey("prop_ct1"),
                    "prop_ct2": fkey("prop_ct2"),
                    "kp": fkey("kp"),
                    "kv": fkey("kv"),
                }
        return catalog

    def _validate_actuator_catalog_usage(self, catalog: dict[str, dict]) -> None:
        """
        Ensure all actuator names referenced in aero configuration exist in actuators.csv
        and contain the required fields for their type.
        """
        if not self.actuators:
            return
        if not catalog:
            raise ValueError("actuators.csv is missing/empty but aero configuration references actuators.")

        def require(act_name: str, expected_type: str, required_fields: tuple[str, ...], frame: str) -> None:
            row = catalog.get(act_name)
            if row is None:
                raise ValueError(f"{frame}: actuator '{act_name}' not found in actuators.csv.")
            if (row.get("type") or "").strip().lower() != expected_type:
                raise ValueError(
                    f"{frame}: actuator '{act_name}' has type '{row.get('type')}', expected '{expected_type}'."
                )
            missing = [k for k in required_fields if row.get(k) is None]
            if missing:
                raise ValueError(f"{frame}: actuator '{act_name}' missing fields {missing} in actuators.csv.")

        for frame, info in self.actuators.items():
            for act_name in (getattr(info, "yaw_actuator", None), getattr(info, "pitch_actuator", None), getattr(info, "actuator", None)):
                if not act_name:
                    continue
                # Propeller link uses `actuator` and expects propeller fields.
                if "prop" in frame.lower():
                    require(act_name, "propeller", ("max_thrust", "kappa_prop", "prop_cutoff_hz"), frame)
                else:
                    require(act_name, "servo", ("kp", "kv"), frame)


    def _apply_actuator_catalog(self, catalog: dict[str, dict]) -> None:
        """
        Apply actuator catalog rows to AeroSurface params based on the actuator name
        assigned in aero configuration (links: <frame>: actuator: P08).
        """
        if not catalog:
            return

        # frame_name -> actuator_name mapping (already parsed from aero config in _parse_actuators)
        frame_to_act = {}
        for frame, info in self.actuators.items():
            act = getattr(info, "actuator", None)
            if act:
                frame_to_act[frame] = act

        # Update surface params
        for surf in self.surfaces:
            if surf.kind != SurfaceKind.PROPELLER:
                continue

            act_name = frame_to_act.get(surf.frame_name)
            if not act_name:
                continue

            row = catalog.get(act_name)
            if not row:
                continue

            # optional safety: ensure this actuator row is a propeller
            if row.get("type") and row["type"] != "propeller":
                continue

            # Write into this prop surface params (what AeroSolver later reads)
            if row.get("max_thrust") is not None:
                surf.params["max_thrust"] = float(row["max_thrust"])
            if row.get("kappa_prop") is not None:
                surf.params["kappa_prop"] = float(row["kappa_prop"])
            if row.get("prop_cutoff_hz") is not None:
                surf.params["prop_cutoff_hz"] = float(row["prop_cutoff_hz"])
            for key in ("kV", "prop_voltage_nominal", "prop_ct0", "prop_ct1", "prop_ct2"):
                if row.get(key) is not None:
                    surf.params[key] = float(row[key])

            # (opzionale) massa prop se vuoi usarla altrove
            if row.get("mass") is not None:
                surf.params["mass_actuator"] = float(row["mass"])

    def _validate_prop_parameters(self) -> None:
        """Ensure propeller surfaces received mandatory parameters."""
        required = ("max_thrust", "kappa_prop", "prop_cutoff_hz")
        for surf in self.surfaces:
            if surf.kind != SurfaceKind.PROPELLER:
                continue
            missing = [k for k in required if k not in surf.params]
            if missing:
                raise ValueError(
                    f"Propeller frame '{surf.frame_name}' missing parameters {missing}. "
                    "Provide them via aero configuration or actuators.csv."
                )
            return
        raise ValueError("No propeller surface defined in aero configuration/URDF.")

    def _norm_none(self, v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in ("none", "null", "~", ""):
            return None
        return v

    ############################################################
    # FIND AERO FRAMES (from aero config or URDF)
    ############################################################
    def _discover_aero_frames(self, root, link_cfg):
        if link_cfg:
            return list(link_cfg.keys())

        frames = []
        for link in root.findall(".//link"):
            name = link.get("name", "")
            if name.startswith("aero_frame_") or name.startswith("prop_frame_"):
                frames.append(name)

        return sorted(frames)


    ############################################################
    # DETERMINE SURFACE KIND
    ############################################################
    def _infer_kind(self, frame, link_cfg):
        if frame in link_cfg and "type" in link_cfg[frame]:
            return SurfaceKind(link_cfg[frame]["type"])

        name = frame.lower()

        if "fuselage" in name:
            return SurfaceKind.FUSELAGE
        if "wing" in name:
            return SurfaceKind.WING
        if "elevator" in name:
            return SurfaceKind.ELEVATOR
        if "rudder" in name:
            return SurfaceKind.RUDDER
        if "prop" in name:
            return SurfaceKind.PROPELLER

        raise ValueError(f"Cannot infer aerodynamic type for frame: {frame}")


    ############################################################
    # GEOMETRY EXTRACTION (from URDF boxes)
    ############################################################
    def _compute_geom(self, root, frame, kind):
        link = root.find(f".//link[@name='{frame}']")
        if link is None:
            raise ValueError(f"URDF missing link for aerodynamic frame '{frame}'.")

        box = link.find("./collision/geometry/box")
        if box is None:
            box = self._find_parent_collision_box(root, frame)

        sx, sy, sz = map(float, box.get("size").split())

        chord = sy
        span = sz
        area = chord * span

        if kind == SurfaceKind.WING or kind == SurfaceKind.ELEVATOR:
            AR = (2 * span + 0.1) / chord if area > 0 else 1.0
        else:
            AR = span / chord if area > 0 else 1.0

        if kind == SurfaceKind.FUSELAGE:
            area = math.pi * 0.25 * sx * sy
        print(f"Computed geom for {frame}: S={area}, chord={chord}, AR={AR}, span={span}")

        return area, AR, chord, span

    def _find_parent_collision_box(self, root, child_link: str, max_hops: int = 16):
        """
        Resolve geometry for an aero frame that is a massless link without collision.

        We traverse the kinematic tree upwards (child -> parent joints) until we
        find a link with a collision box.
        """

        def parent_of(child: str) -> str | None:
            for joint in root.findall(".//joint"):
                ch = joint.find("child")
                if ch is None or ch.get("link") != child:
                    continue
                par = joint.find("parent")
                if par is None:
                    continue
                return par.get("link")
            return None

        cur = child_link
        for _ in range(max_hops):
            par = parent_of(cur)
            if not par:
                break
            par_link = root.find(f".//link[@name='{par}']")
            if par_link is None:
                break
            box = par_link.find("./collision/geometry/box")
            if box is not None:
                return box
            cur = par

        raise ValueError(
            f"Aerodynamic frame '{child_link}' has no collision box and no ancestor link with a collision box."
        )


    ############################################################
    # PARAMETER MERGING: global → type → link
    ############################################################
    def _resolve_params(self, frame, kind, type_cfg, link_cfg):
        p = {}

        p.update(self.global_params)
        type_key = kind.value
        type_params = type_cfg.get(type_key)
        if type_params is None and kind != SurfaceKind.PROPELLER:
            raise ValueError(f"aero configuration missing type configuration for '{type_key}'.")
        if type_params is not None:
            self._require_keys(f"types.{type_key}", type_params, self.REQUIRED_TYPE_KEYS.get(kind, ()))
            p.update(type_params)

        if frame in link_cfg:
            for k, v in link_cfg[frame].items():
                if k not in ("type", "actuator", "actuator_yaw", "actuator_pitch"):
                    p[k] = v

        if "re_nom" in p:
            p["re_nom"] = float(p["re_nom"])
        if "re_a" in p:
            p["re_a"] = float(p["re_a"])

        return p


    ############################################################
    # ACTUATOR PARSING  (NEW AND FINAL)
    ############################################################
    def _parse_actuators(self, root, link_cfg):
        for surf in self.surfaces:
            frame = surf.frame_name

            if frame not in link_cfg:
                continue

            actuator = self._norm_none(link_cfg[frame].get("actuator", None))
            actuator_yaw = self._norm_none(link_cfg[frame].get("actuator_yaw", None))
            actuator_pitch = self._norm_none(link_cfg[frame].get("actuator_pitch", None))

            actuator_main = actuator or actuator_yaw or actuator_pitch
            if actuator_main is None:
                continue

            joint_name, limits = self._find_joint_for_frame(root, frame)

            self.actuators[frame] = ActuatorInfo(
                frame_name=frame,
                joint_name=joint_name,
                actuator=actuator_main,     # NEW — directly from aero config
                limits=limits,
                yaw_actuator=actuator_yaw,
                pitch_actuator=actuator_pitch,
            )


    def _find_joint_for_frame(self, root, frame):
        for joint in root.findall(".//joint"):
            ch = joint.find("child")
            if ch is None:
                continue

            if ch.get("link") == frame:
                lim = joint.find("limit")
                if lim is None:
                    return joint.get("name"), (-math.inf, math.inf)

                low = float(lim.get("lower", -math.inf))
                up  = float(lim.get("upper",  math.inf))
                return joint.get("name"), (low, up)

        return None, (0.0, 0.0)

    def _find_joint_by_name(self, root, joint_name: str):
        for joint in root.findall(".//joint"):
            if joint.get("name") == joint_name:
                return joint
        return None

    def _theta_max_from_joint_limit(self, joint) -> float:
        lim = joint.find("limit")
        if lim is None:
            raise ValueError(f"URDF joint '{joint.get('name')}' missing <limit> tag.")
        lower = lim.get("lower")
        upper = lim.get("upper")
        if lower is None or upper is None:
            raise ValueError(f"URDF joint '{joint.get('name')}' must define lower and upper in <limit>.")
        lo = float(lower)
        hi = float(upper)
        th = max(abs(lo), abs(hi))
        if not math.isfinite(th) or th <= 0.0:
            raise ValueError(f"URDF joint '{joint.get('name')}' has invalid theta_max from limits: ({lo},{hi}).")
        return float(th)

    ############################################################
    # WINGSPAN & PROP RADIUS FOR SLIPSTREAM
    ############################################################
    def _wingspan_from_surfaces(self, surfaces: Iterable[AeroSurface]) -> float:
        wings = [s for s in surfaces if s.kind == SurfaceKind.WING]
        if not wings:
            raise ValueError("Unable to compute wingspan: no wing surfaces defined.")

        left = [abs(s.span) for s in wings if "left" in s.frame_name.lower()]
        right = [abs(s.span) for s in wings if "right" in s.frame_name.lower()]

        if left and right:
            return max(left) + max(right)

        return 2.0 * max(abs(s.span) for s in wings)

    @property
    def wingspan(self) -> float:
        if not hasattr(self, "_wingspan") or self._wingspan is None:
            raise ValueError("Wingspan not computed yet.")
        return float(self._wingspan)


    def _extract_prop_radius(self, root, prop_frame: str) -> float:
        """
        Extract propeller radius from the propeller frame geometry.

        Convention: thrust axis is the prop frame +Z; radius is inferred from the prop
        collision/visual geometry.
        """
        link = root.find(f".//link[@name='{prop_frame}']")
        if link is None:
            raise ValueError(f"Unable to extract prop radius: URDF missing link '{prop_frame}'.")

        cyl = link.find("./collision/geometry/cylinder") or link.find("./visual/geometry/cylinder")
        if cyl is not None and cyl.get("radius") is not None:
            return float(cyl.get("radius"))

        box = link.find("./collision/geometry/box")
        if box is not None and box.get("size") is not None:
            sx, sy, _ = map(float, box.get("size").split())
            return 0.5 * max(sx, sy)

        raise ValueError(
            f"Unable to extract prop radius: link '{prop_frame}' must define a cylinder radius or a collision box."
        )


    ############################################################
    # DEBUG DATA (OPTIONAL)
    ############################################################
    def _compute_debug_dims(self, root):
        self._debug_dims["wing_chord"] = 1.0
        self._debug_dims["tip_to_tip"] = 1.0


    ############################################################
    # PUBLIC PROPERTIES (used by AeroSolver)
    ############################################################
    @property
    def frames(self):
        return [s.frame_name for s in self.surfaces]

    @property
    def geom(self):
        mapping = {
            SurfaceKind.FUSELAGE:   0,
            SurfaceKind.WING:       1,
            SurfaceKind.ELEVATOR:   2,
            SurfaceKind.RUDDER:     3,
            SurfaceKind.PROPELLER:  4,
        }
        return [(s.S, s.AR, s.chord, mapping[s.kind]) for s in self.surfaces]

    @property
    def base_params(self):
        return dict(self.global_params)

    @property
    def debug_dims(self):
        return dict(self._debug_dims)


############################################################
# AERO MODEL ADAPTER (for AeroSolver)
############################################################
class DroneAeroModel:
    """
    Adapter that exposes aerodynamic metadata consumed by `AeroSolver`.

    The class parses the URDF + provided aero configuration and
    builds:
    - ordered aerodynamic frames,
    - per-surface geometry tuples (S, AR, chord, kind code),
    - consolidated aerodynamic parameters for Taichi fields,
    - basic geometric scalars (span, aspect ratios, prop radius).

    All aerodynamic calculations stay identical inside `AeroSolver`; this
    adapter only maps configuration data into the expected attributes.
    """

    # Default aerodynamic frames (empty: filled from aero config/URDF at runtime).
    AERO_FRAMES: List[str] = []

    # Minimal defaults; real values come from aero configuration.
    DEFAULT_BASE_PARAMS: Dict[str, float] = {}

    NOISE_DEFAULTS: Dict[str, float] = {"sigma_mag": 0.0, "sigma_dir": 0.0, "sigma_param": 0.0, "sigma_cp": 0.0}

    def __init__(self, urdf_path: str, config_override: dict | None = None):
        self.urdf_path = str(urdf_path)
        self._model = DroneModel(urdf_path, config_override=config_override)
        self._urdf_root = ET.parse(self._model.urdf_path).getroot()
        self._surfaces = self._sorted_surfaces(self._model.surfaces)
        self.surface_kinds: List[SurfaceKind] = [s.kind for s in self._surfaces]

        # Public fields consumed directly by AeroSolver
        self.frames: List[str] = [s.frame_name for s in self._surfaces]
        self.geom: List[Tuple[float, float, float, int]] = self._build_geom()
        self.base_params, self.base_param_overrides, self.per_surface_params = self._merge_base_params()

        # --- Prop “authoritative” params from per-surface (includes actuators.csv) ---
        for surf in self._surfaces:
            if surf.kind == SurfaceKind.PROPELLER:
                for k in ("max_thrust", "kappa_prop", "prop_cutoff_hz", "kV", "prop_voltage_nominal", "prop_ct0", "prop_ct1", "prop_ct2"):
                    if k in surf.params:
                        self.base_params[k] = float(surf.params[k])
                break

        self.noise_params: Dict[str, float] = dict(self._model.noise_params)
        self.actuators: Dict[str, ActuatorInfo] = dict(self._model.actuators)

        # Geometry scalars (safe defaults if the URDF lacks a given surface)
        self.tip_to_tip: float = self._model.wingspan
        self.AR_wing: float = self._aspect_ratio_for_kind(SurfaceKind.WING)
        self.AR_tail: float = self._aspect_ratio_for_kind(SurfaceKind.ELEVATOR)
        self.S_perp_fus: float = self._fuselage_area()
        self.cg_fus_local_x: float = 0.0
        self.cz_rudder: float = 0.0
        self.prop_radius: float = self._extract_prop_radius()

        # Optional genome embedded in URDF filename "[...].urdf"
        self.genes, self.genes_dict = self._parse_genes_from_name(self.urdf_path)

        # Optional debug numbers
        self.debug_dims = dict(self._model.debug_dims)

    # ------------------------------------------------------------------ #
    # Public helpers                                                     #
    # ------------------------------------------------------------------ #
    def link_indices(self, entity) -> List[int]:
        """Map aerodynamic frame names to link indices on the given entity."""
        return [entity.get_link(name).idx for name in self.frames]

    def required_links(self, servo_joint_names: Sequence[str] | None = None) -> List[str]:
        """
        Compute a minimal-but-safe set of links to keep for the URDF loader.

        Includes:
          - all aerodynamic frames (from aero configuration),
          - the full parent chains for those frames,
          - parent/child links for the requested servo joints and their chains,
          - a fallback 'fuselage' link.
        """
        frames = list(self.frames)
        extra_links: List[str] = []

        root = self._urdf_root or ET.parse(self._model.urdf_path).getroot()
        child_to_parent: Dict[str, str] = {}
        joint_map: Dict[str, Tuple[str, str]] = {}
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            parent_link = parent.get("link")
            child_link = child.get("link")
            if not parent_link or not child_link:
                continue
            child_to_parent[child_link] = parent_link
            name = joint.get("name")
            if name:
                joint_map[name] = (parent_link, child_link)

        chain_seen: set[str] = set()

        def add_chain(link: str) -> None:
            while link and link not in chain_seen:
                chain_seen.add(link)
                extra_links.append(link)
                link = child_to_parent.get(link, "")

        for frame in frames:
            add_chain(frame)
        for jname in servo_joint_names or ():
            pair = joint_map.get(jname)
            if pair:
                add_chain(pair[0])
                add_chain(pair[1])

        return list(dict.fromkeys(frames + extra_links + ["fuselage"]))

    def validate_entity(
        self,
        entity,
        servo_joint_names: Sequence[str] | None = None,
        servo_dof_indices: Sequence[int] | None = None,
    ) -> None:
        """
        Validate consistency between link/joint references and indices.
        """
        missing_links: List[str] = []
        bad_link_indices: List[Tuple[str, int]] = []
        n_links = getattr(entity, "n_links", None)
        for name in self.frames:
            try:
                link = entity.get_link(name)
            except Exception:
                missing_links.append(name)
                continue
            if n_links is not None:
                idx_local = link.idx_local
                if idx_local < 0 or idx_local >= n_links:
                    bad_link_indices.append((name, int(idx_local)))

        if missing_links:
            raise ValueError(f"Missing links for aero frames: {missing_links}")
        if bad_link_indices:
            raise ValueError(f"Out-of-range link indices: {bad_link_indices}")

        if not servo_joint_names:
            return

        missing_joints: List[str] = []
        empty_dof_joints: List[str] = []
        dof_indices: List[int] = []
        mismatched_dof_indices: List[Tuple[str, Optional[int], int]] = []
        n_dofs = getattr(entity, "n_dofs", None)

        for i, name in enumerate(servo_joint_names):
            try:
                joint = entity.get_joint(name)
            except Exception:
                missing_joints.append(name)
                continue

            idxs = getattr(joint, "dofs_idx_local", None)
            if not idxs:
                empty_dof_joints.append(name)
                continue
            if isinstance(idxs, (list, tuple)):
                idx_list = [int(v) for v in idxs]
            else:
                idx_list = [int(idxs)]
            dof_indices.extend(idx_list)

            if servo_dof_indices is not None:
                expected = idx_list[0]
                actual = servo_dof_indices[i] if i < len(servo_dof_indices) else None
                if actual is None or int(actual) != expected:
                    mismatched_dof_indices.append((name, None if actual is None else int(actual), expected))

        if missing_joints:
            raise ValueError(f"Missing servo joints: {missing_joints}")
        if empty_dof_joints:
            raise ValueError(f"Servo joints without DOF indices: {empty_dof_joints}")

        if n_dofs is not None:
            bad_dof = [idx for idx in dof_indices if idx < 0 or idx >= n_dofs]
            if bad_dof:
                raise ValueError(f"Out-of-range DOF indices: {bad_dof}")
        if len(set(dof_indices)) != len(dof_indices):
            raise ValueError(f"Duplicate DOF indices: {dof_indices}")
        if mismatched_dof_indices:
            raise ValueError(f"Servo DOF index mismatch: {mismatched_dof_indices}")

    # ------------------------------------------------------------------ #
    # Internal utilities                                                 #
    # ------------------------------------------------------------------ #
    def _sorted_surfaces(self, surfaces: Iterable[AeroSurface]) -> List[AeroSurface]:
        """Stable sort that keeps the propeller last (solver expects it)."""
        order = {
            SurfaceKind.FUSELAGE: 0,
            SurfaceKind.WING: 1,
            SurfaceKind.ELEVATOR: 2,
            SurfaceKind.RUDDER: 3,
            SurfaceKind.PROPELLER: 4,
        }
        return sorted(list(surfaces), key=lambda s: order.get(s.kind, 5))

    def _build_geom(self) -> List[Tuple[float, float, float, int]]:
        mapping = {
            SurfaceKind.FUSELAGE: 0,
            SurfaceKind.WING: 1,
            SurfaceKind.ELEVATOR: 2,
            SurfaceKind.RUDDER: 3,
            SurfaceKind.PROPELLER: 4,
        }
        return [(s.S, s.AR, s.chord, mapping[s.kind]) for s in self._surfaces]

    def _wing_side_keys(self, name: str) -> Tuple[str, str]:
        """Return canonical (left, right) keys for a wing parameter name."""
        suffix = "_wing"
        if name.endswith(suffix):
            base = name[: -len(suffix)]
        else:
            base = name
        return (f"{base}_wing_left", f"{base}_wing_right")

    def _elevator_side_keys(self, name: str) -> Tuple[str, str]:
        suffix = "_elevator"
        if name.endswith(suffix):
            base = name[: -len(suffix)]
        else:
            base = name
        return (f"{base}_elevator_left", f"{base}_elevator_right")

    def _merge_base_params(self) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Merge aero configuration into aero parameters and per-link overrides."""
        cfg = self._model._load_yaml()
        cfg_global = cfg.get("global") or {}
        cfg_types = cfg.get("types") or {}

        def _ensure(section: str, mapping: Dict[str, float], keys: Iterable[str]) -> None:
            missing = [k for k in keys if k not in mapping]
            if missing:
                raise ValueError(f"Missing keys {missing} in aero configuration section '{section}'")

        _ensure("global", cfg_global, DroneModel.REQUIRED_GLOBAL_KEYS)

        def type_params(kind: SurfaceKind) -> Dict[str, float]:
            params = cfg_types.get(kind.value)
            if params is None:
                raise ValueError(f"aero configuration missing type configuration for '{kind.value}'.")
            _ensure(f"types.{kind.value}", params, DroneModel.REQUIRED_TYPE_KEYS.get(kind, ()))
            return dict(params)

        fus = type_params(SurfaceKind.FUSELAGE)
        wing = type_params(SurfaceKind.WING)
        elev = type_params(SurfaceKind.ELEVATOR)
        prop = dict(cfg_types.get(SurfaceKind.PROPELLER.value, {}))

        root = ET.parse(self._model.urdf_path).getroot()
        prop_frame = next((s.frame_name for s in self._surfaces if s.kind == SurfaceKind.PROPELLER), None)
        if prop_frame is None:
            raise ValueError("No propeller surface defined in aero configuration/URDF.")
        slip_factor = self._model._extract_prop_radius(root, prop_frame) / self._model.wingspan
        wing_scaled = dict(wing)
        wing_scaled["k_slip_wing"] = float(wing["k_slip_wing"]) * float(slip_factor)

        p: Dict[str, float] = {}
        p.update({k: float(v) for k, v in cfg_global.items()})
        p.update({k: float(v) for k, v in wing_scaled.items()})

        p["k_slip_fus"] = float(fus["k_slip_fus"])
        p["k_slip_tail"] = float(elev["k_slip_tail"])
        p["k_eps_tail"] = float(elev["k_eps_tail"])

        if prop:
            if "kappa_prop" in prop:
                p["kappa_prop"] = float(prop["kappa_prop"])
            if "prop_cutoff_hz" in prop:
                p["prop_cutoff_hz"] = float(prop["prop_cutoff_hz"])
            if "max_thrust" in prop:
                p["max_thrust"] = float(prop["max_thrust"])
            for key in ("kV", "prop_voltage_nominal", "prop_ct0", "prop_ct1", "prop_ct2"):
                if key in prop:
                    p[key] = float(prop[key])

        for key in ("rho", "force_cap"):
            if key not in p:
                raise ValueError(f"Missing global aerodynamic constant '{key}' in aero configuration.")

        wing_keys = [
            "cl_alpha_2d",
            "alpha0_2d",
            "cd0",
            "alpha_stall_deg",
            "m_smooth",
            "cp_start",
            "cp_end",
            "cg_to_chord",
            "k_slip_wing",
        ]
        elevator_keys = [
            "cl_alpha_2d",
            "alpha0_2d",
            "cd0",
            "alpha_stall_deg",
            "m_smooth",
            "cp_start",
            "cp_end",
            "cg_to_chord",
            "k_slip_tail",
            "k_eps_tail",
        ]

        def duplicate_from(values: Dict[str, float], keys: Iterable[str], suffix: str):
            for key in keys:
                if key not in values:
                    raise ValueError(f"Missing '{key}' while duplicating {suffix} parameters.")
                base_val = float(values[key])
                if suffix == "wing":
                    l_key, r_key = self._wing_side_keys(key)
                elif suffix == "elevator":
                    l_key, r_key = self._elevator_side_keys(key)
                else:
                    l_key = f"{key}_{suffix}_left"
                    r_key = f"{key}_{suffix}_right"
                p[l_key] = base_val
                p[r_key] = base_val

        duplicate_from(wing_scaled, wing_keys, "wing")
        duplicate_from(elev, elevator_keys, "elevator")

        per_surface_params: List[Dict[str, float]] = [dict(s.params) for s in self._surfaces]

        # Base param overrides: make per-surface resolved params drive the actual solver fields
        # used by `_wing_param` / `_elevator_param` (so link overrides in aero config take effect).
        overrides: Dict[str, float] = {}
        for surf, prm in zip(self._surfaces, per_surface_params):
            name_l = surf.frame_name.lower()
            if surf.kind == SurfaceKind.WING:
                if "left" in name_l:
                    for key in wing_keys:
                        if key in prm:
                            l_key, _ = self._wing_side_keys(key)
                            overrides[l_key] = float(prm[key])
                elif "right" in name_l:
                    for key in wing_keys:
                        if key in prm:
                            _, r_key = self._wing_side_keys(key)
                            overrides[r_key] = float(prm[key])
            elif surf.kind == SurfaceKind.ELEVATOR:
                if "left" in name_l:
                    for key in elevator_keys:
                        if key in prm:
                            l_key, _ = self._elevator_side_keys(key)
                            overrides[l_key] = float(prm[key])
                elif "right" in name_l:
                    for key in elevator_keys:
                        if key in prm:
                            _, r_key = self._elevator_side_keys(key)
                            overrides[r_key] = float(prm[key])

        return p, overrides, per_surface_params

    def _aspect_ratio_for_kind(self, kind: SurfaceKind) -> float:
        """Average aspect ratio for all surfaces of the given kind."""
        ars = [s.AR for s in self._model.surfaces if s.kind == kind]
        if len(ars) == 0:
            return 0.0
        return sum(ars) / len(ars)

    def _fuselage_area(self) -> float:
        """Return the first fuselage reference area, if present."""
        for s in self._model.surfaces:
            if s.kind == SurfaceKind.FUSELAGE:
                return s.S
        return 0.0

    def _extract_prop_radius(self) -> float:
        """Prop radius is inferred from the URDF prop frame geometry."""
        root = ET.parse(self._model.urdf_path).getroot()
        prop_frame = next((s.frame_name for s in self._surfaces if s.kind == SurfaceKind.PROPELLER), None)
        if prop_frame is None:
            raise ValueError("No propeller surface defined in aero configuration/URDF.")
        return self._model._extract_prop_radius(root, prop_frame)

    def _parse_genes_from_name(self, name: str) -> Tuple[List[float], Dict[str, float]]:
        """
        Optional genome parsing from filenames like `[g1,g2,...].urdf`.
        Returns a list and a keyed dict for convenience.
        """
        m = re.search(r"\[([^\]]+)\]\.urdf$", name)
        if not m:
            return [], {}
        try:
            vals = [float(x) for x in m.group(1).split(",")]
        except Exception:
            return [], {}

        keys = [f"gene_{i}" for i in range(len(vals))]
        return vals, dict(zip(keys, vals))
############################################################
