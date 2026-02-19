from __future__ import annotations

"""
Parametric URDF generator for fixed-wing UAVs – genome-driven
=============================================================

This module builds a drone URDF from a *structured genome* of **15 parameters**:

 0) wing_span                     [m]
 1) wing_aspect_ratio             = span / chord
 2) fus_length                    [m]
 3) cg_x_ratio                    = fus_cg_x / fus_length
 4) attach_x_ratio                = wing_attach_x / fus_length
 5) elevator_span                 [m]
 6) elevator_aspect_ratio         = span / chord
 7) rudder_span                   [m]
 8) rudder_aspect_ratio           = span / chord
 9) dihedral_deg                  [deg]
10) sweep_multiplier              [-]
11) twist_multiplier              [-]
12) naca_d1                       [-]  first digit (0..4)
13) naca_d2                       [-]  second digit (2..5)
14) naca_last2                    [-]  last two digits (08..22)

All *physical* dimensions required by the URDF (chords, x-positions) are
derived internally from the genome above.

The output URDF contains a single XML comment with the **raw genome** (no names),
so the geometry can be reconstructed or traced back later.
"""

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union, List, Dict, Optional
import csv


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry parameter container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeometryParams:
    """
    Fully resolved physical parameters (derived from the 15-gene genome).

    All lengths are expressed in meters, angles in radians where not
    otherwise indicated.
    """
    # Wing
    wing_span: float
    wing_chord: float
    wing_attach_x: float  # x-position where wings attach to fuselage (m)
    # Fuselage
    fus_cg_x: float       # fuselage CG position along x (m, body frame)
    fus_length: float
    # Tail – elevator
    elevator_span: float
    elevator_chord: float
    # Tail – rudder
    rudder_span: float
    rudder_chord: float

    # Global scaling & aero modifiers
    density_scale: float = 1.0      # scales all structural densities
    dihedral_deg: float = 0.0       # wing dihedral angle [deg]
    prop_radius: float = 0.10       # propeller radius [m]
    hinge_le_ratio: float = 0.25    # hinge line position as fraction of chord
    sweep_multi: float = 1.0        # sweep joint multiplier (range scaling)
    twist_multi: float = 1.0        # twist joint multiplier (range scaling)
    cl_alpha_2d: float = 2.0        # 2D lift curve slope (scaled internally)
    alpha0_2d: float = 0.0          # zero-lift AoA [rad]


# ─────────────────────────────────────────────────────────────────────────────
#  URDF generator
# ─────────────────────────────────────────────────────────────────────────────

class UrdfMaker:
    """
    Build a *drone_ea*-compatible URDF from a 15-gene genome or from
    fully specified GeometryParams.

    The public API is intentionally small:

      • UrdfMaker(genome_or_params, out_dir="urdf_generated")
      • create_urdf(filename=None) -> str
      • build_tree() -> xml.etree.ElementTree.ElementTree
    """

    # ------------------------------------------------------------------
    # Reference geometry (kept from legacy implementation)
    # These are used to compute mesh scaling from physical dimensions.
    # ------------------------------------------------------------------
    _REF = {
        "tw":   0.017962225,           # wing thickness box X
        "tc":   0.0135567616,          # control-surface thickness box X
        "tfus": 0.10000000000000005,   # fuselage thickness (for scale)
        "hfus": 0.10476803794561983,   # fuselage height    (for scale)
        "fus_len": 0.7273891148,
        "wing_chord": 0.199988144,
        "wing_span_ref": 0.70,
        "elev_chord": 0.139941293,
        "elev_span": 0.36,
        "rudd_chord": 0.119323096638,
        "rudd_span": 0.16357341166,
    }

    # Reference mesh scales for each STL
    _REF_SCALE = {
        "fuselage": (0.00097, 0.001, 0.001),
        "wing":     (0.2,    0.7,   0.2),
        "elevator": (0.0007, 0.0012, 0.001),
        "rudder":   (0.0008, 0.001,  0.001),
    }

    # ------------------------------------------------------------------
    # "Mass model" densities (kg/m³) and constants
    # ------------------------------------------------------------------
    _RHO_FUS_STRUCT = 20.0
    _FUS_SHELL_THICKNESS = 0.02  # [m] shell thickness
    _FUS_FIXED_MASS = 0.200              # e.g. battery, avionics, etc.
    _BATTERY_SIZE   = (0.10, 0.05, 0.05) # box for inertia (10×5×5 cm)

    _RHO_WING = 20.0
    _RHO_ELEV = 20.0
    _RHO_RUDD = 20.0
    _RHO_PROP = 500.0
    _PROP_CAMERA_MASS = 0.03
    _WING_FOAM_FILL = 0.68

    # Lever arms as fraction of chord (from LE reference frame)
    _CG_RATIO   = 0.31
    _COLL_RATIO = 0.25
    _AERO_RATIO = 0.25

    # Inertia fudge factor + collision shrink
    _I_FUDGE = 0.6
    _SHRINK  = 1.0

    # Root offsets and fixed RPYs (legacy values kept)
    _MASS_INTER     = 0.005
    _ROOT_Y_OFFSET  = 0.05

    _RPY_FUSE_COLL = "-1.5898372930676048 6.123233995736766e-17 -1.5707963267948968"
    _RPY_WING_COLL = "1.570796327 -1.558922022 0"
    _RPY_ELEV_COLL = "0 0 0"
    _RPY_RUDD_COLL = "0 0 0"

    # Revolute joint base limits and dynamics
    _REV_LIMIT = {
        "lower": "-0.35",
        "upper": "0.35",
        "effort": "1.5",
        "velocity": "3.665191429",
    }
    _REV_DYN = {
        "damping": "0.2",
        "friction": "0.05",
    }

    # Historical LE shift used by legacy meshes (for visuals)
    _LE_REF = 0.25

    # Servo masses (kg) and box dimensions (m) used for inertia
    _SERVO_WING_SWEEP_MASS = 0.040   # sweep servo per wing side
    _SERVO_WING_TWIST_MASS = 0.030   # twist servo per wing side
    _SERVO_TAIL_ELEV_MASS  = 0.020   # elevator servo
    _SERVO_TAIL_RUDD_MASS  = 0.020   # rudder servo
    _SERVO_SIZE            = (0.03, 0.012, 0.03)  # (sx, sy, sz) ~ 30×12×30 mm

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        genome_or_params: Union[GeometryParams, Sequence[float]],
        *,
        out_dir: Union[str, Path] = "urdf_generated",
        aero_solver_kind: str = "simple",
    ) -> None:
        """
        Construct a URDF maker.

        Parameters
        ----------
        genome_or_params:
            Either:
              • a 15-value genome (Sequence[float]) as specified in the module
                docstring, or
              • a pre-resolved GeometryParams instance.

        out_dir:
            Directory where the generated URDF will be written.
        aero_solver_kind:
            Select which solver aero configuration to use ("simple" or "lisparrow").
        """
        self._raw_genome: List[float] | None = None
        self._actuator_catalog: Dict[Tuple[str, str], Dict[str, str]] = {}
        self._link_actuators: Dict[str, Dict[str, Optional[str]]] = {}
        self._aero_solver_kind = str(aero_solver_kind).strip().lower()
        self._aero_config = self._resolve_aero_config(self._aero_solver_kind, out_dir)
        self._prop_max_thrust: Optional[float] = None
        self._wing_thickness: float = self._REF["tw"]
        self._load_actuator_catalog()
        self._load_link_actuators()

        # Resolve input into GeometryParams
        if isinstance(genome_or_params, GeometryParams):
            prm = genome_or_params
        else:
            # Interpret as genome
            seq = list(genome_or_params)
            if len(seq) != 15:
                raise ValueError("Expected a 15-value genome sequence.")

            (
                wing_span,
                wing_AR,
                fus_length,
                cg_ratio,
                attach_ratio,
                elev_span,
                elev_AR,
                rudd_span,
                rudd_AR,
                dihedral_deg,
                sweep_multi,
                twist_multi,
                naca_d1,
                naca_d2,
                naca_last2,
            ) = seq

            # --- Derived physical quantities --------------------------------
            wing_chord     = wing_span / max(wing_AR, 1e-6)
            elevator_chord = elev_span / max(elev_AR, 1e-6)
            rudder_chord   = rudd_span / max(rudd_AR, 1e-6)
            wing_thickness = max(naca_last2, 0.0) / 100.0 * wing_chord

            fus_cg_x      = -cg_ratio * fus_length
            wing_attach_x = -attach_ratio * fus_length

            density_scale = 1.0
            prop_radius   = 0.10

            hinge_le_ratio = 0.25
            cl_alpha_2d = 2.0
            alpha0_2d_deg = -3.0

            alpha0_2d_rad = alpha0_2d_deg * math.pi / 180.0
            cl_alpha_2d = cl_alpha_2d * math.pi  # legacy scaling kept

            prm = GeometryParams(
                wing_span=wing_span,
                wing_chord=wing_chord,
                wing_attach_x=wing_attach_x,
                fus_cg_x=fus_cg_x,
                fus_length=fus_length,
                elevator_span=elev_span,
                elevator_chord=elevator_chord,
                rudder_span=rudd_span,
                rudder_chord=rudder_chord,
                density_scale=density_scale,
                dihedral_deg=dihedral_deg,
                prop_radius=prop_radius,
                hinge_le_ratio=hinge_le_ratio,
                sweep_multi=sweep_multi,
                twist_multi=twist_multi,
                cl_alpha_2d=cl_alpha_2d,
                alpha0_2d=alpha0_2d_rad,
            )
            self._raw_genome = seq
            if wing_thickness > 0.0:
                self._wing_thickness = wing_thickness

        # Store resolved parameters and derived scalars
        self.p = prm
        self.out = Path(out_dir)
        self.S = prm.density_scale
        self.LE = prm.hinge_le_ratio
        self._apply_actuator_masses()

        # Clamp dihedral to a reasonable range (kept from original implementation)
        self.dihedral = prm.dihedral_deg * math.pi / 180.0

    # ────────────────────────────────────────────────────────────────────
    # Actuator data (YAML + CSV)
    # ────────────────────────────────────────────────────────────────────
    def _load_actuator_catalog(self) -> None:
        """Load actuator properties from actuators.csv into _actuator_catalog."""
        csv_path = Path(__file__).resolve().parents[2] / "genesis" / "assets" / "urdf" / "mydrone" / "actuators.csv"
        catalog: Dict[Tuple[str, str], Dict[str, str]] = {}
        if csv_path.exists():
            try:
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = row.get("name", "").strip()
                        kind = row.get("type", "").strip().lower()
                        if name and kind:
                            catalog[(name, kind)] = {k: v for k, v in row.items()}
            except Exception:
                catalog = {}
        self._actuator_catalog = catalog

    @staticmethod
    def _clean_actuator_name(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        if not isinstance(name, str):
            return None
        stripped = name.strip()
        if not stripped or stripped.lower() == "none":
            return None
        return stripped

    def _get_link_actuator(self, link_name: str, key: str) -> Optional[str]:
        info = self._link_actuators.get(link_name)
        if isinstance(info, dict):
            name = info.get(key)
        elif isinstance(info, str):
            name = info if key == "actuator" else None
        else:
            name = None
        return self._clean_actuator_name(name)

    def _load_link_actuators(self) -> None:
        """Load actuator names per aero link from the solver aero config."""
        links_cfg: Dict[str, Dict] = self._aero_config.get("links", {}) or {}

        link_actuators: Dict[str, Dict[str, Optional[str]]] = {}
        for k, v in links_cfg.items():
            if isinstance(v, dict):
                link_actuators[k] = {
                    "actuator": self._clean_actuator_name(v.get("actuator")),
                    "actuator_yaw": self._clean_actuator_name(v.get("actuator_yaw")),
                    "actuator_pitch": self._clean_actuator_name(v.get("actuator_pitch")),
                }
            else:
                link_actuators[k] = {
                    "actuator": self._clean_actuator_name(v) if isinstance(v, str) else None,
                    "actuator_yaw": None,
                    "actuator_pitch": None,
                }
        self._link_actuators = link_actuators

    @staticmethod
    def _resolve_aero_config(
        solver_kind: str,
        out_dir: Union[str, Path, None] = None,
    ) -> dict:
        # Prefer a local YAML to avoid importing heavy Genesis/MuJoCo stack.
        candidates: List[Path] = []
        env_path = os.getenv("AERO_CONFIG_PATH", "").strip()
        if env_path:
            candidates.append(Path(env_path))
        if out_dir:
            candidates.append(Path(out_dir) / "aero_parameters.yaml")
        repo_default = (
            Path(__file__).resolve().parents[2]
            / "genesis"
            / "assets"
            / "urdf"
            / "mydrone"
            / "aero_parameters.yaml"
        )
        candidates.append(repo_default)

        try:
            import yaml as _yaml  # type: ignore
        except Exception:
            _yaml = None

        if _yaml is not None:
            for path in candidates:
                try:
                    if path.is_file():
                        with open(path, "r") as f:
                            cfg = _yaml.safe_load(f)
                        if isinstance(cfg, dict):
                            return cfg
                except Exception:
                    pass

        from genesis.engine.solvers.drones.simple_drone import SimpleDroneAeroParameters
        from genesis.engine.solvers.drones.lisparrow import LisparrowAeroParameters
        name = (solver_kind or "").strip().lower()
        if name in ("lisparrow", "cpp", "morphing"):
            return LisparrowAeroParameters.as_dict()
        return SimpleDroneAeroParameters.as_dict()

    def _actuator_mass(self, name: Optional[str], kind: str, fallback: Optional[float]) -> Optional[float]:
        if not name:
            return fallback
        entry = self._actuator_catalog.get((name, kind.lower()))
        if entry and "mass" in entry:
            try:
                return float(entry["mass"])
            except Exception:
                return fallback
        return fallback

    def _actuator_value(
        self,
        name: Optional[str],
        kind: str,
        key: str,
        fallback: Optional[float] = None,
    ) -> Optional[float]:
        if not name:
            return fallback
        entry = self._actuator_catalog.get((name, kind.lower()))
        if not entry:
            return fallback
        raw = entry.get(key)
        if raw is None:
            return fallback
        value = str(raw).strip()
        if not value or value.lower() == "none":
            return fallback
        try:
            return float(value)
        except Exception:
            return fallback

    @staticmethod
    def _format_float_str(value: Optional[str], fallback: str) -> str:
        if value is None:
            return fallback
        s = str(value).strip()
        if not s or s.lower() == "none":
            return fallback
        try:
            return f"{float(s):g}"
        except Exception:
            return fallback

    def _actuator_dynamics(
        self,
        name: Optional[str],
        kind: str,
        fallback: Dict[str, str],
        *,
        ratio: float = 1.0,
    ) -> Dict[str, str]:
        if not name:
            return dict(fallback)
        entry = self._actuator_catalog.get((name, kind.lower()))
        if not entry:
            return dict(fallback)
        damping_s = self._format_float_str(entry.get("damping"), fallback.get("damping", "0.0"))
        friction_s = self._format_float_str(entry.get("friction"), fallback.get("friction", "0.0"))

        r = max(float(ratio), 1e-6)
        d = float(damping_s) * (r * r)
        f = float(friction_s) * r

        return {"damping": f"{d:g}", "friction": f"{f:g}"}


    def _apply_actuator_masses(self) -> None:
        """Override servo/prop masses based on YAML-selected actuators and catalog."""
        wing_act = self._get_link_actuator("aero_frame_left_wing", "actuator") or self._get_link_actuator(
            "aero_frame_right_wing", "actuator"
        )
        elev_act = self._get_link_actuator("aero_frame_elevator_left", "actuator") or self._get_link_actuator(
            "aero_frame_elevator_right", "actuator"
        )
        rudd_act = self._get_link_actuator("aero_frame_rudder", "actuator")
        prop_act = self._get_link_actuator("prop_frame_fuselage_0", "actuator")

        self._SERVO_WING_SWEEP_MASS = self._actuator_mass(wing_act, "servo", self._SERVO_WING_SWEEP_MASS)
        self._SERVO_WING_TWIST_MASS = self._actuator_mass(wing_act, "servo", self._SERVO_WING_TWIST_MASS)
        self._SERVO_TAIL_ELEV_MASS = self._actuator_mass(elev_act, "servo", self._SERVO_TAIL_ELEV_MASS)
        self._SERVO_TAIL_RUDD_MASS = self._actuator_mass(rudd_act, "servo", self._SERVO_TAIL_RUDD_MASS)

        prop_mass = self._actuator_mass(prop_act, "propeller", None)
        prop_radius = self._actuator_value(prop_act, "propeller", "radius", None)
        self._prop_max_thrust = self._actuator_value(prop_act, "propeller", "max_thrust", None)

        if prop_radius is not None and prop_radius > 0.0:
            self.p.prop_radius = prop_radius
        if prop_mass is not None:
            r = self.p.prop_radius
            h = 0.005
            vol = math.pi * r * r * h
            if vol > 0:
                self._RHO_PROP = prop_mass / vol

    # ────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _indent(elem: ET.Element, lvl: int = 0) -> None:
        """
        In-place pretty-print indentation for an ElementTree node.

        This function modifies .text and .tail of elements in the tree
        so that the serialized XML is human-readable.
    """
        pad = "\n" + lvl * "  "
        if len(elem):
            if not (elem.text or "").strip():
                elem.text = pad + "  "
            for child in elem:
                UrdfMaker._indent(child, lvl + 1)
            if not (elem.tail or "").strip():
                elem.tail = pad
        elif lvl and not (elem.tail or "").strip():
            elem.tail = pad

    def _origin(
        self,
        parent: ET.Element,
        xyz: Tuple[float, float, float],
        rpy: str = "0 0 0",
    ) -> None:
        """Attach an <origin> tag with the given xyz and rpy values."""
        ET.SubElement(
            parent,
            "origin",
            xyz=" ".join(f"{v:.12g}" for v in xyz),
            rpy=rpy,
        )

    def _add_inertial(
        self,
        link: ET.Element,
        xyz: Tuple[float, float, float],
        mass: float,
        inertia_diag: Tuple[float, float, float],
    ) -> None:
        """
        Add an <inertial> tag to the given link with:
        - a box-diagonal inertia tensor (ixx, iyy, izz),
        - zero products of inertia (ixy, ixz, iyz),
        - and the specified mass.
        """
        inertial = ET.SubElement(link, "inertial")
        self._origin(inertial, xyz)
        ET.SubElement(inertial, "mass", value=f"{mass:.6g}")
        ET.SubElement(
            inertial,
            "inertia",
            ixx=f"{inertia_diag[0]:.6g}",
            ixy="0",
            ixz="0",
            iyy=f"{inertia_diag[1]:.6g}",
            iyz="0",
            izz=f"{inertia_diag[2]:.6g}",
        )

    def _I_box(
        self,
        mass: float,
        sx: float,
        sy: float,
        sz: float,
        *,
        fudge: float | None = None,
    ) -> Tuple[float, float, float]:
        """
        Compute the inertia tensor diagonal of a solid box (sx, sy, sz)
        with a scaling fudge factor (empirical).

        Returns (ixx, iyy, izz).
        """
        k = self._I_FUDGE if fudge is None else fudge
        ixx = k * mass * (sy**2 + sz**2) / 12.0
        iyy = k * mass * (sx**2 + sz**2) / 12.0
        izz = k * mass * (sx**2 + sy**2) / 12.0
        return ixx, iyy, izz

    # ------------------------------------------------------------------
    # Genome handling
    # ------------------------------------------------------------------
    def _raw_genome_values(self) -> List[float]:
        """
        Return the 15-value genome:
        - If a raw genome was provided, return it.
        - Otherwise, reconstruct it approximately from GeometryParams.
        """
        if self._raw_genome is not None:
            return self._raw_genome

        p = self.p
        return [
            p.wing_span,
            p.wing_span / p.wing_chord if p.wing_chord else float("nan"),
            p.fus_length,
            p.fus_cg_x / p.fus_length if p.fus_length else float("nan"),
            p.wing_attach_x / p.fus_length if p.fus_length else float("nan"),
            p.elevator_span,
            p.elevator_span / p.elevator_chord if p.elevator_chord else float("nan"),
            p.rudder_span,
            p.rudder_span / p.rudder_chord if p.rudder_chord else float("nan"),
            p.dihedral_deg,
            p.sweep_multi,
            p.twist_multi,
            3.0,
            4.0,
            16.0,
        ]

    # ------------------------------------------------------------------
    # Mesh scale mapping
    # ------------------------------------------------------------------
    def _scale(self, what: str, dims: Tuple[float, float, float]) -> str:
        """
        Compute mesh scaling factors for a given component type.

        Parameters
        ----------
        what:
            One of "fuselage", "wing", "elevator", "rudder".

        dims:
            Physical dimensions used to scale the mesh.
            For fuselage: (tfus, hfus, fus_length)
            For others   : (thickness, chord, span)

        Returns
        -------
        scale_str : str
            Space-separated "sx sy sz" scaling factor suitable for URDF.
        """
        ref_scale = self._REF_SCALE[what]

        if what == "fuselage":
            # Fuselage: X = length, Y = height, Z = thickness
            tfus, hfus, fus_len = dims
            tfus_ref = self._REF["tfus"]
            hfus_ref = self._REF["hfus"]
            fus_len_ref = self._REF["fus_len"]

            sx = ref_scale[0] * fus_len / max(fus_len_ref, 1e-9)
            sy = ref_scale[1] * hfus    / max(hfus_ref,    1e-9)
            sz = ref_scale[2] * tfus    / max(tfus_ref,    1e-9)
            return f"{sx:.6g} {sy:.6g} {sz:.6g}"

        # For wing meshes the STL encodes:
        #   span along Y, chord along Z, thickness along X.
        # Map:
        #   X ← thickness, Y ← span, Z ← chord
        ref_dims = {
            "wing":     (self._REF["tw"],   self._REF["wing_chord"], self._REF["wing_span_ref"]),
            "elevator": (self._REF["tc"],   self._REF["elev_chord"], self._REF["elev_span"]),
            "rudder":   (self._REF["tc"],   self._REF["rudd_chord"], self._REF["rudd_span"]),
        }[what]

        thk_cur, chord_cur, span_cur = dims
        thk_ref, chord_ref, span_ref = ref_dims

        # Elevator mesh is oriented with chord along X, span along Y, thickness along Z.
        if what == "elevator":
            sx = ref_scale[0] * chord_cur / max(chord_ref, 1e-9)   # chord  → X
            sy = ref_scale[1] * span_cur  / max(span_ref,  1e-9)   # span   → Y
            sz = ref_scale[2] * thk_cur   / max(thk_ref,   1e-9)   # thick  → Z
            return f"{sx:.6g} {sy:.6g} {sz:.6g}"

        # Rudder is treated as a vertical wing: chord along X (longitudinal),
        # thickness along Y, span along Z (height).
        if what == "rudder":
            sx = ref_scale[0] * chord_cur / max(chord_ref, 1e-9)   # chord  → X
            sy = ref_scale[1] * thk_cur   / max(thk_ref,   1e-9)   # thick  → Y
            sz = ref_scale[2] * span_cur  / max(span_ref,  1e-9)   # span   → Z
            return f"{sx:.6g} {sy:.6g} {sz:.6g}"

        # Wing visual mesh orientation: chord on X, span on Y, thickness on Z.
        if what == "wing":
            sx = ref_scale[0] * chord_cur / max(chord_ref, 1e-9)  # chord     → X
            sy = ref_scale[1] * span_cur  / max(span_ref,  1e-9)  # span      → Y
            sz = ref_scale[2] * thk_cur   / max(thk_ref,   1e-9)  # thickness → Z
            return f"{sx:.6g} {sy:.6g} {sz:.6g}"

        sx = ref_scale[0] * thk_cur   / max(thk_ref,   1e-9)  # thickness → X
        sy = ref_scale[1] * span_cur  / max(span_ref,  1e-9)  # span      → Y
        sz = ref_scale[2] * chord_cur / max(chord_ref, 1e-9)  # chord     → Z

        return f"{sx:.6g} {sy:.6g} {sz:.6g}"

    # ────────────────────────────────────────────────────────────────────
    # Materials & root
    # ────────────────────────────────────────────────────────────────────

    def _materials(self, robot: ET.Element) -> None:
        """Create a small set of named materials (grey, red, black)."""
        materials = {
            "grey":  (0.7,  0.7,  0.7,  1.0),
            "red":   (0.7,  0.0,  0.0,  1.0),
            "black": (0.2,  0.2,  0.2,  1.0),
        }

        for name, rgba in materials.items():
            mat = ET.SubElement(robot, "material", name=name)
            ET.SubElement(mat, "color", rgba=" ".join(map(str, rgba)))

    def _root_link(self, robot: ET.Element) -> None:
        """
        Create a massless 'root_link' to which the fuselage is attached.

        This is common in ROS/Gazebo models and allows easy re-use in
        larger systems.
        """
        link = ET.SubElement(robot, "link", name="root_link")
        self._add_inertial(link, (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0))

    def _add_fixed_mass(
        self,
        robot: ET.Element,
        *,
        name: str,
        parent: str,
        xyz: Tuple[float, float, float],
        rpy: str = "0 0 0",
        m: float,
        size: Tuple[float, float, float] | None = None,
        sphere_radius: float = 0.015,
    ) -> None:
        """
        Create a link with fixed mass `m` attached to `parent` by a fixed joint.

        If `size` is provided:
            - inertia is assumed to be a solid box with those dimensions.
        Otherwise:
            - inertia is approximated as a solid sphere with radius `sphere_radius`.

        This is used for servo masses and other small lumped masses.
        """
        if m <= 0.0:
            inertia = (0.0, 0.0, 0.0)
        else:
            if size is not None:
                sx, sy, sz = size
                inertia = self._I_box(m, sx, sy, sz, fudge=1.0)
            else:
                r = sphere_radius
                i_val = (2.0 / 5.0) * m * (r * r)  # solid sphere around center
                inertia = (i_val, i_val, i_val)

        link = ET.SubElement(robot, "link", name=name)
        self._add_inertial(link, (0.0, 0.0, 0.0), m, inertia)

        joint = ET.SubElement(robot, "joint", name=f"fixed_joint_{name}", type="fixed")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=name)
        self._origin(joint, xyz, rpy)

    # ────────────────────────────────────────────────────────────────────
    # Fuselage
    # ────────────────────────────────────────────────────────────────────

    def _fuselage(self, robot: ET.Element) -> None:
        """Create fuselage link, its inertial model, collision and visual."""
        p = self.p

        # Box dims for URDF geometry; mass uses a cylindrical cross-section.
        box = (self._REF["tfus"], self._REF["hfus"], p.fus_length)
        t = self._FUS_SHELL_THICKNESS  # 1 cm
        D_out, H_out, L = box  # outer diameters + length

        # Inner diameters (clamp to avoid negative)
        D_in = max(D_out - 2.0 * t, 1e-6)
        H_in = max(H_out - 2.0 * t, 1e-6)

        A_out = math.pi * 0.25 * D_out * H_out
        A_in  = math.pi * 0.25 * D_in  * H_in
        volume_shell = (A_out - A_in) * L

        # Mass model: structural shell + fixed payload (battery, avionics, etc.)
        m_shell = volume_shell * self._RHO_FUS_STRUCT * self.S
        m_batt  = self._FUS_FIXED_MASS
        m_total = m_shell + m_batt

        # Inertia (shell + battery)
        I_shell = self._I_box(m_shell, *box)
        bx, by, bz = self._BATTERY_SIZE
        I_batt  = self._I_box(m_batt, bx, by, bz, fudge=1.0)
        I_total = tuple(a + b for a, b in zip(I_shell, I_batt))

        # Main fuselage link
        fus = ET.SubElement(robot, "link", name="fuselage")
        self._add_inertial(fus, (p.fus_cg_x, 0.0, -0.04), m_total, I_total)

        # Collision box (no artificial shrink; use physical size)
        box_coll = (
            box[0] * self._SHRINK,
            box[1] * self._SHRINK,
            box[2] * self._SHRINK,
        )
        coll = ET.SubElement(fus, "collision", name="fuselage_collision_0")
        self._origin(
            coll,
            (-0.5 * p.fus_length, 0.0, 0.0024556534860724526),
            self._RPY_FUSE_COLL,
        )
        ET.SubElement(
            ET.SubElement(coll, "geometry"),
            "box",
            size=" ".join(f"{v:.12g}" for v in box_coll),
        )

        # Visual (mesh with scale)
        vis = ET.SubElement(fus, "visual", name="fuselage_visual_0")
        self._origin(vis, (0.0, 0.0, 0.0))
        mesh = ET.SubElement(
            ET.SubElement(vis, "geometry"),
            "mesh",
            filename="package://meshes/fuselage_only.stl",
        )
        mesh.set("scale", self._scale("fuselage", box))
        ET.SubElement(vis, "material", name="grey")

        # Root → fuselage joint (URDF base alignment)
        j0 = ET.SubElement(robot, "joint", name="fixed_joint_base_fuselage", type="fixed")
        ET.SubElement(j0, "parent", link="root_link")
        ET.SubElement(j0, "child", link="fuselage")
        self._origin(j0, (0.0, 0.0, 0.0), "3.141592653589793 0 0")

        # Aero frame attached at fuselage CG
        jaf = ET.SubElement(
            robot,
            "joint",
            name="fixed_joint_aero_frame_fuselage",
            type="fixed",
        )
        ET.SubElement(jaf, "parent", link="fuselage")
        ET.SubElement(jaf, "child", link="aero_frame_fuselage")
        self._origin(jaf, (p.fus_cg_x, 0.0, -0.03), "0 3.141592653589793 0")

        aero_fus = ET.SubElement(robot, "link", name="aero_frame_fuselage")
        self._add_inertial(aero_fus, (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0))

    # ────────────────────────────────────────────────────────────────────
    # Propeller frame
    # ────────────────────────────────────────────────────────────────────

    def _prop_frame(self, robot: ET.Element) -> None:
        """
        Create a propeller frame link attached to the fuselage.

        Modeled as a thin cylinder for visualization, and a small box for
        collision. Mass and inertia are derived from a solid cylinder.
        """
        p = self.p

        joint = ET.SubElement(
            robot,
            "joint",
            name="fixed_joint_prop_frame_fuselage_0",
            type="fixed",
        )
        ET.SubElement(joint, "parent", link="fuselage")
        ET.SubElement(joint, "child", link="prop_frame_fuselage_0")
        self._origin(
            joint,
            (0.0, 0.0, 0.0),
            "3.141592653589793 -1.5707963267948966 0",
        )

        r = p.prop_radius
        h = 0.005  # thin cylinder
        m = math.pi * r * r * h * self._RHO_PROP * self.S
        I = self._I_box(m, 2 * r, 2 * r, h)

        prop_link = ET.SubElement(robot, "link", name="prop_frame_fuselage_0")
        self._add_inertial(prop_link, (0.0, 0.0, 0.0), m, I)
        if self._PROP_CAMERA_MASS > 0.0:
            self._add_fixed_mass(
                robot,
                name="prop_camera_payload",
                parent="prop_frame_fuselage_0",
                xyz=(0.0, 0.0, 0.0),
                m=self._PROP_CAMERA_MASS,
                size=None,
            )

        # Visual cylinder
        vis = ET.SubElement(prop_link, "visual", name="prop_frame_fuselage_0_visual_0")
        self._origin(vis, (0.0, 0.0, 0.0))
        ET.SubElement(
            ET.SubElement(vis, "geometry"),
            "cylinder",
            radius=f"{r}",
            length=f"{h}",
        )
        ET.SubElement(vis, "material", name="black")

        # Collision box (simplified disc)
        coll = ET.SubElement(prop_link, "collision", name="prop_frame_collision")
        self._origin(coll, (0.0, 0.0, 0.0))
        ET.SubElement(
            ET.SubElement(coll, "geometry"),
            "box",
            size="0.2 0.2 0.005",
        )

    # ────────────────────────────────────────────────────────────────────
    # Wings (sweep & twist joints + LE shift for visuals/frames)
    # ────────────────────────────────────────────────────────────────────

    def _wings(self, robot: ET.Element) -> None:
        """Create both left and right wing assemblies with joints and aero frames."""
        p = self.p
        thickness = self._wing_thickness
        left_yaw_act = self._get_link_actuator("aero_frame_left_wing", "actuator_yaw")
        left_pitch_act = self._get_link_actuator("aero_frame_left_wing", "actuator_pitch")
        right_yaw_act = self._get_link_actuator("aero_frame_right_wing", "actuator_yaw")
        right_pitch_act = self._get_link_actuator("aero_frame_right_wing", "actuator_pitch")

        rho = self._RHO_WING * self.S * self._WING_FOAM_FILL
        chord = p.wing_chord
        span_total = p.wing_span  # per-side span

        def _service_link(name: str) -> None:
            """
            Create a small helper link with non-zero inertia.

            This is used as an intermediate "structural" link in the wing
            actuator chain.
            """
            link = ET.SubElement(robot, "link", name=name)
            self._add_inertial(
                link,
                (0.0, 0.0, 0.0),
                self._MASS_INTER,
                self._I_box(self._MASS_INTER, thickness, thickness, thickness),
            )

        def _panel(
            side: str,
            sign: int,
        ) -> None:
            """
            Create a wing panel (either in propwash or free stream).

            Parameters
            ----------
            side:
                "left" or "right".
            """
            name = f"{side}_wing"
            span_side = span_total
            span_offset = 0.5 * span_side

            # Geometric box used for inertia (dims order doesn't matter,
            # only the magnitudes enter the inertia calculation)
            box_geom = (thickness, chord, span_side)
            mass_panel = math.prod(box_geom) * rho
            I_panel = self._I_box(mass_panel, *box_geom)

            # Link with inertia at aerodynamic CG
            link = ET.SubElement(robot, "link", name=name)
            self._add_inertial(
                link,
                ((self._AERO_RATIO - self.LE) * chord, -sign * span_offset, 0.0),
                mass_panel,
                I_panel,
            )

            # Visual (half-wing mesh scaled to full span per side)
            vis = ET.SubElement(link, "visual", name=f"{name}_visual")
            # The wing STL has span along Y and chord along Z.
            # To make the leading edge face forward (consistent between left
            # and right), add a 180° rotation around Y that flips the chord.
            if side == "left":
                # Left wing: mirror in X (existing π roll) and flip chord with π pitch.
                rpy = "0 3.141592653589793 0"
            else:
                # Right wing: only flip chord with π pitch.
                rpy = "3.141592653589793 3.141592653589793 0"
            vis_shift_x = (self.LE - self._LE_REF) * chord
            self._origin(vis, (vis_shift_x, 0.0, 0.0), rpy)
            mesh = ET.SubElement(
                ET.SubElement(vis, "geometry"),
                "mesh",
                filename="package://meshes/wing0009.stl",
            )
            mesh.set("scale", self._scale("wing", (thickness, chord, span_side)))
            ET.SubElement(vis, "material", name="red")

            # Collision geometry:
            # IMPORTANT: DroneAeroModel assumes box size is (thickness, chord, span)
            # so that chord=sy and span=sz.
            coll = ET.SubElement(link, "collision", name=f"{name}_collision")
            coll_box = (thickness, chord, span_side)  # (sx, sy, sz) = (thk, chord, span)
            # Rotate the box so it still looks like (chord, span, thickness) in the link frame.
            # This rotation maps: X(thk)->Z, Y(chord)->X, Z(span)->Y
            self._origin(
                coll,
                (0.5 * chord, sign * span_offset, 0.0),
                "0 -1.5707963267948966 -1.5707963267948966",
            )
            ET.SubElement(
                ET.SubElement(coll, "geometry"),
                "box",
                size=" ".join(f"{v:.12g}" for v in coll_box),
            )

            # Aero frame anchored at CG (LE-based frame)
            ja = ET.SubElement(
                robot,
                "joint",
                name=f"fixed_joint_aero_frame_{name}",
                type="fixed",
            )
            ET.SubElement(ja, "parent", link=name)
            ET.SubElement(ja, "child", link=f"aero_frame_{name}")
            self._origin(
                ja,
                ((self._AERO_RATIO - self.LE) * chord, -sign * span_offset, 0.0),
                "0 3.141592654 0",
            )

            aero_link = ET.SubElement(robot, "link", name=f"aero_frame_{name}")
            self._add_inertial(aero_link, (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0))

        def _wing_side(side: str, sign: int) -> None:
            """
            Build one side of the wing (left or right) including:

              • fixed offset from fuselage with dihedral;
              • sweep + twist revolute joints;
              • servo masses;
              • visual mesh and aero/collision panels.
            """
            # Joint limits scale with sweep/twist multipliers (kept compatible)
            limit_sweep  = 0.7 / max(self.p.sweep_multi, 0.5)   # [rad]
            limit_twist  = 0.5 / max(self.p.twist_multi, 0.5)   # [rad]
            effort_sweep = 0.75 * max(self.p.sweep_multi, 0.5)
            effort_twist = 0.6 * max(self.p.twist_multi, 0.5)
            if side == "left":
                yaw_act = left_yaw_act
                pitch_act = left_pitch_act
            else:
                yaw_act = right_yaw_act
                pitch_act = right_pitch_act

            # Attach wing root to fuselage with dihedral
            fj = ET.SubElement(robot, "joint", name=f"fixed_joint_{side}_wing", type="fixed")
            ET.SubElement(fj, "parent", link="fuselage")
            ET.SubElement(fj, "child", link=f"fuselage_{side}_0")
            self._origin(
                fj,
                (p.wing_attach_x, self._ROOT_Y_OFFSET * sign, -0.03),
                f"{self.dihedral * sign:.12g} 0 0",
            )

            # Intermediate service links
            _service_link(f"fuselage_{side}_0")
            _service_link(f"fuselage_{side}_1")

            # 1) SWEEP joint: fuselage_{side}_0 → fuselage_{side}_1
            js = ET.SubElement(
                robot,
                "joint",
                name=f"joint_0_sweep_{side}_wing",
                type="revolute",
            )
            ET.SubElement(js, "parent", link=f"fuselage_{side}_0")
            ET.SubElement(js, "child", link=f"fuselage_{side}_1")
            self._origin(js, (0.0, 0.0, 0.0))
            ET.SubElement(js, "axis", xyz="0 0 1")
            ET.SubElement(
                js,
                "limit",
                lower=f"{-limit_sweep:.6g}",
                upper=f"{limit_sweep:.6g}",
                effort=f"{effort_sweep:.6g}",
                velocity=self._REV_LIMIT["velocity"],
            )
            ET.SubElement(
                js,
                "dynamics",
                **self._actuator_dynamics(yaw_act, "servo", self._REV_DYN, ratio=p.sweep_multi),
            )

            # Sweep servo mass
            self._add_fixed_mass(
                robot,
                name=f"{side}_wing_servo_sweep",
                parent=f"fuselage_{side}_0",
                xyz=(0.0, 0.0, 0.0),
                m=self._SERVO_WING_SWEEP_MASS,
                size=None,
                sphere_radius=0.015,
            )

            # 2) TWIST joint: fuselage_{side}_1 → {side}_wing
            jt = ET.SubElement(
                robot,
                "joint",
                name=f"joint_1_twist_{side}_wing",
                type="revolute",
            )
            ET.SubElement(jt, "parent", link=f"fuselage_{side}_1")
            ET.SubElement(jt, "child", link=f"{side}_wing")
            # Do not cancel the lateral ROOT_Y_OFFSET here: keep the wing
            # link at the same y as fuselage_{side}_1.
            self._origin(jt, (0.0, 0.0, 0.0))
            ET.SubElement(jt, "axis", xyz="0 1 0")
            ET.SubElement(
                jt,
                "limit",
                lower=f"{-limit_twist:.6g}",
                upper=f"{limit_twist:.6g}",
                effort=f"{effort_twist:.6g}",
                velocity=self._REV_LIMIT["velocity"],
            )

            ET.SubElement(
                jt,
                "dynamics",
                **self._actuator_dynamics(pitch_act, "servo", self._REV_DYN, ratio=p.twist_multi),
            )

            # Twist servo mass
            self._add_fixed_mass(
                robot,
                name=f"{side}_wing_servo_twist",
                parent=f"fuselage_{side}_1",
                # Co-located with the twist joint (no extra y-offset).
                xyz=(0.0, 0.0, 0.0),
                m=self._SERVO_WING_TWIST_MASS,
                size=None,
                sphere_radius=0.015,
            )

            # Single wing panel (no propwash/free split)
            _panel(side, sign)

        # Build right and left sides
        _wing_side("right", +1)
        _wing_side("left",  -1)

    # ────────────────────────────────────────────────────────────────────
    # Elevator (pitch) – LE-based frames for aero, legacy signs for coll
    # ────────────────────────────────────────────────────────────────────

    def _elevator(self, robot: ET.Element) -> None:
        """Create elevator joint, links, aero frames and collision geometry."""
        p = self.p
        thickness = 0.16 * p.elevator_chord

        # Elevator treated as a horizontal wing: chord → X, span → Y, thickness → Z
        box_full = (p.elevator_chord, p.elevator_span, thickness)
        span_half = p.elevator_span / 2.0
        box_half = (thickness, p.elevator_chord, span_half)

        rho = self._RHO_ELEV * self.S * self._WING_FOAM_FILL
        m_half = math.prod(box_half) * rho
        I_half = self._I_box(m_half, *box_half)

        chord = p.elevator_chord
        elev_pitch_act = self._get_link_actuator(
            "aero_frame_elevator_left",
            "actuator_pitch",
        ) or self._get_link_actuator("aero_frame_elevator_right", "actuator_pitch")

        # Main revolute hinge (pitch)
        jh = ET.SubElement(robot, "joint", name="elevator_pitch_joint", type="revolute")
        ET.SubElement(jh, "parent", link="fuselage")
        ET.SubElement(jh, "child", link="elevator_hinge")
        self._origin(jh, (-p.fus_length - 0.03, 0.0, 0.0))
        ET.SubElement(jh, "axis", xyz="0 1 0")
        ET.SubElement(
            jh,
            "limit",
            lower="-0.25",
            upper="0.25",  # keep original values
            effort="1.0",
            velocity="3.665191429",
        )
        ET.SubElement(
            jh,
            "dynamics",
            **self._actuator_dynamics(elev_pitch_act, "servo", self._REV_DYN),
        )

        # Elevator servo mass attached at the hinge location
        self._add_fixed_mass(
            robot,
            name="elevator_servo",
            parent="fuselage",
            xyz=(-p.fus_length - 0.03, 0.0, 0.0),
            m=self._SERVO_TAIL_ELEV_MASS,
            size=None,
        )

        # Hinge link (structural intermediate)
        hinge_link = ET.SubElement(robot, "link", name="elevator_hinge")
        I_hinge = self._I_box(
            self._MASS_INTER * 4,
            p.elevator_chord,
            p.elevator_span,
            thickness,
        )
        self._add_inertial(hinge_link, (0.0, 0.0, 0.0), self._MASS_INTER * 4, I_hinge)

        # Whole elevator visual mesh (for convenience)
        vis_h = ET.SubElement(hinge_link, "visual")
        self._origin(vis_h, (0.0, 0.0, 0.0))
        mesh_h = ET.SubElement(
            ET.SubElement(vis_h, "geometry"),
            "mesh",
            filename="package://meshes/elevator.stl",
        )
        mesh_h.set("scale", self._scale("elevator", (thickness, p.elevator_chord, p.elevator_span)))
        ET.SubElement(vis_h, "material", name="grey")

        # Left and right elevator halves
        for sign, side in ((+1, "left"), (-1, "right")):
            fj = ET.SubElement(robot, "joint", name=f"fixed_joint_elevator_{side}", type="fixed")
            ET.SubElement(fj, "parent", link="elevator_hinge")
            ET.SubElement(fj, "child", link=f"elevator_{side}")
            self._origin(fj, (0.0, span_half * sign, 0.0))

            ln = ET.SubElement(robot, "link", name=f"elevator_{side}")
            # CG from LE (+ forward); aero frame mirrors this
            self._add_inertial(ln, (self._CG_RATIO * chord, 0.0, 0.0), m_half, I_half)

            ja = ET.SubElement(
                robot,
                "joint",
                name=f"fixed_joint_aero_frame_elevator_{side}",
                type="fixed",
            )
            ET.SubElement(ja, "parent", link=f"elevator_{side}")
            ET.SubElement(ja, "child", link=f"aero_frame_elevator_{side}")
            self._origin(ja, (-self._AERO_RATIO * chord, 0.0, 0.0), "0 3.141592654 0")

            aero = ET.SubElement(robot, "link", name=f"aero_frame_elevator_{side}")
            self._add_inertial(aero, (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0))

            coll = ET.SubElement(ln, "collision")
            self._origin(coll, (-self._COLL_RATIO * chord, 0.0, 0.0), self._RPY_ELEV_COLL)
            ET.SubElement(
                ET.SubElement(coll, "geometry"),
                "box",
                size=" ".join(f"{v:.12g}" for v in box_half),
            )

    # ────────────────────────────────────────────────────────────────────
    # Rudder (yaw)
    # ────────────────────────────────────────────────────────────────────

    def _rudder(self, robot: ET.Element) -> None:
        """Create rudder yaw joint, link, aero frame and collision geometry."""
        p = self.p
        chord = p.rudder_chord
        span = p.rudder_span
        thickness = 0.16 * chord

        # Rudder geometry is aligned like a vertical wing:
        #   X → chord (longitudinal), Y → thickness, Z → span (height)
        box_geom = (thickness, chord, span)
        mass_rudder = math.prod(box_geom) * self._RHO_RUDD * self.S * self._WING_FOAM_FILL
        I_rudder = self._I_box(mass_rudder, *box_geom)
        rudder_yaw_act = self._get_link_actuator("aero_frame_rudder", "actuator_yaw")

        # Rudder yaw joint (attached to elevator hinge)
        jy = ET.SubElement(robot, "joint", name="rudder_yaw_joint", type="revolute")
        ET.SubElement(jy, "parent", link="elevator_hinge")
        ET.SubElement(jy, "child", link="rudder")
        self._origin(jy, (0.0, 0.0, 0.0))
        ET.SubElement(jy, "axis", xyz="0 0 1")
        ET.SubElement(
            jy,
            "limit",
            lower="-0.25",
            upper="0.25",
            effort="1.0",
            velocity="3.665191429",
        )
        ET.SubElement(
            jy,
            "dynamics",
            **self._actuator_dynamics(rudder_yaw_act, "servo", self._REV_DYN),
        )

        # Rudder servo mass
        self._add_fixed_mass(
            robot,
            name="rudder_servo",
            parent="elevator_hinge",
            xyz=(0.0, 0.0, 0.0),
            m=self._SERVO_TAIL_RUDD_MASS,
            size=None,
        )

        # Rudder link
        ln = ET.SubElement(robot, "link", name="rudder")
        self._add_inertial(ln, (-self._CG_RATIO * chord, 0.0, -0.08), mass_rudder, I_rudder)

        # Collision geometry
        coll = ET.SubElement(ln, "collision", name="rudder_collision_0")
        self._origin(
            coll,
            (-self._COLL_RATIO * chord, 0.0, -0.10499447081596952),
            self._RPY_RUDD_COLL,
        )
        ET.SubElement(
            ET.SubElement(coll, "geometry"),
            "box",
            size=" ".join(f"{v:.12g}" for v in box_geom),
        )

        # Visual mesh
        vis = ET.SubElement(ln, "visual", name="rudder_visual_0")
        self._origin(vis, (0.0, 0.0, 0.0))
        mesh = ET.SubElement(
            ET.SubElement(vis, "geometry"),
            "mesh",
            filename="package://meshes/rudder.stl",
        )
        mesh.set("scale", self._scale("rudder", (thickness, chord, span)))
        ET.SubElement(vis, "material", name="red")

        # Aero frame at rudder CG
        ja = ET.SubElement(
            robot,
            "joint",
            name="fixed_joint_aero_frame_rudder",
            type="fixed",
        )
        ET.SubElement(ja, "parent", link="rudder")
        ET.SubElement(ja, "child", link="aero_frame_rudder")
        self._origin(ja, (-self._AERO_RATIO * chord, 0.0, -0.08), "0 3.141592653589793 0")

        aero = ET.SubElement(robot, "link", name="aero_frame_rudder")
        self._add_inertial(aero, (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0))

    # ────────────────────────────────────────────────────────────────────
    # Metadata & export
    # ────────────────────────────────────────────────────────────────────

    def _add_metadata(self, robot: ET.Element) -> None:
        """
        Write a single XML comment with the raw 15-gene genome.

        This is useful for traceability and reconstruction of the geometry
        from the URDF alone.
        """
        comment_text = "[" + ", ".join(f"{v:g}" for v in self._raw_genome_values()) + "]"
        robot.append(ET.Comment(comment_text))
        if self._prop_max_thrust is not None:
            act = self._get_link_actuator("prop_frame_fuselage_0", "actuator")
            if act:
                robot.append(
                    ET.Comment(f"propeller_actuator={act} max_thrust={self._prop_max_thrust:g}")
                )

    def build_tree(self) -> ET.ElementTree:
        """
        Build the full ElementTree for the URDF, but do not write it to disk.

        Returns
        -------
        tree : xml.etree.ElementTree.ElementTree
            The URDF XML tree with all links, joints and metadata.
        """
        robot = ET.Element("robot", name="drone_param")

        self._materials(robot)
        self._root_link(robot)
        self._fuselage(robot)
        self._prop_frame(robot)
        self._wings(robot)
        self._elevator(robot)
        self._rudder(robot)
        self._add_metadata(robot)

        return ET.ElementTree(robot)

    def _params_as_string(self) -> str:
        """Return genome as a filename-friendly string."""
        return "[" + ", ".join(f"{v:g}" for v in self._raw_genome_values()) + "]"

    def create_urdf(self, filename: str | None = None) -> str:
        """
        Generate the URDF file on disk.

        Parameters
        ----------
        filename:
            Name of the output file.
            If None, the genome list string is used as the filename.

        Returns
        -------
        path_str : str
            The path to the written URDF file, as a string.
        """
        if filename is None:
            filename = f"{self._params_as_string()}.urdf"

        self.out.mkdir(parents=True, exist_ok=True)

        tree = self.build_tree()
        self._indent(tree.getroot())

        out_path = self.out / filename
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simple genome examples (values chosen for sanity, not performance)
    genome1 = [0.7, 3.5, 0.73, 0.38, 0.38, 0.5, 4.0, 0.2, 2.0, -10, 2.0, 2.5, 3, 4, 16]
    genome2 = [0.7, 3.5, 0.73, 0.38, 0.38, 0.5, 4.0, 0.2, 2.0, 0, 2.0, 2.5, 3, 4, 16]
    genome3 = [0.488441, 2.04645, 0.634358, 0.412812, 0.355771, 0.505386, 2.34147, 0.220355, 1.70431, 1.59352, 2.458, 2.83091, 4, 4, 12]

    for genome in (genome1, genome2, genome3):
        path = UrdfMaker(genome).create_urdf()
        print("URDF written to:", path)
