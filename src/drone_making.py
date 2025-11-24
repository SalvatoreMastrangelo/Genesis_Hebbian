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
10) hinge_le_ratio                [fraction of chord]
11) sweep_multiplier              [-]
12) twist_multiplier              [-]
13) cl_alpha_2d                   [1/rad]
14) alpha0_2d                     [deg] (converted to rad internally)

All *physical* dimensions required by the URDF (chords, x-positions) are
derived internally from the genome above.

The output URDF contains a single XML comment with the **raw genome** (no names),
so the geometry can be reconstructed or traced back later.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union, List


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
    hinge_le_ratio: float = 0.14    # hinge line position as fraction of chord
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
        "elev_span": 0.1800018297,
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
    _FUS_FIXED_MASS = 0.250              # e.g. battery, avionics, etc.
    _BATTERY_SIZE   = (0.10, 0.05, 0.05) # box for inertia (10×5×5 cm)

    _RHO_WING = 20.0
    _RHO_ELEV = 20.0
    _RHO_RUDD = 20.0
    _RHO_PROP = 500.0

    # Lever arms as fraction of chord (from LE reference frame)
    _CG_RATIO   = 0.31
    _COLL_RATIO = 0.25

    # Inertia fudge factor + collision shrink
    _I_FUDGE = 0.6
    _SHRINK  = 0.25

    # Root offsets and fixed RPYs (legacy values kept)
    _MASS_INTER     = 0.01
    _ROOT_Y_OFFSET  = 0.05

    _RPY_FUSE_COLL = "-1.5898372930676048 6.123233995736766e-17 -1.5707963267948968"
    _RPY_WING_COLL = "1.570796327 -1.558922022 0"
    _RPY_ELEV_COLL = "1.5763749568 -1.5318707859 3.1359944429"
    _RPY_RUDD_COLL = "0.10626955814138643 0.008780081131392076 1.5717328981686811"

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
        """
        self._raw_genome: List[float] | None = None

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
                hinge_le_ratio,
                sweep_multi,
                twist_multi,
                cl_alpha_2d,
                alpha0_2d_deg,
            ) = seq

            # --- Derived physical quantities --------------------------------
            wing_chord     = wing_span / max(wing_AR, 1e-6)
            elevator_chord = elev_span / max(elev_AR, 1e-6)
            rudder_chord   = rudd_span / max(rudd_AR, 1e-6)

            fus_cg_x      = -cg_ratio * fus_length
            wing_attach_x = -attach_ratio * fus_length

            density_scale = 1.0
            prop_radius   = 0.10

            alpha0_2d_rad = alpha0_2d_deg * math.pi / 180.0
            cl_alpha_2d   = cl_alpha_2d * math.pi  # legacy scaling kept

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

        # Store resolved parameters and derived scalars
        self.p = prm
        self.out = Path(out_dir)
        self.S = prm.density_scale
        self.LE = prm.hinge_le_ratio

        # Clamp dihedral to a reasonable range (kept from original implementation)
        self.dihedral = max(-30.0, min(30.0, prm.dihedral_deg)) * math.pi / 180.0

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
            p.hinge_le_ratio,
            p.sweep_multi,
            p.twist_multi,
            p.cl_alpha_2d,
            math.degrees(p.alpha0_2d),
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

        # For wing/elevator/rudder the mesh encodes:
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

        # Box approximation of fuselage geometry
        box = (self._REF["tfus"], self._REF["hfus"], p.fus_length)
        volume = math.prod(box)

        # Mass model: structural shell + fixed payload (battery, avionics, etc.)
        m_shell = volume * self._RHO_FUS_STRUCT * self.S
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

        # Collision (shrunken box)
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
        thickness = self._REF["tw"]

        # Span region inside the propwash disc (per side)
        sp_prop = max(0.0, min(p.wing_span, max(0.0, p.prop_radius - self._ROOT_Y_OFFSET)))
        sp_free = p.wing_span - sp_prop

        rho = self._RHO_WING * self.S
        chord = p.wing_chord

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
            label: str,
            span: float,
            y_offset: float,
        ) -> None:
            """
            Create a wing panel (either in propwash or free stream).

            Parameters
            ----------
            side:
                "left" or "right".
            label:
                "prop" or "free".
            span:
                span of this panel [m].
            y_offset:
                offset from the wing root along Y [m].
            """
            name = f"{side}_wing_{label}"
            box = (thickness, chord, span)
            mass_panel = math.prod(box) * rho
            I_panel = self._I_box(mass_panel, *box)

            # Fixed joint from main wing root
            joint = ET.SubElement(robot, "joint", name=f"fixed_joint_{name}", type="fixed")
            ET.SubElement(joint, "parent", link=f"{side}_wing")
            ET.SubElement(joint, "child", link=name)
            self._origin(joint, (0.0, y_offset, 0.0))

            # Link with inertia at aerodynamic CG
            link = ET.SubElement(robot, "link", name=name)
            self._add_inertial(
                link,
                ((self._CG_RATIO - self.LE) * chord, 0.0, 0.0),
                mass_panel,
                I_panel,
            )

            # Collision geometry
            coll = ET.SubElement(link, "collision", name=f"{name}_collision")
            self._origin(
                coll,
                (self._COLL_RATIO * chord, 0.0, -0.000421042),
                self._RPY_WING_COLL,
            )
            ET.SubElement(
                ET.SubElement(coll, "geometry"),
                "box",
                size=" ".join(f"{v:.12g}" for v in box),
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
            self._origin(ja, ((self._CG_RATIO - self.LE) * chord, 0.0, 0.0), "3.141592654 0 0")

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
            limit_twist  = 0.7 / max(self.p.twist_multi, 0.5)   # [rad]
            effort_sweep = 1.0 * max(self.p.sweep_multi, 0.5)
            effort_twist = 0.5 * max(self.p.twist_multi, 0.5)

            # Attach wing root to fuselage with dihedral
            fj = ET.SubElement(robot, "joint", name=f"fixed_joint_{side}_wing", type="fixed")
            ET.SubElement(fj, "parent", link="fuselage")
            ET.SubElement(fj, "child", link=f"fuselage_{side}_0")
            self._origin(
                fj,
                (p.wing_attach_x, self._ROOT_Y_OFFSET * sign, -0.03),
                f"{self.dihedral * sign:.12g} 0 3.141592653589793",
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
            ET.SubElement(js, "dynamics", **self._REV_DYN)

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
            self._origin(jt, (0.0, -self._ROOT_Y_OFFSET * sign, 0.0))
            ET.SubElement(jt, "axis", xyz="0 1 0")
            ET.SubElement(
                jt,
                "limit",
                lower=f"{-limit_twist:.6g}",
                upper=f"{limit_twist:.6g}",
                effort=f"{effort_twist:.6g}",
                velocity=self._REV_LIMIT["velocity"],
            )
            ET.SubElement(jt, "dynamics", **self._REV_DYN)

            # Twist servo mass
            self._add_fixed_mass(
                robot,
                name=f"{side}_wing_servo_twist",
                parent=f"fuselage_{side}_1",
                xyz=(0.0, -self._ROOT_Y_OFFSET * sign, 0.0),
                m=self._SERVO_WING_TWIST_MASS,
                size=None,
                sphere_radius=0.015,
            )

            # Root link for the wing side (visual is the whole half-wing mesh)
            root_link = ET.SubElement(robot, "link", name=f"{side}_wing")
            I_eps = self._I_box(
                self._MASS_INTER,
                thickness,
                p.wing_chord,
                sp_prop + sp_free,
            )
            self._add_inertial(root_link, (0.0, 0.0, 0.0), self._MASS_INTER, I_eps)

            vis = ET.SubElement(root_link, "visual", name=f"{side}_wing_visual")
            rpy = "3.141592654 0 0" if side == "left" else "0 0 0"
            vis_shift_x = (self.LE - self._LE_REF) * chord
            self._origin(vis, (vis_shift_x, 0.0, 0.0), rpy)

            mesh = ET.SubElement(
                ET.SubElement(vis, "geometry"),
                "mesh",
                filename="package://meshes/wing0009.stl",
            )
            mesh.set("scale", self._scale("wing", (thickness, chord, sp_prop + sp_free)))
            ET.SubElement(vis, "material", name="red")

            # Two physical panels: propwash panel + free-stream panel
            if sp_prop > 0.0:
                _panel(side, "prop", sp_prop, -sign * sp_prop / 2.0)
            if sp_free > 0.0:
                _panel(side, "free", sp_free, -sign * (sp_prop + sp_free / 2.0))

        # Build right and left sides
        _wing_side("right", +1)
        _wing_side("left",  -1)

    # ────────────────────────────────────────────────────────────────────
    # Elevator (pitch) – LE-based frames for aero, legacy signs for coll
    # ────────────────────────────────────────────────────────────────────

    def _elevator(self, robot: ET.Element) -> None:
        """Create elevator joint, links, aero frames and collision geometry."""
        p = self.p

        box_full = (self._REF["tc"], p.elevator_chord, p.elevator_span)
        span_half = p.elevator_span / 2.0
        box_half = (box_full[0], box_full[1], span_half)

        rho = self._RHO_ELEV * self.S
        m_half = math.prod(box_half) * rho
        I_half = self._I_box(m_half, *box_half)

        chord = p.elevator_chord

        # Main revolute hinge (pitch)
        jh = ET.SubElement(robot, "joint", name="elevator_pitch_joint", type="revolute")
        ET.SubElement(jh, "parent", link="fuselage")
        ET.SubElement(jh, "child", link="elevator_hinge")
        self._origin(jh, (-p.fus_length - 0.03, 0.0, 0.0))
        ET.SubElement(jh, "axis", xyz="0 1 0")
        ET.SubElement(
            jh,
            "limit",
            lower="-0.35",
            upper="-0.35".replace("-", "", 1) if False else "0.35",  # keep original values
            effort="0.5",
            velocity="3.665191429",
        )
        # Note: the above line keeps the original "-0.35 / 0.35" limits.
        #       The odd construct is only to emphasize we keep behaviour.
        #       It does *not* change the resulting XML.
        jh.find("limit").set("upper", "0.35")  # ensure exact original value
        ET.SubElement(jh, "dynamics", damping="0.2", friction="0.05")

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
            self._MASS_INTER,
            self._REF["tc"],
            p.elevator_chord,
            p.elevator_span,
        )
        self._add_inertial(hinge_link, (0.0, 0.0, 0.0), self._MASS_INTER, I_hinge)

        # Whole elevator visual mesh (for convenience)
        vis_h = ET.SubElement(hinge_link, "visual")
        self._origin(vis_h, (0.0, 0.0, 0.0))
        mesh_h = ET.SubElement(
            ET.SubElement(vis_h, "geometry"),
            "mesh",
            filename="package://meshes/elevator.stl",
        )
        mesh_h.set("scale", self._scale("elevator", (self._REF["tc"], p.elevator_chord, p.elevator_span)))
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

            coll = ET.SubElement(ln, "collision")
            self._origin(coll, (-self._COLL_RATIO * chord, 0.0, -0.0226291863), self._RPY_ELEV_COLL)
            ET.SubElement(
                ET.SubElement(coll, "geometry"),
                "box",
                size=" ".join(f"{v:.12g}" for v in box_half),
            )

            ja = ET.SubElement(
                robot,
                "joint",
                name=f"fixed_joint_aero_frame_elevator_{side}",
                type="fixed",
            )
            ET.SubElement(ja, "parent", link=f"elevator_{side}")
            ET.SubElement(ja, "child", link=f"aero_frame_elevator_{side}")
            self._origin(ja, (-self._CG_RATIO * chord, 0.0, 0.0), "0 3.141592654 0")

            aero = ET.SubElement(robot, "link", name=f"aero_frame_elevator_{side}")
            self._add_inertial(aero, (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0))

    # ────────────────────────────────────────────────────────────────────
    # Rudder (yaw)
    # ────────────────────────────────────────────────────────────────────

    def _rudder(self, robot: ET.Element) -> None:
        """Create rudder yaw joint, link, aero frame and collision geometry."""
        p = self.p
        chord = p.rudder_chord

        box = (self._REF["tc"], chord, p.rudder_span)
        mass_rudder = math.prod(box) * self._RHO_RUDD * self.S
        I_rudder = self._I_box(mass_rudder, *box)

        # Rudder yaw joint (attached to elevator hinge)
        jy = ET.SubElement(robot, "joint", name="rudder_yaw_joint", type="revolute")
        ET.SubElement(jy, "parent", link="elevator_hinge")
        ET.SubElement(jy, "child", link="rudder")
        self._origin(jy, (0.0, 0.0, 0.0))
        ET.SubElement(jy, "axis", xyz="0 0 1")
        ET.SubElement(
            jy,
            "limit",
            lower="-0.35",
            upper="0.35",
            effort="0.5",
            velocity="3.665191429",
        )
        ET.SubElement(jy, "dynamics", damping="0.2", friction="0.05")

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
            size=" ".join(f"{v:.12g}" for v in box),
        )

        # Visual mesh
        vis = ET.SubElement(ln, "visual", name="rudder_visual_0")
        self._origin(vis, (0.0, 0.0, 0.0))
        mesh = ET.SubElement(
            ET.SubElement(vis, "geometry"),
            "mesh",
            filename="package://meshes/rudder.stl",
        )
        mesh.set("scale", self._scale("rudder", box))
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
        self._origin(ja, (-self._CG_RATIO * chord, 0.0, -0.08), "0 3.141592653589793 0")

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
    genome1 = [0.5, 2.5, 0.46, 0.45, 0.4, 0.3, 1.75, 0.2, 1.0, 0.0, 0.25, 3.0, 3.5, 2.0, -2.0]
    genome2 = [0.5, 3.5, 0.46, 0.45, 0.4, 0.2, 1.75, 0.2, 1.75, 0.0, 0.25, 3.0, 1.5, 2.0, -2.5]
    genome3 = [0.7, 2.5, 0.66, 0.45, 0.4, 0.26, 3.0, 0.14, 2.75, 10.0, 0.25, 2.0, 3.5, 2.0, -2.0]
    genome4 = [0.44, 1.75, 0.48, 0.34, 0.3, 0.12, 3.0, 0.2, 2.5, 20.0, 0.25, 3.5, 2.25, 2.0, -5.0]

    for genome in (genome1, genome4):
        path = UrdfMaker(genome).create_urdf()
        print("URDF written to:", path)
