# Propeller Physics Model

This note documents the current propeller model used in Genesis for:

- thrust
- induced velocity
- propeller power

It also separates:

- values taken from the real vehicle setup
- values taken from dimensional propeller data
- values that remain conservative modeling choices

## 1. Core Modeling Choice

The current model is built around the standard nondimensional propeller relations:

\[
J = \frac{V_n}{nD}
\]

\[
T = \rho n^2 D^4 C_T(J)
\]

\[
P = \rho n^3 D^5 C_P(J)
\]

where:

- `J` is advance ratio
- `V_n` is the positive axial inflow seen by the propeller
- `n` is propeller speed in revolutions per second
- `D` is propeller diameter
- `rho` is air density
- `C_T(J)` is thrust coefficient
- `C_P(J)` is power coefficient

These are the standard MIT/APC dimensional-analysis definitions.

Primary references:

1. MIT Unified Engineering, propeller dimensional analysis:
   https://ocw.mit.edu/ans7870/16/16.unified/propulsionS04/UnifiedPropulsion7/UnifiedPropulsion7.htm
2. APC propeller performance database:
   https://www.apcprop.com/technical-information/performance-data/?v=1ee0bf89c5d1

The actual drone from the paper uses:

- motor: `T-Motor AT-2306 KV2300`
- battery: `2S LiPo`
- propeller: `GWS 8x4`

Reference:

3. *Sharp turning maneuvers with avian-inspired wing and tail morphing*:
   https://www.nature.com/articles/s44172-022-00035-2

## 2. What Comes From the Real Setup

These values are setup-specific and should stay tied to the actual vehicle:

- `radius = 0.1 m`
  - therefore `D = 0.2 m`
- `max_thrust = 5.2 N`
- `kV = 2300 rpm/V`
- `prop_voltage_nominal = 7.4 V`
- `prop_cutoff_hz`

These are read from:

- [actuators.csv](/home/andrea/Documents/Genesis/genesis/assets/urdf/mydrone/actuators.csv)

and parsed through:

- [drone_model.py](/home/andrea/Documents/Genesis/genesis/assets/urdf/common/drone_model.py)

## 3. What Can Reasonably Be Taken From APC Data

The APC database is not the exact propeller used in the paper, because the paper uses a `GWS 8x4`, not an `APC 8x4E`.

Still, APC is a useful proxy because the nondimensional coefficients are driven much more by:

- diameter
- pitch ratio
- general blade family / low-Re propeller class

than by motor brand.

So:

- `D`, `kV`, battery voltage, and static thrust target should come from the real setup
- `C_T(J)` and `C_P(J)` can be approximated from APC `8x4E` data as a proxy for a small `8x4` propeller

What can be taken from APC:

- `C_T(0)`
- `C_P(0)`
- the shape of `C_T(J)`
- the shape of `C_P(J)`
- efficiency trends versus `J`

What cannot be taken directly from APC:

- the exact coefficients of the paper's `GWS 8x4`
- motor electrical losses
- ESC losses
- `throttle -> rpm` mapping for the actual aircraft
- induced velocity directly

Induced velocity is still computed afterward using momentum theory.

## 4. Fitting Procedure Used for APC Proxy Data

The APC database provides tabulated points in nondimensional form, including:

- `J`
- `C_T`
- `C_P`
- efficiency

The fitting used here is intentionally simple:

1. Take APC `8x4E` points as a proxy for a small `8x4` propeller.
2. Restrict the fit to the low-to-mid `J` range that is most relevant for the current vehicle model.
3. Fit a quadratic with ordinary least squares using `numpy.polyfit`.

For the direct fits:

\[
C_T(J) \approx a_0 + a_1 J + a_2 J^2
\]

\[
C_P(J) \approx b_0 + b_1 J + b_2 J^2
\]

using:

```python
a2, a1, a0 = np.polyfit(J, Ct, 2)
b2, b1, b0 = np.polyfit(J, Cp, 2)
```

Then the `C_T` fit is converted into the normalized solver form:

\[
C_T(J) = C_{T0}(1 - c_{T1}J - c_{T2}J^2)
\]

through:

\[
C_{T0} = a_0,\qquad
c_{T1} = -\frac{a_1}{a_0},\qquad
c_{T2} = -\frac{a_2}{a_0}
\]

For `C_P`, the direct quadratic fit is valid, but the current solver uses the restricted monotone form:

\[
C_P(J) = C_{P0}(1 - c_{P1}J - c_{P2}J^2)
\]

with `c_{P1} >= 0` and `c_{P2} >= 0`.

That creates a limitation: the APC `8x4E` data show a slight initial increase in `C_P` at low `J`, so a monotone normalized fit cannot reproduce the raw APC curve exactly.

## 5. Recommended Proxy Coefficients

Using APC `8x4E` data as a proxy and fitting a simple quadratic over the practical low-to-mid `J` range gives:

\[
C_T(J) \approx 0.0806 + 0.0082J - 0.1306J^2
\]

\[
C_P(J) \approx 0.0479 + 0.0145J - 0.0312J^2
\]

These are the most direct "from the database" coefficients.

Two fit ranges were checked:

- `J <= 0.35`
- `J <= 0.43`

The `J <= 0.35` fit is the cleaner one for the current solver because:

- it stays in the practical low-to-mid operating range
- it keeps `C_T(J)` monotone in the solver form
- it avoids overfitting the higher-`J` tail where the simple quadratic shape becomes less robust

The checked direct fits were:

For `J <= 0.35`:

\[
C_T(J) \approx 0.081167 - 0.007035J - 0.085757J^2
\]

\[
C_P(J) \approx 0.048011 + 0.011648J - 0.023656J^2
\]

For `J <= 0.43`:

\[
C_T(J) \approx 0.080576 + 0.008202J - 0.130566J^2
\]

\[
C_P(J) \approx 0.047885 + 0.014538J - 0.031161J^2
\]

The `J <= 0.43` `C_T` fit has a positive linear term, which translates poorly into the current normalized solver form. For that reason the `J <= 0.35` fit is preferred for `C_T`.

However, the current code uses the normalized forms:

\[
C_T(J) = C_{T0}\max(0, 1 - c_{T1}J - c_{T2}J^2)
\]

\[
C_P(J) = C_{P0}\max(0, 1 - c_{P1}J - c_{P2}J^2)
\]

For `C_T(J)`, the preferred APC-proxy set in the current normalized solver form is:

- `prop_ct0 = 0.081`
- `prop_ct1 = 0.087`
- `prop_ct2 = 1.06`

For `C_P(J)`, the APC data are slightly awkward for the current normalized form because `C_P` rises a little at low `J` before decreasing. So there are two options:

Option A: faithful polynomial proxy from the raw APC fit

\[
C_P(J) \approx 0.0479 + 0.0145J - 0.0312J^2
\]

Option B: simple monotone approximation compatible with the current code

- `prop_cp0 = 0.049`
- `prop_cp1 = 0.0`
- `prop_cp2 = 0.015`

This monotone `C_P(J)` is less faithful than the direct polynomial, but it is simple and stable.

The monotone `C_P(J)` approximation above comes from a constrained fit over `J <= 0.43` with:

- `c_{P1} >= 0`
- `c_{P2} >= 0`

and gives a best simple approximation close to:

\[
C_P(J) \approx 0.04897(1 - 0.01457J^2)
\]

which is why `cp1` is set to zero and `cp2` is very small.

## 6. What Makes Sense to Change

### Keep these as they are tied to the real aircraft

- `radius`
- `max_thrust`
- `kV`
- `prop_voltage_nominal`
- `prop_cutoff_hz`

### Change these if you want the model closer to APC-style dimensional data

- `prop_ct0`
- `prop_ct1`
- `prop_ct2`
- `prop_cp0`
- `prop_cp1`
- `prop_cp2`

### Practical recommendation

For this project, the most sensible compromise is:

- keep real setup values from the paper
- use APC only for the nondimensional aerodynamic curve shape

That means the following values are reasonable:

- `prop_ct0 = 0.081`
- `prop_ct1 = 0.087`
- `prop_ct2 = 1.06`
- `prop_cp0 = 0.049`
- `prop_cp1 = 0.0`
- `prop_cp2 = 0.015`

This is more data-driven than the previous conservative set:

- old `Ct`: `0.088, 0.30, 1.20`
- old `Cp`: `0.043, 0.10, 0.80`

The old set was intentionally more penalizing. The APC-proxy set is closer to measured propeller behavior for a small `8x4` class propeller.

## 7. Where the Parameters Are Used

### Thrust and induced velocity

Used in:

- [simple_drone.py](/home/andrea/Documents/Genesis/genesis/engine/solvers/drones/simple_drone.py)

Key section:

- [simple_drone.py](/home/andrea/Documents/Genesis/genesis/engine/solvers/drones/simple_drone.py#L971)

The implementation does:

1. Compute no-load maximum speed:

\[
n_{no\_load,max} = \frac{k_V V_{nom}}{60}
\]

2. Compute the loaded speed that matches static thrust:

\[
n_{static,target} = \sqrt{\frac{T_{static,max}}{\rho D^4 C_{T0}}}
\]

3. Use:

\[
n_{loaded,max} = \min(n_{no\_load,max}, n_{static,target})
\]

4. Map filtered throttle to actual prop speed:

\[
n = throttle_{flt} \, n_{loaded,max}
\]

5. Build positive axial inflow:

\[
V_n = \max(-s_{prop} v_{body,prop,z}, 0)
\]

6. Compute:

\[
J = \frac{V_n}{nD}
\]

7. Compute thrust:

\[
T = \rho n^2 D^4 C_T(J)
\]

### Induced velocity

Once final thrust `T` is known, induced velocity is computed from actuator-disk momentum theory:

\[
T = 2 \rho A v_i (V_n + v_i)
\]

with:

\[
A = \pi \left(\frac{D}{2}\right)^2
\]

and therefore:

\[
v_i = \frac{-V_n + \sqrt{V_n^2 + \frac{2T}{\rho A}}}{2}
\]

This is why the model is:

- propeller dimensional analysis for `T` and `P`
- momentum theory for `v_i`

### Power

Used in:

- [power.py](/home/andrea/Documents/Genesis/src/winged_drone_train/control/power.py#L726)

The current implementation computes:

\[
P = \rho n^3 D^5 C_P(J)
\]

with the same `J = V_n / (nD)`.

## 8. Current CSV Values

The current values stored in:

- [actuators.csv](/home/andrea/Documents/Genesis/genesis/assets/urdf/mydrone/actuators.csv#L2)

are:

- `kV = 2300`
- `prop_voltage_nominal = 7.4`
- `prop_ct0 = 0.081`
- `prop_ct1 = 0.087`
- `prop_ct2 = 1.06`
- `prop_cp0 = 0.049`
- `prop_cp1 = 0.0`
- `prop_cp2 = 0.015`

Those were conservative. If the goal is to align more closely with the APC `8x4E` proxy, the `Ct/Cp` coefficients above should be updated.

## 9. Bottom Line

The best split is:

- use the paper for the real hardware values
- use APC for the nondimensional aerodynamic curve shape
- use momentum theory only for induced velocity

That is the cleanest combination of:

- physically standard dimensional analysis
- available public data
- compatibility with the current solver structure
