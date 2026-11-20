"""Streamlit app: optimal exploration-drilling campaign by Expected Information Gain.
Code belongs to Stanford University
Run with:   streamlit run eig_app.py
Or as the "Sequential Decisions" half of the combined app:  streamlit run voi_app.py

The user supplies the stochastic isotherm ensemble as an .npz, the number of well
prospects, their map positions and their per-depth costs.  The app enumerates every
campaign that fits the budget, scores each by the EIG its measurements deliver about
which realization is true, and draws the winning strategy in 3D.

Self-contained on purpose: the EIG calculation needs only the temperature field, so
nothing here imports the GEOPHIRES economics used by the NPV side of the project.
"""
import io
import itertools

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp

if __name__ == "__main__":        # launched directly; voi_app.py configures the page itself
    st.set_page_config(page_title="EIG drilling strategy", layout="wide")

# ---------------------------------------------------------------- model constants
FEET_TO_METRES = 0.3048          # the published model extent is quoted in feet
SUBCELL_SIGMA = 1.2              # lateral smoothing that recovers sub-cell isotherm depth
SURFACE_TEMP_C = 20.0            # ground-surface temperature, used as a fifth knot at z = 0
CRITICAL_TEMP_C = 374.0          # temperatures are clipped at the critical point of water
DEFAULT_DEPTHS = (2000.0, 2500.0, 3000.0)
MEASUREMENT_BANDWIDTH = 5.0      # Gaussian kernel width, °C, in the measurement likelihood

# Default map positions, in grid-index units, for as many prospects as the user asks for.
DEFAULT_POSITIONS = [(14, 12), (34, 30), (5, 44), (25, 25), (5, 5),
                     (44, 8), (20, 44), (44, 44), (30, 5), (8, 25)]
# Default per-depth costs in MUSD; prospect i cycles through this list.
DEFAULT_COSTS = [(2.0, 2.6, 3.2), (3.0, 3.9, 4.8), (3.5, 4.6, 5.6)]


# ============================================================ temperature field
def isotherm_surfaces(lith_blocks, resolution, top_m, base_m, n_isotherms,
                      sigma=SUBCELL_SIGMA):
    """Depth to each isotherm, shape (realizations, nx, ny, n_isotherms).

    The blocks hold n_isotherms + 1 lithology codes, one per temperature band, so the
    isotherms are the interfaces between them: code k's surface is the depth above which
    every cell is cooler than code k + 1.  Counting cells locates that only to half a
    cell, but the resulting staircase is dither rather than noise -- neighbouring columns
    step at different lateral positions -- so light lateral smoothing recovers the smooth
    kriged surface underneath it.
    """
    surfaces = []
    for code in range(1, n_isotherms + 1):
        cells_above = (lith_blocks > code).sum(axis=-1)              # count per column
        depth = base_m - (base_m - top_m) * cells_above / resolution[2]
        if sigma:
            depth = gaussian_filter(depth, (0.0, sigma, sigma), mode="nearest")
        surfaces.append(depth)
    return np.stack(surfaces, axis=-1)


def enforce_minimum_gap(surfaces, minimum_gap):
    """Push isotherm depths apart so that d[j+1] >= d[j] + minimum_gap.

    Substituting e[j] = d[j] - j*gap turns the gap constraint into plain monotonicity,
    so the least-movement answer is the isotonic regression of e.  With a handful of
    knots the exact solution is the best of the consecutive-block partitions, which we
    enumerate.
    """
    n_knots = surfaces.shape[-1]
    shifted = (surfaces - np.arange(n_knots) * minimum_gap).reshape(-1, n_knots)
    best = shifted.copy()
    best_error = np.full(len(shifted), np.inf)
    for cut_pattern in itertools.product([0, 1], repeat=n_knots - 1):
        blocks, run = [], 1
        for cut in cut_pattern:                                       # build one partition
            if cut:
                blocks.append(run); run = 1
            else:
                run += 1
        blocks.append(run)
        candidate = np.empty_like(shifted)
        start = 0
        for block in blocks:                                          # average within a block
            candidate[:, start:start + block] = shifted[:, start:start + block].mean(
                axis=1, keepdims=True)
            start += block
        error = ((candidate - shifted) ** 2).sum(axis=1)
        feasible = np.all(np.diff(candidate, axis=1) >= -1e-9, axis=1) & (error < best_error)
        best[feasible] = candidate[feasible]
        best_error[feasible] = error[feasible]
    return best.reshape(surfaces.shape) + np.arange(n_knots) * minimum_gap


def temperature_function(knot_depths, knot_temps, critical=CRITICAL_TEMP_C):
    """Return T(z): piecewise-linear through the knots, extrapolated below, then clipped.

    Below the deepest isotherm the profile continues on the surface-anchored span-average
    gradient, so there is no depth at which the field is undefined.
    """
    shallowest = knot_depths[..., 0:1]
    deepest = knot_depths[..., -1:]
    span_gradient = (knot_temps[-1] - knot_temps[0]) / (deepest - shallowest)
    slopes = np.diff(knot_temps) / np.diff(knot_depths, axis=-1)

    def evaluate(depths):
        z = np.asarray(depths, float).reshape(*([1] * (knot_depths.ndim - 1)), -1)
        out = knot_temps[0] + slopes[..., 0:1] * (z - shallowest)
        for i in range(1, knot_temps.size - 1):                       # add one hinge per knot
            out = out + (slopes[..., i:i + 1] - slopes[..., i - 1:i]) * \
                np.maximum(z - knot_depths[..., i:i + 1], 0.0)
        out = np.where(z > deepest, knot_temps[-1] + span_gradient * (z - deepest), out)
        return np.minimum(out, critical)

    return evaluate


def depth_interpolator(temperature, z_centres, critical):
    """Return T(z) by linear interpolation down a sampled temperature volume.

    temperature is (..., nz) on the shared depth grid z_centres, which this project
    stores DESCENDING (index 0 is the deepest sample), so both are flipped to ascending
    first.  Depths outside the sampled span extrapolate on the end segment rather than
    clamping, since clamping would invent an isothermal layer.
    """
    order = np.argsort(z_centres)
    depths_up = z_centres[order]
    values_up = temperature[..., order]

    def evaluate(query):
        z = np.atleast_1d(np.asarray(query, float))
        upper = np.clip(np.searchsorted(depths_up, z), 1, len(depths_up) - 1)
        lower = upper - 1
        span = depths_up[upper] - depths_up[lower]
        weight = (z - depths_up[lower]) / span                        # outside [0,1] = extrapolation
        out = values_up[..., lower] * (1.0 - weight) + values_up[..., upper] * weight
        return np.minimum(out, critical)

    return evaluate


def isotherm_depths_from_volume(temperature, z_centres, targets):
    """Depth to each target temperature, shape (..., n_targets).

    Temperature rises monotonically with depth here, so each isotherm is a single-valued
    surface: find the first sample at or past the target and interpolate the crossing.
    Columns that never reach it extrapolate downward on their deepest segment.
    """
    order = np.argsort(z_centres)
    depths_up = z_centres[order]
    values_up = temperature[..., order]
    surfaces = []
    for target in targets:
        at_or_past = values_up >= target
        reached = at_or_past.any(axis=-1)
        first = np.argmax(at_or_past, axis=-1)                        # 0 where never reached
        upper = np.clip(np.where(reached, first, len(depths_up) - 1), 1, len(depths_up) - 1)
        lower = upper - 1
        t_low = np.take_along_axis(values_up, lower[..., None], -1)[..., 0]
        t_high = np.take_along_axis(values_up, upper[..., None], -1)[..., 0]
        gradient = np.where(np.abs(t_high - t_low) < 1e-9, np.nan, t_high - t_low)
        fraction = (target - t_low) / gradient                        # extrapolates when outside [0,1]
        depth = depths_up[lower] + fraction * (depths_up[upper] - depths_up[lower])
        surfaces.append(np.where(np.isfinite(depth), depth, depths_up[-1]))
    return np.stack(surfaces, axis=-1)


def npz_contents(file_bytes):
    """Which kind of ensemble is this? Reads the archive's name list, not its arrays."""
    try:
        keys = set(np.load(io.BytesIO(file_bytes), allow_pickle=True).files)
    except Exception:
        return "unreadable", set()
    if {"temperature", "z_centres"} <= keys:
        return "temperature", keys
    if {"lith_blocks", "resolution"} <= keys:
        return "lithology", keys
    return "unknown", keys


@st.cache_data(show_spinner="Reading the temperature model…")
def build_field_from_temperature(file_bytes):
    """Load a temperature model directly: temperature volume plus its depth grid."""
    data = np.load(io.BytesIO(file_bytes), allow_pickle=True)
    temperature = np.asarray(data["temperature"], float)
    z_centres = np.asarray(data["z_centres"], float).ravel()
    if temperature.shape[-1] != z_centres.size:
        raise ValueError(
            f"temperature's last axis is {temperature.shape[-1]} but z_centres has "
            f"{z_centres.size} entries -- the last axis must be depth")
    if temperature.ndim != 4:
        raise ValueError(
            f"temperature must be (realizations, nx, ny, nz); this array is "
            f"{temperature.ndim}-dimensional")
    # The file states its own grid resolution, so hold it to that rather than silently
    # trusting the array shape: a mismatch means the volume was reshaped wrongly.
    if "resolution" in data.files:
        stated = tuple(int(v) for v in np.asarray(data["resolution"]).ravel())
        if stated != temperature.shape[1:]:
            raise ValueError(
                f"the file states resolution {stated} but temperature's grid axes are "
                f"{temperature.shape[1:]}")

    # Prefer the file's own depth frame, falling back to the sampled extent.
    top_m = float(data["top_m"]) if "top_m" in data.files else float(z_centres.min())
    base_m = float(data["base_m"]) if "base_m" in data.files else float(z_centres.max())
    critical = float(data["critical_temp_c"]) if "critical_temp_c" in data.files \
        else CRITICAL_TEMP_C

    return dict(
        kind="temperature",
        resolution=tuple(int(v) for v in temperature.shape[1:]),
        n_realizations=temperature.shape[0],
        top_m=top_m,
        base_m=base_m,
        z_centres=z_centres,
        temperature=temperature,
        critical=critical,
    )


@st.cache_data(show_spinner="Building the temperature field…")
def build_field(file_bytes, top_ft, base_ft, isotherm_temps):
    """Load an .npz ensemble and reduce it to the knots of a temperature field.

    Returns plain arrays only: st.cache_data pickles whatever it caches, and the
    T(z) closure is not picklable, so it is attached afterwards by attach_temperature.
    """
    data = np.load(io.BytesIO(file_bytes), allow_pickle=True)
    if "lith_blocks" not in data or "resolution" not in data:
        raise KeyError("the .npz must contain 'lith_blocks' and 'resolution'")
    resolution = tuple(int(v) for v in data["resolution"])
    lith_blocks = data["lith_blocks"].reshape(-1, *resolution)
    top_m, base_m = top_ft * FEET_TO_METRES, base_ft * FEET_TO_METRES
    cell_height = (base_m - top_m) / resolution[2]

    # The isotherms are the interfaces between temperature bands, so n codes give n - 1
    # of them.  The user's list sets how many we build; anything more has no interface.
    isotherm_temps = sorted(float(t) for t in isotherm_temps)
    n_available = int(lith_blocks.max()) - 1
    if len(isotherm_temps) > n_available:
        raise ValueError(
            f"you listed {len(isotherm_temps)} isotherms but the blocks hold "
            f"{int(lith_blocks.max())} lithology codes, which define only {n_available} "
            f"interfaces -- list at most {n_available} temperatures")
    surfaces = isotherm_surfaces(lith_blocks, resolution, top_m, base_m,
                                 len(isotherm_temps))

    gapped = enforce_minimum_gap(surfaces, cell_height)
    knot_temps = np.array([SURFACE_TEMP_C] + list(isotherm_temps), float)
    knot_depths = np.concatenate(
        [np.zeros(gapped.shape[:-1] + (1,)), np.maximum(gapped, cell_height)], axis=-1)

    z_centres = base_m - (base_m - top_m) * (np.arange(resolution[2]) + 0.5) / resolution[2]
    return dict(
        kind="lithology",
        resolution=resolution,
        n_realizations=lith_blocks.shape[0],
        top_m=top_m,
        base_m=base_m,
        isotherm_temps=isotherm_temps,
        z_centres=z_centres,
        knot_depths=knot_depths,
        knot_temps=knot_temps,
    )


def attach_temperature(field):
    """Add the (unpicklable, so uncached) T(z) callable to a cached field dict.

    A lithography ensemble gives knots to interpolate between; a temperature model is
    already sampled on a depth grid, so that one interpolates down the grid instead.
    """
    field = dict(field)
    if field["kind"] == "temperature":
        field["temperature_at"] = depth_interpolator(
            field["temperature"], field["z_centres"], field["critical"])
    else:
        field["temperature_at"] = temperature_function(field["knot_depths"],
                                                       field["knot_temps"])
    return field


def sample_wells(field, positions, depths):
    """Temperatures at each (position, depth) pair: shape (realizations, n_positions, n_depths)."""
    depth_grid = np.asarray(depths, float)
    volume = field["temperature_at"](depth_grid)                      # (nr, nx, ny, n_depths)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return volume[:, xs, ys, :]


# ================================================================== EIG machinery
def weight_matrix(temperatures, bandwidth):
    """Posterior weights for every hypothetical truth, un-normalised.

    temperatures is (M, n_measurements).  Row n is the likelihood of realization n's
    readings under each realization m, so W[n, m] = prod_j K_h(T_j^(n) - T_j^(m)) with a
    Gaussian kernel.  Working in logs keeps many measurements from underflowing.
    """
    gaps = temperatures[:, None, :] - temperatures[None, :, :]
    log_weights = -0.5 * np.sum((gaps / bandwidth) ** 2, axis=2)
    log_weights -= logsumexp(log_weights, axis=1, keepdims=True)      # normalise each row
    return np.exp(log_weights)


def expected_information_gain(temperatures, bandwidth, prior_entropy, leave_one_out=False):
    """Average KL(posterior || uniform prior) over all realizations taken as truth.

    With a uniform prior over M realizations the inner KL is exactly log M - H(posterior),
    so no sampling is needed: averaging the rows of the weight matrix IS the outer
    expectation.  leave_one_out zeroes the diagonal, removing the realization's
    guaranteed self-match -- a pessimistic variant that matches a Monte Carlo estimand.
    """
    weights = weight_matrix(temperatures, bandwidth)
    if leave_one_out:
        np.fill_diagonal(weights, 0.0)
        weights = weights / weights.sum(axis=1, keepdims=True)
    safe = np.maximum(weights, 1e-300)
    entropy = -np.sum(np.where(weights > 0, weights * np.log(safe), 0.0), axis=1)
    return float(np.mean(prior_entropy - entropy))


def enumerate_campaigns(n_wells, depths, costs, budget, max_holes):
    """Every affordable set of (well, depth) pairs that drills each well at most once.

    Walks the wells in order, choosing to skip each one or drill it at one depth, and
    abandons a branch as soon as it overruns the budget.  Enumerating subsets of the
    well x depth actions instead would generate every combination before testing it --
    tens of millions at ten prospects, nearly all of them unaffordable or drilling one
    well twice.
    """
    out = []

    def walk(well, chosen, spent):
        if well == n_wells or len(chosen) == max_holes:
            return
        walk(well + 1, chosen, spent)                                 # skip this prospect
        for depth_index in range(len(depths)):
            cost = costs[well][depth_index]
            if spent + cost <= budget + 1e-9:
                chosen.append((well, depth_index))
                out.append(tuple(chosen))                             # record once, on the push
                walk(well + 1, chosen, spent + cost)
                chosen.pop()

    walk(0, [], 0.0)
    return out


@st.cache_data(show_spinner=False)
def score_campaigns(temperatures_bytes, shape, n_wells, depths, costs, budget,
                    max_holes, bandwidth):
    """Score every affordable campaign. Cached on the raw temperature array and settings."""
    temperatures = np.frombuffer(temperatures_bytes, dtype=np.float64).reshape(shape)
    n_realizations = temperatures.shape[0]
    prior_entropy = np.log(n_realizations)
    flat = temperatures.reshape(n_realizations, -1)                   # index = well*n_depths+depth
    n_depths = len(depths)

    campaigns = enumerate_campaigns(n_wells, depths, costs, budget, max_holes)
    rows = []
    for campaign in campaigns:
        columns = [w * n_depths + d for w, d in campaign]
        block = flat[:, columns]
        rows.append(dict(
            campaign=campaign,
            eig=expected_information_gain(block, bandwidth, prior_entropy),
            cost=sum(costs[w][d] for w, d in campaign),
            n_holes=len(campaign),
        ))
    rows.sort(key=lambda r: -r["eig"])
    return rows, prior_entropy


def drilling_order(temperatures, campaign, depths, bandwidth, prior_entropy):
    """Order a campaign by greedy marginal gain: the most informative hole goes first.

    EIG is a set function, so the total is the same whatever order the holes are drilled
    in.  Ordering by marginal gain is therefore a presentation choice -- but a useful one,
    because it front-loads the information and shows where the curve flattens.
    """
    n_depths = len(depths)
    flat = temperatures.reshape(temperatures.shape[0], -1)
    remaining, chosen, curve = list(campaign), [], []
    while remaining:
        scored = []
        for action in remaining:
            trial = chosen + [action]
            columns = [w * n_depths + d for w, d in trial]
            scored.append((expected_information_gain(
                flat[:, columns], bandwidth, prior_entropy), action))
        gain, action = max(scored)
        chosen.append(action)
        curve.append(gain)
        remaining.remove(action)
    return chosen, curve


# ====================================================================== plotting
def mean_isotherm_surface(field, target):
    """Mean depth at which the ensemble reaches `target` °C, shape (nx, ny).

    Computed on demand rather than precomputed at load time, so the temperature drawn
    in 3D can be chosen after the field is built.
    """
    depths = field["z_centres"]
    volume = field["temperature"] if field["kind"] == "temperature" \
        else field["temperature_at"](depths)
    return isotherm_depths_from_volume(volume, depths, [target])[..., 0].mean(axis=0)


def strategy_figure(field, positions, well_names, depths, ordered_campaign, surfaces):
    """3D view of the campaign: one slider step per hole drilled, in the chosen order.

    surfaces is a list of (temperature, depth-surface) pairs to stack behind the wells,
    each flat-coloured by its own temperature on a shared scale, or empty for none.
    """
    resolution = field["resolution"]
    xs = np.arange(resolution[0])
    ys = np.arange(resolution[1])
    traces, step_first_trace = [], []

    # Depth to an isotherm is single-valued because T rises monotonically with depth, so
    # each one plots directly as z = D(x, y).  Flat-colouring by temperature rather than
    # by depth means one shared colourbar reads as the temperature scale.
    if len(surfaces) == 1:
        # One isotherm: a temperature colourbar would be a single value, so colour the
        # surface by its own depth instead, which is the informative quantity.
        temperature, surface_depth = surfaces[0]
        traces.append(go.Surface(
            x=xs, y=ys, z=surface_depth.T, colorscale="Viridis_r", opacity=0.55,
            hoverinfo="skip", showlegend=False,
            colorbar=dict(title=dict(text=f"depth to<br>{temperature:.0f} °C (m)",
                                     side="right"),
                          len=0.5, thickness=13, x=0.9)))
    elif surfaces:
        # Several isotherms stacked: flat-colour each by its own temperature so the one
        # shared colourbar reads as the temperature scale.
        band_temps = [t for t, _ in surfaces]
        low, high = min(band_temps), max(band_temps)
        for index, (temperature, surface_depth) in enumerate(surfaces):
            traces.append(go.Surface(
                x=xs, y=ys, z=surface_depth.T,
                surfacecolor=np.full_like(surface_depth.T, float(temperature)),
                colorscale="Plasma", cmin=low, cmax=high,
                opacity=0.45, showscale=(index == 0), hoverinfo="skip",
                showlegend=False,
                colorbar=dict(title=dict(text="°C", side="top"), len=0.62,
                              thickness=13, x=0.9, tickvals=band_temps)))

    # Every prospect as an open candidate marker at its collar, one trace each so the
    # legend names them, plus a single text trace for the labels.
    for well, (x, y) in enumerate(positions):
        traces.append(go.Scatter3d(
            x=[x], y=[y], z=[field["top_m"]], mode="markers",
            name=f"{well_names[well]} (candidate)",
            marker=dict(size=7, color="rgba(0,0,0,0)",
                        line=dict(width=2, color="#6b6b6b")),
            hovertemplate=f"{well_names[well]} ({x}, {y})<extra></extra>"))
    traces.append(go.Scatter3d(
        x=[p[0] for p in positions], y=[p[1] for p in positions],
        z=[field["top_m"]] * len(positions), mode="text", text=well_names,
        # No explicit colour: it has to stay legible on the app's dark chart theme.
        textposition="top center", textfont=dict(size=13),
        showlegend=False, hoverinfo="skip"))
    static_traces = len(traces)

    palette = ["#c0392b", "#1f6fb4", "#0f8a6a", "#e67e22", "#8e44ad",
               "#d35400", "#2c3e50", "#27ae60", "#c2185b", "#00838f"]
    for step, (well, depth_index) in enumerate(ordered_campaign):
        step_first_trace.append(len(traces))
        x, y = positions[well]
        target = depths[depth_index]
        colour = palette[well % len(palette)]
        traces.append(go.Scatter3d(                                   # the hole itself
            x=[x, x], y=[y, y], z=[field["top_m"], target], mode="lines",
            line=dict(color=colour, width=10), showlegend=False,
            hoverinfo="skip"))
        traces.append(go.Scatter3d(                                   # marker at total depth
            x=[x], y=[y], z=[target], mode="markers",
            marker=dict(size=6, color=colour, symbol="diamond"),
            name=f"{step + 1}. {well_names[well]} @ {target:.0f} m",
            hovertemplate=f"{well_names[well]} @ {target:.0f} m<extra></extra>"))
        # The depth label is anchored a little above or below the marker and alternates
        # sides.  Two prospects far apart on the map can still project onto nearly the
        # same screen point, and without these lanes their labels overprint.
        lane = (step % 3 - 1) * 110.0
        traces.append(go.Scatter3d(
            x=[x], y=[y], z=[target + lane], mode="text",
            # Just the depth: the well name is already on the collar above it, and the
            # full "W2 @ 2000 m" collides with its neighbour when prospects sit close.
            text=[f"{target:.0f} m"],
            textposition="middle right" if step % 2 == 0 else "middle left",
            textfont=dict(size=11, color=colour),
            showlegend=False, hoverinfo="skip"))

    figure = go.Figure(data=traces)
    n_traces = len(traces)
    traces_per_hole = 3                                               # stick, marker, label
    steps = []
    for step in range(len(ordered_campaign) + 1):                     # step 0 = nothing drilled
        visible = [True] * static_traces + [False] * (n_traces - static_traces)
        for already in range(step):
            for offset in range(traces_per_hole):
                visible[step_first_trace[already] + offset] = True
        label = "no wells" if step == 0 else f"{step} hole{'s' if step > 1 else ''}"
        steps.append(dict(method="update", label=label,
                          args=[{"visible": visible},
                                {"title": f"Drilling strategy — {label}"}]))
    # A little headroom above the collars so the well labels are not clipped.
    span = field["base_m"] - field["top_m"]
    # No template or explicit colours here: Streamlit only applies its own (dark) chart
    # theme to figures that do not set one, and the surrounding page is dark.
    figure.update_layout(
        sliders=[dict(active=len(ordered_campaign), steps=steps,
                      currentvalue=dict(prefix="drilled: "), pad=dict(t=30))],
        legend=dict(x=0.01, y=0.99, font=dict(size=11)),
        scene=dict(
            xaxis=dict(title="X (W → E)", showspikes=False, zeroline=False),
            yaxis=dict(title="Y (S → N)", showspikes=False, zeroline=False),
            # The range is already descending, which reverses the axis -- adding
            # autorange="reversed" on top of it makes plotly discard the range.
            zaxis=dict(title="depth (m)", showspikes=False, zeroline=False,
                       range=[field["base_m"], field["top_m"] - 0.03 * span]),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.85),
            # Far enough back that the axis titles at the bottom corners are not clipped.
            camera=dict(eye=dict(x=1.45, y=-1.45, z=0.7))),
        # Keep the camera where the user left it when the slider restyles the traces.
        uirevision="strategy",
        # The side margins carry the X and Y axis titles, which sit at the bottom
        # corners of the box and clip against the canvas edge without them.
        showlegend=True, margin=dict(l=55, r=55, t=40, b=55), height=520,
        title=dict(text=f"Drilling strategy — {len(ordered_campaign)} holes", y=0.97))
    return figure


def map_figure(field, positions, well_names, map_depth):
    """Top view of the prospect locations over the temperature spread at one depth.

    This is a sanity check on where the wells sit, so it shows no policy -- the
    background is the standard deviation of temperature across realizations, which is
    where the ensemble disagrees and therefore where a measurement has most to resolve.
    """
    temperature = field["temperature_at"](np.array([map_depth]))[..., 0]
    spread = temperature.std(axis=0)
    figure = go.Figure(go.Heatmap(
        z=spread.T, colorscale="Inferno",
        colorbar=dict(title="sd of T (°C)")))
    for well, (x, y) in enumerate(positions):
        figure.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", showlegend=False,
            text=[f"{well_names[well]} ({x}, {y})"], hoverinfo="text",
            marker=dict(size=13, color="white", symbol="circle",
                        line=dict(width=2, color="black"))))
        # Labels go in annotations rather than marker text so they can carry a dark
        # backing box: plain white text disappears over the bright end of the colourscale.
        above = y < field["resolution"][1] * 0.85
        figure.add_annotation(
            x=x, y=y, text=well_names[well], showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor="rgba(0,0,0,0.55)", borderpad=2,
            yshift=15 if above else -15, yanchor="bottom" if above else "top")
    figure.update_layout(
        title=f"Temperature spread across realizations at {map_depth:.0f} m",
        height=520, margin=dict(l=0, r=0, t=50, b=0),
        # constrain="domain" keeps the cells square by shrinking the plotting area
        # instead of padding the ranges out with empty space on both sides.
        xaxis=dict(title="X (W → E)", range=[-0.5, field["resolution"][0] - 0.5],
                   constrain="domain"),
        yaxis=dict(title="Y (S → N)", range=[-0.5, field["resolution"][1] - 0.5],
                   scaleanchor="x", scaleratio=1, constrain="domain"))
    return figure


# ========================================================================== app
st.title("Exploration drilling by Expected Information Gain")
st.caption(
    "Pick the campaign that tells you the most about which subsurface realization is "
    "true, subject to a drilling budget.  EIG here is the expected KL divergence from "
    "the uniform prior over realizations to the posterior produced by the measurements.")

with st.sidebar:
    st.header("Upload File")
    uploaded = st.file_uploader("Ensemble (.npz)", type="npz")
    st.caption("Please upload a temperature model. The file should contain the "
               "temperature values, the depth of each sample, and the grid resolution.")

if uploaded is None:
    st.info("Upload an .npz ensemble in the sidebar to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
kind, keys = npz_contents(file_bytes)
if kind in ("unreadable", "unknown"):
    st.error(
        "Could not recognise this .npz. It needs either `temperature` + `z_centres` "
        "(a temperature model) or `lith_blocks` + `resolution` (a lithology ensemble). "
        + (f"This file holds: {', '.join(sorted(keys))}." if keys else ""))
    st.stop()

with st.sidebar:
    if kind == "lithology":
        # The lithology blocks carry no depth frame, so it has to be supplied.  These
        # interface temperatures define the field itself, unlike the one drawn in 3D.
        top_ft = st.number_input("Model top (ft)", value=1689.0, step=10.0)
        base_ft = st.number_input("Model base (ft)", value=14000.0, step=100.0)
        isotherm_text = st.text_input("Isotherm temperatures (°C)",
                                      value="125, 150, 175, 200",
                                      help="One per lithology interface, coolest first.")
        try:
            isotherm_temps = [float(v) for v in isotherm_text.replace(",", " ").split()]
        except ValueError:
            st.error("Could not read the isotherm list — give comma-separated numbers.")
            st.stop()
        if not isotherm_temps:
            st.error("Give at least one isotherm temperature.")
            st.stop()
    else:
        st.caption("Depth frame read from the file.")

try:
    if kind == "temperature":
        field = attach_temperature(build_field_from_temperature(file_bytes))
    else:
        field = attach_temperature(
            build_field(file_bytes, top_ft, base_ft, tuple(isotherm_temps)))
except Exception as exc:                                              # bad file, wrong keys, ...
    st.error(f"Could not build the temperature field: {exc}")
    st.stop()

resolution = field["resolution"]

with st.sidebar:
    st.header("Prospects")
    n_wells = st.number_input("Number of wells", min_value=1, max_value=10, value=3, step=1)
    depth_text = st.text_input("Candidate depths (m)", value="2000, 2500, 3000")
    st.header("Budget")
    budget = st.number_input("Budget (MUSD)", min_value=0.0, value=8.0, step=0.5)

bandwidth = MEASUREMENT_BANDWIDTH     # fixed kernel width in the likelihood
max_holes = int(n_wells)              # a campaign may drill every prospect

try:
    depths = tuple(float(v) for v in depth_text.replace(",", " ").split())
except ValueError:
    st.error("Could not read the depth list — give comma-separated numbers.")
    st.stop()
if not depths:
    st.error("Give at least one candidate depth.")
    st.stop()

n_wells = int(n_wells)
st.subheader("Prospect positions and costs")
st.caption("Positions are grid indices. Cost is the all-in cost of drilling that well "
           "to that depth, in MUSD.")
table = []
for i in range(n_wells):
    x, y = DEFAULT_POSITIONS[i % len(DEFAULT_POSITIONS)]
    cost_row = DEFAULT_COSTS[i % len(DEFAULT_COSTS)]
    row = {"well": f"W{i + 1}",
           "x": min(x, resolution[0] - 1),
           "y": min(y, resolution[1] - 1)}
    for j, depth in enumerate(depths):
        # Beyond the three default depths, scale the last known cost with depth.
        row[f"{depth:.0f} m"] = (float(cost_row[j]) if j < len(cost_row)
                                 else round(cost_row[-1] * depth / depths[len(cost_row) - 1], 2))
    table.append(row)

edited = st.data_editor(
    table, hide_index=True, use_container_width=True,
    column_config={
        "well": st.column_config.TextColumn("well", width="small"),
        "x": st.column_config.NumberColumn("x", min_value=0, max_value=resolution[0] - 1, step=1),
        "y": st.column_config.NumberColumn("y", min_value=0, max_value=resolution[1] - 1, step=1),
        **{f"{d:.0f} m": st.column_config.NumberColumn(f"{d:.0f} m", min_value=0.0,
                                                       step=0.1, format="%.2f") for d in depths}},
    key="prospect_table")

well_names = [str(row["well"]) for row in edited]
positions = [(int(row["x"]), int(row["y"])) for row in edited]
costs = [[float(row[f"{d:.0f} m"]) for d in depths] for row in edited]
cheapest = min(min(c) for c in costs)
if budget < cheapest:
    st.warning(f"The budget of {budget:.2f} MUSD will not pay for any hole — the cheapest "
               f"is {cheapest:.2f} MUSD.")
    st.stop()

temperatures = sample_wells(field, positions, depths)                 # (M, n_wells, n_depths)

# Scoring is exhaustive, so check the size of the search before committing to it.
cost_tuples = tuple(tuple(c) for c in costs)
n_campaigns = len(enumerate_campaigns(n_wells, depths, cost_tuples,
                                      float(budget), int(max_holes)))
if n_campaigns > 200_000:
    st.error(
        f"{n_campaigns:,} campaigns fit this budget — too many to score exhaustively. "
        "Lower the budget, cut the maximum number of holes, or offer fewer depths.")
    st.stop()
if n_campaigns > 20_000:
    st.info(f"Scoring {n_campaigns:,} campaigns; this may take a few seconds.")

rows, prior_entropy = score_campaigns(
    np.ascontiguousarray(temperatures, dtype=np.float64).tobytes(), temperatures.shape,
    n_wells, depths, cost_tuples, float(budget), int(max_holes), float(bandwidth))

if not rows:
    st.warning("No campaign fits the budget.")
    st.stop()


best = rows[0]
ordered, curve = drilling_order(temperatures, best["campaign"], depths,
                               float(bandwidth), prior_entropy)

# The map comes first, straight under the inputs, so the prospect positions just typed
# into the table can be checked against the temperature field before anything else.
st.subheader("Map view")
map_depth = st.select_slider("Depth for the temperature map (m)",
                             options=[f"{d:.0f}" for d in depths],
                             value=f"{depths[len(depths) // 2]:.0f}")
st.plotly_chart(
    map_figure(field, positions, well_names, float(map_depth)),
    use_container_width=True)
st.caption("Well locations over the standard deviation of temperature across "
           "realizations. Bright areas are where the ensemble disagrees most.")

st.subheader("Optimal strategy")
st.markdown(f"**EIG of optimal strategy: {best['eig']:.4f}**")
# Written in the same sequence the 3D view reveals and the table below lists, so the
# three read as one policy.  The total EIG is the same in any order.
st.markdown("**Policy:** " + " → ".join(
    f"{well_names[w]}@{depths[d]:.0f}" for w, d in ordered))
st.markdown(f"**Cost:** {best['cost']:.2f} MUSD")
if best["eig"] > 0.995 * prior_entropy:
    st.warning(
        "The EIG is at its maximum, so the leading campaigns are not really "
        "distinguishable — the measurements already identify the realization. Cut the "
        "budget or use a larger ensemble to get a meaningful ranking.")

order_rows = []
previous = 0.0
for (well, depth_index), running in zip(ordered, curve):
    order_rows.append({
        "step": len(order_rows) + 1,
        "well": well_names[well],
        "depth (m)": f"{depths[depth_index]:.0f}",
        "cost (MUSD)": f"{costs[well][depth_index]:.2f}",
        "cumulative EIG": f"{running:.4f}",
        "marginal gain": f"{running - previous:.4f}"})
    previous = running
st.dataframe(order_rows, hide_index=True, use_container_width=True)

st.subheader("3D strategy")
controls = st.columns([2, 1])
isotherm_to_draw = controls[0].number_input(
    "Isotherm to draw (°C)", min_value=0.0, value=200.0, step=25.0, format="%.0f",
    help="The surface is the mean depth at which the model reaches this temperature.")
show_isotherm = controls[1].checkbox("Show isotherm surface", value=True)

surfaces = []
if show_isotherm:
    band_surface = mean_isotherm_surface(field, float(isotherm_to_draw))
    # A hot isotherm can sit below the modelled base, where the depth axis clips it.
    share_below = float((band_surface > field["base_m"]).mean())
    if share_below > 0.02:
        st.warning(
            f"{100 * share_below:.0f}% of the {isotherm_to_draw:.0f} °C surface lies "
            f"below the modelled base at {field['base_m']:.0f} m, so that part is "
            "extrapolated and not drawn. Pick a cooler isotherm.")
    else:
        surfaces.append((float(isotherm_to_draw), band_surface))
st.plotly_chart(
    strategy_figure(field, positions, well_names, depths, ordered, surfaces),
    use_container_width=True)
st.caption("Collars sit at the model top; each stick runs down to that hole's total "
           "depth. Drag the slider to add the holes in drilling order.")
