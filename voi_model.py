"""Shared Cape Modern VOI model: temperature field + GEOPHIRES economics.

Imported by Sequential_Drilling.ipynb, Bickel_Sequential_DP.ipynb and
approximate_methods.ipynb so all three run off one definition of the model.
"""
import numpy as np, os, contextlib, functools, itertools
from scipy.ndimage import gaussian_filter
import numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

NPZ      = 'stochastic_daa-generated_5_iso_Fercho.npz'
TOP_FT   = 1689.0
BASE_FT  = 14000.0
ISO_T    = [125, 150, 175, 200]
GRAD_MIN, GRAD_MAX = 0.020, 0.200      # degC/m  (20-200 degC/km)
T_MIN,   T_MAX     = 20.0,  400.0      # clip extrapolation
SUBCELL_SIGMA      = 1.2               # lateral smoothing for sub-cell surfaces; 0 disables

_d   = np.load(NPZ, allow_pickle=True)
RES  = tuple(int(v) for v in _d['resolution'])
lith = _d['lith_blocks'].reshape(-1, *RES)
NR   = lith.shape[0]
TOP, BASE = TOP_FT*0.3048, BASE_FT*0.3048
CELL = (BASE-TOP)/RES[2]
z_centres = BASE - (BASE-TOP)*(np.arange(RES[2])+0.5)/RES[2]

def iso_depth_raw(block, k):
    """Isotherm depth from cell counting -> quantised to the 75 m cell height."""
    return BASE - (BASE-TOP)*(block > k).sum(axis=-1)/RES[2]

def iso_surface(k, sigma=SUBCELL_SIGMA, block=None):
    """Sub-cell isotherm depth (nr, nx, ny).

    Cell counting locates the surface only to +-half a cell (RMS ~21.7 m here).
    But the staircase is dither, not noise: neighbouring columns cross the cell
    boundary at different lateral positions, so the step pattern encodes where the
    true surface lies in between.  GemPy surfaces are kriging output and therefore
    smooth at the covariance range, so light lateral smoothing recovers that.
    On a synthetic surface digitised the same way this cuts RMS error from
    21.5 m to 6.0 m.  Set sigma=0 to fall back to the raw quantised estimate.
    """
    b = lith if block is None else block
    dep = iso_depth_raw(b, k)
    if sigma:
        dep = gaussian_filter(dep, (0,)*(dep.ndim-2) + (sigma, sigma), mode='nearest')
    return dep

SURF = np.stack([iso_surface(k) for k in (1,2,3,4)], axis=-1)   # (nr, nx, ny, 4)

COLS = ['#2c5f8a','#5ba3cf','#f2d06b','#e8873a','#c0392b']
LBL  = ['< 125 °C','125–150 °C','150–175 °C','175–200 °C','> 200 °C']
cmap, norm = ListedColormap(COLS), BoundaryNorm(np.arange(.5,6.5),5)

def plot_sections(y=25, reals=(0,1,2,3)):
    fig = plt.figure(figsize=(15.5,9)); gs = fig.add_gridspec(2,3,hspace=.34,wspace=.3)
    def _ax(ax, ft=False):
        ax.set_ylim(BASE,TOP); ax.set_xlim(0,RES[0]-1)
        ax.set_ylabel('depth (m)'); ax.set_xlabel('model x  (W $\\rightarrow$ E)')
        if ft:
            t=ax.twinx(); t.set_ylim(BASE/.3048,TOP/.3048)
            t.set_ylabel('depth (ft)',fontsize=8); t.tick_params(labelsize=7)
    for i,(r,pos) in enumerate(zip(reals,[(0,0),(0,1),(0,2),(1,0)])):
        ax=fig.add_subplot(gs[pos])
        ax.imshow(lith[r,:,y,:].T[::-1],cmap=cmap,norm=norm,aspect='auto',
                  extent=[0,RES[0]-1,BASE,TOP],interpolation='nearest')
        for j,T in enumerate(ISO_T):
            ax.plot(np.arange(RES[0]), SURF[r,:,y,j], color='k', lw=1.4)
        _ax(ax, ft=(i==2)); ax.set_title(f'realization {r}  (y={y})',fontsize=10)
    ax=fig.add_subplot(gs[1,1:])
    for j,(T,c) in enumerate(zip(ISO_T,['#2e86c1','#16a085','#e67e22','#c0392b'])):
        dm=SURF[:,:,y,j]; mu,sd=dm.mean(0),dm.std(0)
        ax.plot(mu,color=c,lw=2,label=f'{T} °C')
        ax.fill_between(np.arange(RES[0]),mu-sd,mu+sd,color=c,alpha=.22)
    _ax(ax,ft=True); ax.grid(alpha=.25,ls=':'); ax.legend(fontsize=8,ncol=4,loc='lower left')
    ax.set_title(f'Isotherm depth, {NR} realizations (mean $\\pm$ 1$\\sigma$)',fontsize=10)
    h=[Line2D([0],[0],marker='s',ls='',mfc=c,mec='none',ms=11) for c in COLS]
    fig.legend(h,LBL,loc='lower center',ncol=5,frameon=False,fontsize=9,bbox_to_anchor=(.5,-.005))
    return fig

def temperature_volume(surf):
    """Continuous T from the four sub-cell isotherm depths, evaluated at any depth.

    surf : (..., 4) isotherm depths, ascending.  Piecewise-linear in depth
    (hinge form) with linear extrapolation beyond the outermost isotherms.
    """
    T   = np.asarray(ISO_T, float)
    seg = np.diff(surf, axis=-1)
    seg = np.where(np.abs(seg) < 1e-6, np.nan, seg)
    s   = np.diff(T)/seg
    s   = np.where(np.isfinite(s), s, np.nanmedian(s))
    s   = np.clip(s, GRAD_MIN, GRAD_MAX)
    def at(z):
        z   = np.asarray(z, float)
        z   = z.reshape(*([1]*(surf.ndim-1)), -1)
        out = T[0] + s[...,0:1]*(z - surf[...,0:1])
        for i in (1,2):
            out = out + (s[...,i:i+1]-s[...,i-1:i])*np.maximum(z - surf[...,i:i+1], 0)
        return np.clip(out, T_MIN, T_MAX)
    return at

T_at = temperature_volume(SURF)        # callable: T_at(depth_in_metres)
Tvol  = T_at(z_centres)                # (nr, nx, ny, nz) sampled on the model grid
Tmean, Tstd = Tvol.mean(0), Tvol.std(0)

import warnings
from itertools import product

# ============================ ASSUMPTIONS ====================================
# 1. TOP_FT / BASE_FT  : model spans 1689-14000 ft (515-4267 m).  Base is Figure 2's
#                        axis limit; top is a least-squares fit of the block's
#                        interface depths against Figure 2's contours.
# 2. ISO_T             : the four isotherms are 125/150/175/200 °C (fitted; the
#                        only physically possible option - see the depth-frame cell).
# 3. SUBCELL_SIGma     : 1.2-cell lateral smoothing to recover isotherm depth below
#                        the 75 m quantisation (cuts RMS error 21.5 -> 6.0 m on a
#                        synthetic test).
# 4. one-cell gap      : enforce_gap() projects isotherm depths onto
#                        d[j+1] >= d[j] + CELL.  Two isotherms cannot share a depth;
#                        this bounds every implied gradient by dT/CELL = 333 °C/km.
# 5. T_SURFACE_C = 20  : ground-surface temperature, added as a FIFTH knot at depth 0.
#                        Makes everything above the 125 °C isotherm interpolation and
#                        anchors the gradient (median 56.6 °C/km vs ~56 from Acord-1).
# 6. UNLIMITED down-hole extrapolation below the 200 °C isotherm, on the
#                        surface-anchored span-average gradient.
# 7. T_CRIT_C = 374    : temperatures are CLIPPED to the critical point of water
#                        (373.95 °C).  Above it a binary/flash plant model is not
#                        meaningful, so the ceiling is physical rather than arbitrary.
#                        Consequence: NO NaN anywhere.  But NPV is convex in T, so
#                        cells sitting at the ceiling carry the most optimistic
#                        economics the model can express - read the "at ceiling"
#                        fraction next to any NPV.
# =============================================================================
T_SURFACE_C = 20.0
T_CRIT_C    = 374.0
WINDOW      = (2000., 3500.)

def _compositions(n):
    out=[]
    for cuts in product([0,1], repeat=n-1):
        blocks=[]; cur=1
        for c in cuts:
            if c: blocks.append(cur); cur=1
            else: cur+=1
        blocks.append(cur); out.append(tuple(blocks))
    return out

def enforce_gap(surf, min_gap=None):
    """Project isotherm depths onto d[j+1] >= d[j] + min_gap (default one cell).

    Minimal-movement (least-squares) projection: substituting e[j]=d[j]-j*gap turns
    the gap constraint into monotonicity of e, so the answer is the isotonic
    regression of e; with four knots the exact solution comes from enumerating the
    8 consecutive-block partitions.  RMS movement 19.6 m, 23% of columns touched,
    ordering preserved.
    """
    gap = CELL if min_gap is None else min_gap
    n   = surf.shape[-1]
    flat = (surf - np.arange(n)*gap).reshape(-1, n)
    best, best_sse = flat.copy(), np.full(len(flat), np.inf)
    for blocks in _compositions(n):
        cand = np.empty_like(flat); k = 0
        for b in blocks:
            cand[:, k:k+b] = flat[:, k:k+b].mean(axis=1, keepdims=True); k += b
        sse  = ((cand-flat)**2).sum(axis=1)
        take = np.all(np.diff(cand,axis=1) >= -1e-9, axis=1) & (sse < best_sse)
        best[take] = cand[take]; best_sse[take] = sse[take]
    return best.reshape(surf.shape) + np.arange(n)*gap

SURF_GAP = enforce_gap(SURF)
KNOT_T   = np.array([T_SURFACE_C] + list(ISO_T), float)
KNOT_D   = np.concatenate([np.zeros(SURF_GAP.shape[:-1]+(1,)),
                           np.maximum(SURF_GAP, CELL)], axis=-1)

def temperature_fn(knot_d=KNOT_D, knot_t=KNOT_T, t_crit=T_CRIT_C):
    """T(depth) through five knots: 20 °C at surface + the four isotherms.

    Interpolates between knots; below the 200 °C isotherm extrapolates without
    limit on the surface-anchored span-average gradient; clips at the critical
    point of water.  Returns no NaN.
    """
    d0, d1 = knot_d[...,0:1], knot_d[...,-1:]
    g = (knot_t[-1]-knot_t[0])/(d1-d0)
    s = np.diff(knot_t)/np.diff(knot_d, axis=-1)
    def at(z):
        zz = np.asarray(z, float).reshape(*([1]*(knot_d.ndim-1)), -1)
        out = knot_t[0] + s[...,0:1]*(zz - knot_d[...,0:1])
        for i in range(1, knot_t.size-1):
            out = out + (s[...,i:i+1]-s[...,i-1:i])*np.maximum(zz - knot_d[...,i:i+1], 0)
        out = np.where(zz > d1, knot_t[-1] + g*(zz-d1), out)
        return np.minimum(out, t_crit)
    return at

def at_ceiling(T, t_crit=T_CRIT_C):
    """Fraction/mask of values pinned at the critical-point ceiling."""
    return np.asarray(T) >= t_crit - 1e-6

T_at  = temperature_fn()
Tvol  = T_at(z_centres)
Tmean, Tstd = Tvol.mean(0), Tvol.std(0)


# ------------------------------------------------------------------ economics
from geophires_x_client import GeophiresXClient
from geophires_x_client.geophires_input_parameters import GeophiresInputParameters

DEPTHS  = (2000., 2500., 3000.)            # m
PREM    = {2000.: 0., 2500.: 2., 3000.: 5.}   # depth premium, MUSD
CY      = {2000.: 1, 2500.: 2, 3000.: 3}      # construction years
P0, P1  = "0.12", "0.30"                      # $/kWh start / end
GAMMA   = 0.9
M       = Tvol.shape[0]                       # realizations

@functools.lru_cache(maxsize=200000)
def geo(g, dkm, cy):
    """GEOPHIRES-X: (gradient C/km, depth km, construction yrs) -> (NPV, drill cost) MUSD."""
    par = {"Gradient 1": g, "Reservoir Depth": dkm, "End-Use Option": 1,
           "Number of Production Wells": 2, "Number of Injection Wells": 2,
           "Construction Years": cy, "Starting Electricity Sale Price": P0,
           "Ending Electricity Sale Price": P1}
    try:
        with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn), \
             contextlib.redirect_stderr(dn):
            r = GeophiresXClient().get_geophires_result(GeophiresInputParameters(par))
        npv = dr = np.nan
        for ln in open(r.output_file_path):
            lc = ln.casefold()
            if 'project npv' in lc:
                npv = float(''.join(c for c in ln.split(':')[1] if c.isdigit() or c in '.-'))
            elif 'drilling and completion costs:' in lc:
                dr = float(''.join(c for c in ln.split(':')[1] if c.isdigit() or c in '.-'))
        return npv, dr
    except Exception:
        return np.nan, np.nan

DRILL = {d: geo(round((200-15.)/(d/1000.), 2), round(d/1000., 3), CY[d])[1] for d in DEPTHS}
C_DRILL = {z: DRILL[z] + PREM[z] for z in DEPTHS}     # 11.2 / 15.6 / 21.4 MUSD

def npv_of(T, z):
    """NPV (MUSD) for an observed TD temperature T at depth z. Vectorised over T."""
    T = np.atleast_1d(np.asarray(T, float))
    out = np.array([geo(round((t-15.)/(z/1000.), 2), round(z/1000., 3), CY[z])[0] for t in T])
    return np.nan_to_num(out, nan=-1e3) - PREM[z]

WELLS_BASE     = {'W1': (14, 12), 'W2': (34, 30), 'W3': (5, 44)}
WELLS_ADJACENT = {'W1': (14, 12), 'W2': ( 8, 42), 'W3': (5, 44)}
