"""Tools for dosetest_R_map.ipynb: probe a dose test device by device and save
each reading by block, row, column.

Call setup(...) once after connecting the SMU; then m(), skip(), redo(),
goto(), here(), show_grid(), show_iv(), grid(), table(), plot(), export().
All state (settings, cursor, SMU handle) lives in this module.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

COLS = ["time", "sample", "block", "row", "col", "dose", "R", "fit_res", "status", "note"]

# set by setup()
smu = None
SAMPLE = None
DATA_DIR = None
BLOCKS = []
N_ROWS = N_COLS = 0
ORDER = "raster"
R_SHORT = 350.0
R_OPEN = 1e6
SHOW_IV = True
SHOW_GRID = True
DOSES = None
OUT = OUT_IV = None
SITES = []
cursor = 0


def setup(smu_, sample, data_dir, blocks, n_rows, n_cols, order="raster",
          r_short=350.0, r_open=1e6, show_iv=True, show_grid=True, doses=None):
    """Store the settings, build the site list, and resume at the first unmeasured device.

    order: "raster" (down each column, columns left to right) or "snake"
           (every other column goes back up).
    doses: optional dict {(block, row, col): dose}, or a callable dose(block, row, col).
    """
    global smu, SAMPLE, DATA_DIR, BLOCKS, N_ROWS, N_COLS, ORDER, R_SHORT, R_OPEN
    global SHOW_IV, SHOW_GRID, DOSES, OUT, OUT_IV, SITES, cursor
    smu = smu_
    SAMPLE = sample
    DATA_DIR = Path(data_dir)
    BLOCKS = list(blocks)
    N_ROWS, N_COLS = int(n_rows), int(n_cols)
    ORDER = order
    R_SHORT, R_OPEN = r_short, r_open
    SHOW_IV, SHOW_GRID = show_iv, show_grid
    DOSES = doses

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT = DATA_DIR / f"{SAMPLE}_R.csv"
    OUT_IV = DATA_DIR / f"{SAMPLE}_IV.csv"   # raw I-V points of every measurement
    SITES = _sites()
    cursor = _first_open()
    here()


def dose_of(block, row, col):
    if DOSES is None:
        return np.nan
    if callable(DOSES):
        return DOSES(block, row, col)
    return DOSES.get((block, row, col), np.nan)


def _sites():
    out = []
    for b in BLOCKS:
        for c in range(1, N_COLS + 1):
            rows = range(1, N_ROWS + 1)
            if ORDER == "snake" and c % 2 == 0:
                rows = reversed(rows)
            out += [(b, r, c) for r in rows]
    return out


def _load():
    return pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLS)


def _done():
    d = _load()
    return set(zip(d.block, d.row.astype(int), d.col.astype(int)))


def _first_open():
    done = _done()
    return next((k for k, s in enumerate(SITES) if s not in done), len(SITES))


def _now():
    return dt.datetime.now().isoformat(timespec="milliseconds")


def _save(row):
    new = not OUT.exists()
    pd.DataFrame([row], columns=COLS).to_csv(OUT, mode="a", header=new, index=False)


def _status(R):
    if R < 0 or R > R_OPEN:
        return "OPEN"
    if R < R_SHORT:
        return "SHORT"
    return "OK"


def _plot_iv(iv, R, title):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(iv.V * 1e3, iv.I * 1e6, "o", ms=3)
    if np.isfinite(R) and R != 0:
        slope, icpt = np.polyfit(iv.V, iv.I, 1)
        ax.plot(iv.V * 1e3, (slope * iv.V + icpt) * 1e6, "-", lw=1)
    ax.set(xlabel=r"$V$ (mV)", ylabel=r"$I$ ($\mu$A)",
           title=f"{title}   $R$ = {R:.1f} " + r"$\Omega$")
    plt.tight_layout()
    plt.show()


def show_iv(block, row, col, all_reps=False):
    """Plot the I-V of a device: the latest reading, or every reading."""
    if not OUT_IV.exists():
        return print("No I-V data yet")
    d = pd.read_csv(OUT_IV)
    d = d[(d.block == block) & (d.row == row) & (d.col == col)]
    if d.empty:
        return print(f"No I-V for {block} r{row} c{col}")
    times = d.time.unique() if all_reps else d.time.unique()[-1:]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for t in times:
        x = d[d.time == t]
        slope = np.polyfit(x.V, x.I, 1)[0]
        ax.plot(x.V * 1e3, x.I * 1e6, "o-", ms=3, lw=0.8,
                label=f"{t[11:19]}  {1 / slope:.1f} " + r"$\Omega$")
    ax.set(xlabel=r"$V$ (mV)", ylabel=r"$I$ ($\mu$A)", title=f"{block} r{row} c{col}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def here():
    if cursor >= len(SITES):
        print("All devices measured.")
    else:
        b, r, c = SITES[cursor]
        print(f"Next: block {b}, row {r}, col {c}   ({cursor + 1}/{len(SITES)})")


def m(note=""):
    """Measure the current device, save it, move to the next."""
    global cursor
    if cursor >= len(SITES):
        return here()
    b, r, c = SITES[cursor]
    try:
        R, res = smu.iv()
    except RuntimeError:
        smu.safe()
        smu.output(True)                 # back to on at 0 V for the next landing
        smu.check_errors()               # clear the limit error
        _save([_now(), SAMPLE, b, r, c, dose_of(b, r, c), np.nan, np.nan, "SHORT", "current limit"])
        print(f"{b} r{r} c{c}: SHORT (current limit)")
        cursor += 1
        here()
        if SHOW_GRID:
            show_grid(b, last=(r, c))
        return
    st = _status(R)
    t = _now()
    _save([t, SAMPLE, b, r, c, dose_of(b, r, c), R, res, st, note])
    iv = smu.last_iv.assign(time=t, block=b, row=r, col=c)[["time", "block", "row", "col", "V", "I"]]
    iv.to_csv(OUT_IV, mode="a", header=not OUT_IV.exists(), index=False)
    flag = "" if st == "OK" else f"   <-- {st}"
    print(f"{b} r{r} c{c}: R = {R:.1f} Ohm  (fit {res * 100:.3f} %){flag}")
    cursor += 1
    here()
    if SHOW_IV:
        _plot_iv(iv, R, f"{b} r{r} c{c}  {st}")
    if SHOW_GRID:
        show_grid(b, last=(r, c))


def skip(note="skipped"):
    """Record the current device as not measured, move on."""
    global cursor
    if cursor >= len(SITES):
        return here()
    b, r, c = SITES[cursor]
    _save([_now(), SAMPLE, b, r, c, dose_of(b, r, c), np.nan, np.nan, "SKIP", note])
    print(f"{b} r{r} c{c}: skipped")
    cursor += 1
    here()
    if SHOW_GRID:
        show_grid(b, last=(r, c))


def redo(note="redo"):
    """Measure the previous device again. The newest reading is the one used."""
    global cursor
    cursor = max(cursor - 1, 0)
    m(note)


def goto(block, row, col):
    """Jump to a device."""
    global cursor
    cursor = SITES.index((block, row, col))
    here()


def table():
    """Latest reading per device."""
    d = _load()
    return d.groupby(["block", "row", "col"], as_index=False).last()


def grid(block, value="R"):
    """Block as a N_ROWS x N_COLS table, laid out like the chip."""
    d = table()
    d = d[d.block == block]
    return (d.pivot(index="row", columns="col", values=value)
             .reindex(index=range(1, N_ROWS + 1), columns=range(1, N_COLS + 1)))


def show_grid(block=None, last=None):
    """Block grid as it fills. Yellow: next device. Box: just measured. Red: short/open."""
    from IPython.display import display
    if block is None:
        block = SITES[min(cursor, len(SITES) - 1)][0]
    R, st = grid(block), grid(block, "status")
    txt = R.apply(lambda col: col.map(lambda x: "" if pd.isna(x) else f"{x:.0f}"))
    txt = txt.where(~st.isin(["SHORT", "OPEN", "SKIP"]),
                    st.replace({"SHORT": "SHORT", "OPEN": "OPEN", "SKIP": "skip"}))
    nxt = SITES[cursor] if cursor < len(SITES) else None
    n_done = st.notna().values.sum()

    def css(_):
        out = pd.DataFrame("", index=txt.index, columns=txt.columns)
        out[st.isin(["SHORT", "OPEN"])] = "color: #c00; font-weight: bold"
        out[st == "SKIP"] = "color: #999"
        if nxt and nxt[0] == block:
            out.loc[nxt[1], nxt[2]] += "; background-color: #ffe680"
        if last:
            out.loc[last[0], last[1]] += "; outline: 2px solid #1f6fd1"
        return out

    try:
        sty = (txt.style.apply(css, axis=None)
                  .set_caption(f"{SAMPLE} block {block}: R (Ohm), {n_done}/{N_ROWS * N_COLS} done")
                  .set_properties(**{"text-align": "right", "min-width": "4.5em"}))
        display(sty)
    except ImportError:                  # pandas Styler needs jinja2
        print(f"block {block}: {n_done}/{N_ROWS * N_COLS} done")
        display(txt)


def plot(block):
    """Colour map of a block. Red letters: S = short, O = open, X = skipped."""
    g = grid(block)
    st = grid(block, "status")
    ok = g.where(st == "OK")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(ok.values.astype(float), norm=LogNorm(), cmap="viridis")
    for (i, j), s in np.ndenumerate(st.values):
        if s in ("SHORT", "OPEN", "SKIP"):
            ax.text(j, i, {"SHORT": "S", "OPEN": "O", "SKIP": "X"}[s],
                    ha="center", va="center", color="r", fontsize=9)
    ax.set(xticks=range(N_COLS), xticklabels=range(1, N_COLS + 1),
           yticks=range(N_ROWS), yticklabels=range(1, N_ROWS + 1),
           xlabel="col", ylabel="row", title=f"{SAMPLE} block {block}")
    fig.colorbar(im, ax=ax, label=r"$R$ ($\Omega$)")
    plt.tight_layout()


def export():
    """Write one CSV per block laid out like the chip, plus a ddh5 of everything."""
    for b in BLOCKS:
        grid(b).to_csv(DATA_DIR / f"{SAMPLE}_{b}_grid.csv")
    try:
        from labcore.data.datadict import DataDict
        from labcore.data.datadict_storage import datadict_to_hdf5
        d = table()
        dd = DataDict(
            block=dict(values=d.block.map({b: k for k, b in enumerate(BLOCKS)}).values),
            row=dict(values=d.row.values.astype(int)),
            col=dict(values=d.col.values.astype(int)),
            R=dict(values=d.R.values.astype(float), axes=["block", "row", "col"], unit="Ohm"),
            fit_res=dict(values=d.fit_res.values.astype(float), axes=["block", "row", "col"]),
            dose=dict(values=d.dose.values.astype(float), axes=["block", "row", "col"]),
        )
        dd.add_meta("sample", SAMPLE)
        dd.add_meta("blocks", ",".join(BLOCKS))
        dd.validate()
        datadict_to_hdf5(dd, str(DATA_DIR / f"{SAMPLE}_R"))
        print("ddh5 written")
    except Exception as ex:
        print(f"ddh5 export skipped ({ex!r}); CSVs are complete")
