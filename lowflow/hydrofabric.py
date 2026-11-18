"""Hydrofabric (NGIAB subset geopackage) reader: divides, attributes, network topology.

Conventions of hydrofabric v2.2 subsets
    divides            divide_id 'cat-N', areasqkm
    divide-attributes  divide_id + model attributes (names resolved from the YAML convention)
    flowpaths          id 'wb-N', toid 'nex-M'
    nexus              id 'nex-M', toid 'wb-K' (absent or terminal at the basin outlet)
Divide 'cat-N' drains to flowpath 'wb-N'.

The upstream set U(f) of flowpath f holds every flowpath whose downstream path passes through f,
including f itself. Its divides define the drainage area of f and the area weights used to
aggregate forcing and attributes.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def find_gpkg(gauges_root, gauge_id) -> Path:
    hits = sorted((Path(gauges_root) / f"gage-{gauge_id}" / "config").glob("*.gpkg"))
    if not hits:
        raise FileNotFoundError(f"no subset gpkg for gage-{gauge_id}")
    return hits[0]


def num(x) -> int:
    """'cat-123', 'wb-123', 'nex-123' or 123 -> 123."""
    return int(str(x).split("-")[-1])


# ---- attribute names ------------------------------------------------------------------------
def resolve_attribute_columns(requested, available) -> dict:
    """Map YAML names such as 'mode.bexp_soil_layers_stag=1' onto gpkg columns such as
    'bexp_soil_layers_stag_1' (case-insensitive, several rewrites tried)."""
    low = {c.lower(): c for c in available}
    out, missing = {}, []
    for name in requested:
        rest = name.split(".", 1)[1] if "." in name else name
        for cand in (name, rest, rest.replace("=", "_"), name.replace("=", "_"), rest.replace("=", "."),
                     rest.replace("=", "")):
            if cand.lower() in low:
                out[name] = low[cand.lower()]
                break
        else:
            missing.append(name)
    if missing:
        raise KeyError(f"unresolved static attributes {missing}; available: {sorted(available)}")
    return out


# ---- basin ------------------------------------------------------------------------------------
class Basin:
    """Divides, attributes and flowpath network of one gauged NGIAB subset."""

    def __init__(self, gauges_root, gauge_id, static_attributes):
        self.gauge_id = str(gauge_id).zfill(8)
        self.gpkg = find_gpkg(gauges_root, self.gauge_id)
        con = sqlite3.connect(str(self.gpkg))
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            dv = pd.read_sql("SELECT divide_id, areasqkm FROM divides", con)
            cols = [r[1] for r in con.execute("PRAGMA table_info('divide-attributes')")]
            res = resolve_attribute_columns(static_attributes, cols)
            sel = ", ".join(["divide_id"] + [f'"{c}"' for c in res.values()])
            at = pd.read_sql(f"SELECT {sel} FROM 'divide-attributes'", con)
            at = at.rename(columns={v: k for k, v in res.items()})
            fp = pd.read_sql("SELECT id, toid FROM flowpaths", con)
            nx = pd.read_sql("SELECT id, toid FROM nexus", con) if "nexus" in tables else pd.DataFrame(columns=["id", "toid"])
        finally:
            con.close()

        dv["n"] = dv.divide_id.map(num)
        dv = dv.drop_duplicates("n").set_index("n").sort_index()
        self.divide_n = dv.index.to_numpy()                                  # numeric divide ids, sorted
        self.divide_ids = dv.divide_id.tolist()
        self.area = dv.areasqkm.to_numpy(np.float64)
        at["n"] = at.divide_id.map(num)
        at = at.drop_duplicates("n").set_index("n").reindex(self.divide_n)
        self.static_names = list(static_attributes)
        self.static = at[self.static_names].to_numpy(np.float64)            # (n_div, n_static), may hold NaN

        # flowpath network wb -> nex -> wb
        nex_to = {str(a): (str(b) if b is not None else None) for a, b in zip(nx.id, nx.toid)}
        fp = fp[fp.id.astype(str).str.startswith("wb-")]
        self.fp_n = np.array(sorted(fp.id.map(num).unique()))
        down = {}
        for a, b in zip(fp.id, fp.toid):
            nxt = nex_to.get(str(b)) if b is not None else None
            down[num(a)] = num(nxt) if (nxt and str(nxt).startswith("wb-") and num(nxt) in set(self.fp_n)) else None
        self.down = down
        up = {f: [] for f in self.fp_n}
        for f, d in down.items():
            if d is not None:
                up[d].append(f)
        self._up = up
        self._closure = {}
        self.div_pos = {n: i for i, n in enumerate(self.divide_n)}

    # upstream flowpaths of f (including f), memoized
    def upstream(self, f) -> np.ndarray:
        f = num(f)
        if f in self._closure:
            return self._closure[f]
        stack, seen = [f], []
        visited = set()
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x); seen.append(x)
            stack.extend(self._up.get(x, []))
        out = np.array(sorted(seen))
        self._closure[f] = out
        return out

    def upstream_divides(self, f) -> np.ndarray:
        """Row positions (into the divide arrays) of the divides draining to f or above."""
        return np.array([self.div_pos[n] for n in self.upstream(f) if n in self.div_pos], dtype=np.int64)

    def drainage_area(self, f) -> float:
        return float(self.area[self.upstream_divides(f)].sum())

    def outlet(self) -> int:
        """Most downstream flowpath (largest drainage area among terminal flowpaths)."""
        terminal = [f for f in self.fp_n if self.down.get(f) is None]
        return int(max(terminal, key=self.drainage_area))

    def weights(self, flowpaths) -> tuple:
        """Sparse area-weight matrix W (n_fp x n_div): row f holds A_c / A_f over c in U(f)."""
        rows, cols, vals, areas = [], [], [], []
        for i, f in enumerate(flowpaths):
            idx = self.upstream_divides(f)
            a = self.area[idx]
            rows += [i] * len(idx); cols += idx.tolist(); vals += (a / a.sum()).tolist(); areas.append(a.sum())
        return np.array(rows), np.array(cols), np.array(vals), np.array(areas)


def aggregate_static(static, names, rows, cols, vals, n_rows) -> np.ndarray:
    """Area-weighted static attributes per flowpath; circ_mean.* averaged as unit vectors.
    NaN attributes of a divide are skipped and the weights renormalized."""
    out = np.full((n_rows, len(names)), np.nan)
    for j, name in enumerate(names):
        x = static[cols, j]
        ok = np.isfinite(x)
        if name.startswith("circ_mean"):
            r = np.deg2rad(x)
            s = np.bincount(rows[ok], vals[ok] * np.sin(r[ok]), minlength=n_rows)
            c = np.bincount(rows[ok], vals[ok] * np.cos(r[ok]), minlength=n_rows)
            out[:, j] = np.rad2deg(np.arctan2(s, c)) % 360
        else:
            num_ = np.bincount(rows[ok], vals[ok] * x[ok], minlength=n_rows)
            den = np.bincount(rows[ok], vals[ok], minlength=n_rows)
            out[:, j] = np.where(den > 0, num_ / np.where(den > 0, den, 1), np.nan)
    return out
