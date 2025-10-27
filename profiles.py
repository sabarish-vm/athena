import numpy as np
from matplotlib.axes import Axes
from numpy import float64, typing as npt
import pathlib
from natsort import os_sorted
import typing as tp
import yt
import xarray as xr
from joblib import Parallel, delayed
from unitconvert.custom import UnitSystem
from astropy import units as au
import astropy.constants as ac
import re


class Profiles:
    def __init__(self, dirpath: str, units: dict = {}, drop_angles=False) -> None:
        self.dirpath: pathlib.Path = pathlib.Path(dirpath)
        self.files: tp.List[pathlib.Path] = os_sorted(
            [i for i in self.dirpath.glob("*.athdf")]
        )
        self.units = units
        self.drop_angles: bool = drop_angles

    def setup_units(self) -> None:
        m_cgs: float = 0
        l_cgs: float = 0
        t_cgs: float = 0
        with open("../inputs/hydro/athinput.bondi") as f:
            for line in f:
                line = line.strip()
                if line.__contains__("cgs"):
                    if len(match := re.findall("^\\s*mass_cgs+.*=(.*)#", line)) > 0:
                        m_cgs = float(match[0])
                    elif len(match := re.findall("^\\s*length_cgs+.*=(.*)#", line)) > 0:
                        l_cgs = float(match[0])
                    elif len(match := re.findall("^\\s*time_cgs+.*=(.*)#", line)) > 0:
                        t_cgs = float(match[0])
        assert m_cgs > 0 and t_cgs > 0 and l_cgs > 0, "Units must be greater than zero"

        _l = au.def_unit(l_cgs * au.cm, "_l")  # pyright: ignore
        _m = au.def_unit(m_cgs * au.g, "_m")  # pyright: ignore
        _t = au.def_unit(t_cgs * au.s, "_t")  # pyright: ignore
        _a = _l / _t
        bondi = UnitSystem(to_be_ones=[], units=[_l, _m, _t])
        self.u = bondi
        self.code_length_in_bondi = (bondi.L / (ac.G * _m / _a**2)).si  # pyright: ignore
        self.code_time_in_bondi = (bondi.T / (ac.G * _m / _a**3)).si  # pyright: ignore

    def __load_single(
        self, path: pathlib.Path
    ) -> tp.Tuple[
        npt.NDArray[np.float64], tp.List[str], float, npt.NDArray[np.float64]
    ]:
        units_override = {}

        units_override.update(self.units)

        ds = yt.load(path, units_override=units_override)  # pyright: ignore
        dd = ds.all_data()
        if not self.drop_angles:
            distances = np.array(dd.fcoords[:, 0])
            prof = (dd.to_dataframe(["vel1", "vel2", "rho", "press"])).to_numpy()
        else:
            distances = np.array(dd.fcoords[:, 0])
            prof = dd.to_dataframe(["vel1", "vel2", "rho", "press"])
            prof["r"] = distances

            prof = prof.drop_duplicates(subset=["r"])
            distances = prof["r"].to_numpy()
            prof = prof[["vel1", "vel2", "rho", "press"]].to_numpy()
        return (
            prof,
            ["vel1", "vel2", "rho", "press"],
            float(ds.current_time),
            distances,
        )

    def load_all(self):
        d0, fields, _t, distances = self.__load_single(self.files[0])

        def job(p) -> tp.Tuple[float, npt.NDArray[float64]]:
            dp, _, _t, _ = self.__load_single(p)
            return _t, dp

        resP = list(
            Parallel(n_jobs=4, prefer="processes")(delayed(job)(x) for x in self.files)
        )
        d0 = np.stack([d0] + [i[1] for i in resP if i is not None], axis=0)
        times = [_t] + [i[0] for i in resP if i is not None]

        data = xr.DataArray(
            d0,
            dims=["t", "r", "F"],
            coords={"t": times, "r": distances, "F": fields},
            name="profile",
        )
        self.data = data


class Helper:
    @staticmethod
    def split_by_sign(
        source: npt.NDArray, destination: tp.List[npt.NDArray]
    ) -> tp.List[npt.NDArray]:
        sign_positive = source > 0
        sign_changes = np.diff(sign_positive).astype(bool)
        split_indices = np.where(sign_changes)[0] + 1
        res = []
        assert type(destination) is list, (
            "destination arg must be a *list* of arrays which needs to be split, if a single array needs to be passed pass it like [array]"
        )
        for i in destination:
            res.append(np.split(i, split_indices))
        return res

    @staticmethod
    def plot_abs_log(
        ax: Axes, x: npt.NDArray, y: npt.NDArray, col: str = "blue"
    ) -> None:
        [xs, ys] = Helper.split_by_sign(y, [x, y])
        assert len(xs) == len(ys)
        for xi, yi in zip(xs, ys):
            sgn = -1 if yi[0] < 0 else 1
            ax.plot(xi, yi * sgn, ls="-" if sgn < 0 else "--", color=col)
        ax.set_xscale("log")
        ax.set_yscale("log")


if __name__ == "__main__":
    prof = Profiles("./")
