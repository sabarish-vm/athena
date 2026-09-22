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
from pathlib import Path
from astropy import constants as ac


class Profiles:
    def __init__(
        self,
        dirpath: str,
        units: dict = {},
        drop_angles=False,
        input_file: str = "",
        result_path: Path = Path("./results.nc"),
        n_files_jump: int = 1,
    ) -> None:
        self.dirpath: Path  # Simulation directory
        self.files: tp.List[pathlib.Path]  # List of snapshots
        self.yt_units: dict  # dictionary for yt_units
        self.input_file: Path  # Path to the input file
        self.result_path: Path  # Path to the saved dataset
        self.data: xr.DataArray
        self.drop_angles: bool  # If given a 3D dataset, drop angular coords?
        self.n_files_jump: int

        self.n_files_jump = n_files_jump
        self.dirpath = pathlib.Path(dirpath)
        self.files = os_sorted([i for i in self.dirpath.glob("*.athdf")])[
            :: self.n_files_jump
        ]
        self.yt_units = units
        self.drop_angles: bool = drop_angles
        if input_file == "":
            self.input_file = list(Path(".").glob("athinput*"))[0]
        else:
            self.input_file = Path(input_file)
        self.result_path = result_path

    def setup_units(self) -> None:
        m_cgs: float = 0
        l_cgs: float = 0
        t_cgs: float = 0
        sbm_cgs: float = 0
        with open(self.input_file) as f:
            for line in f:
                line = line.strip()
                if line.__contains__("cgs"):
                    if (
                        len(
                            match := re.findall(
                                "^\\s*sbm_cgs\\s*=\\s*(.*?)\\s*#?$", line
                            )
                        )
                        > 0
                    ):
                        sbm_cgs = float(match[0])
                    if (
                        len(
                            match := re.findall(
                                "^\\s*mass_cgs\\s*=\\s*(.*?)\\s*#?$", line
                            )
                        )
                        > 0
                    ):
                        m_cgs = float(match[0])
                    elif (
                        len(
                            match := re.findall(
                                "^\\s*length_cgs\\s*=\\s*(.*?)\\s*#?$", line
                            )
                        )
                        > 0
                    ):
                        l_cgs = float(match[0])
                    elif (
                        len(
                            match := re.findall(
                                "^\\s*time_cgs\\s*=\\s*(.*?)\\s*#?$", line
                            )
                        )
                        > 0
                    ):
                        t_cgs = float(match[0])

        assert m_cgs > 0 and t_cgs > 0 and l_cgs > 0, "Units must be greater than zero"
        _l = au.def_unit("_l",l_cgs * au.cm)  # pyright: ignore
        _m = au.def_unit("_m",m_cgs * au.g,)  # pyright: ignore
        _t = au.def_unit("_t",t_cgs * au.s)  # pyright: ignore
        _a = _l / _t
        bondi = UnitSystem(to_be_ones=[], units=[_l, _m, _t])
        self.u = bondi
        with open(self.input_file) as f:
            for line in f:
                line = line.strip()
                if line.__contains__("infty") or line.__contains__("mbh"):
                    if (
                        len(
                            match := re.findall(
                                "^\\s*rho_infty\\s*=\\s*(.*?)\\s*#?$", line
                            )
                        )
                        > 0
                    ):
                        self.rho_infty = float(match[0])
                    elif (
                        len(
                            match := re.findall(
                                "^\\s*cs_infty\\s*=\\s*(.*?)\\s*#?$", line
                            )
                        )
                        > 0
                    ):
                        self.cs_infty = float(match[0])
                    elif (
                        len(
                            match := re.findall(
                                "^\\s*ur_infty\\s*=\\s*(.*?)\\s*#?$", line
                            )
                        )
                        > 0
                    ):
                        self.ur_infty = float(match[0])
                    elif (
                        len(match := re.findall("^\\s*mbh\\s*=\\s*(.*?)\\s*#?$", line))
                        > 0
                    ):
                        self.mbh = float(match[0])
        self.code_length_in_bondi = (bondi.L / (ac.G * _m / _a**2)).si  # pyright: ignore
        self.code_time_in_bondi = (bondi.T / (ac.G * _m / _a**3)).si  # pyright: ignore
        self.GN = self.u.from_another_value(ac.G)
        self.mdot = (
            np.pi * (self.GN * self.mbh) ** 2 * self.rho_infty / self.cs_infty**3
        )
        self.sbm_cgs = sbm_cgs

    def __load_single(
        self, path: pathlib.Path
    ) -> tp.Tuple[
        npt.NDArray[np.float64], tp.List[str], float, npt.NDArray[np.float64]
    ]:
        units_override = {}

        units_override.update(self.yt_units)

        ds = yt.load(path, units_override=units_override)  # pyright: ignore
        dd = ds.all_data()
        if not self.drop_angles:
            distances = np.array(dd.fcoords[:, 0])
            prof = (dd.to_dataframe(["vel1", "rho", "press", "cell_mass"])).to_numpy()
        else:
            distances = np.array(dd.fcoords[:, 0])
            prof = dd.to_dataframe(["vel1", "rho", "press", "cell_mass"])
            prof["r"] = distances

            prof = prof.drop_duplicates(subset=["r"])
            distances = prof["r"].to_numpy()
            prof = prof[["vel1", "rho", "press", "cell_mass"]].to_numpy()
        return (
            prof,
            ["vel1", "rho", "press", "mass"],
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
            attrs={"mdot": self.mdot, "sbm_cgs": self.sbm_cgs},
        )
        self.data = data

    def save_data(self):
        self.data.to_netcdf(self.result_path)

    def load_data(self):
        if self.result_path.exists():
            self.data = xr.load_dataarray(self.result_path)
        else:
            self.load_all()


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
        ax: Axes, x: npt.NDArray, y: npt.NDArray, kwargs: dict = {}
    ) -> None:
        [xs, ys] = Helper.split_by_sign(y, [x, y])
        assert len(xs) == len(ys)
        for xi, yi in zip(xs, ys):
            sgn = -1 if yi[0] < 0 else 1
            ax.plot(xi, yi * sgn, ls="-" if sgn < 0 else "--", **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")


if __name__ == "__main__":
    prof = Profiles("./")
