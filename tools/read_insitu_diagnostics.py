# Copyright 2022-2025
#
# This file is part of HiPACE++.
#
# Authors: AlexanderSinn
# License: BSD-3-Clause-LBNL

import numpy as np
from numpy.lib import recfunctions as rfn
import glob
import json
from scipy import constants

class InSituReader:

    def _count_braces(self, b):
        num_open_braces = 0
        num_char = 0
        for bb in b:
            num_char += 1
            if bb == b'{'[0]:
                num_open_braces += 1
            elif bb == b'}'[0]:
                num_open_braces -= 1
            if num_open_braces == 0:
                break
        return num_char

    def _get_buffer(self, file):
        with open(file, "rb") as f:
            bytes = f.read()
            end = self._count_braces(bytes)
            datatype_obj = json.JSONDecoder().raw_decode(bytes[:end].decode())
        return {"buffer" : bytes, "dtype" : np.dtype(datatype_obj[0]), "offset" : datatype_obj[1]}

    def _read_file(self, filenames):
        glob_file_names = list(glob.iglob(filenames))
        assert len(glob_file_names) > 0, f"File does not exist: '{filenames}'"
        return np.sort(rfn.stack_arrays([
            np.frombuffer(**self._get_buffer(file)) for file in glob_file_names
        ], usemask=False, asrecarray=False, autoconvert=True), order="time")

    def __init__(self, filenames):
        """
        Extract the insitu diagnostics.

        Parameters
        ----------

        filenames: string
            Path and name of all files containing insitu diagnostics of a single beam.
            Use '*' as a wildcard to read in multiple files eg. "diags/insitu/reduced_beam.*.txt"

        """
        self._all_data = self._read_file(filenames)
        self._avg_comps = {}
        self._slice_comps = {}

        # read quantities from structured array
        for comp in self._all_data.dtype.names:
            if comp in ["time", "step"]:
                self.__dict__[comp] = self._all_data[comp]
            elif comp in ["n_slices", "charge", "mass", "z_lo", "z_hi",
                          "normalized_density_factor", "is_normalized_units"]:
                self.__dict__[comp] = self._all_data[comp][0]
            elif comp in ["average", "total", "integrated"]:
                for acomp in self._all_data[comp].dtype.names:
                    self._avg_comps[acomp] = lambda c=comp, a=acomp: self._all_data[c][a]
            else:
                self._slice_comps[comp] = lambda c=comp: self._all_data[c]

        # conversion functions for derived quantities

        # distance axis
        if "time" in self.__dict__:
            self.__dict__["distance"] = self.time * constants.c

        # zeta axis
        if all(k in self.__dict__ for k in ["z_lo", "z_hi", "n_slices"]):
            z_lo = self.z_lo
            z_hi = self.z_hi
            n_slices = self.n_slices
            dzh = 0.5 * (z_hi - z_lo) / n_slices
            self.__dict__["zeta"] = np.linspace(z_lo + dzh, z_hi - dzh, n_slices)

        for comps in [self._avg_comps, self._slice_comps]:

            # emittance x/y/z
            for d in ["x", "y", "z"]:
                if all(k in comps for k in [f"[{d}^2]", f"[{d}]", f"[u{d}^2]",
                                            f"[u{d}]", f"[{d}*u{d}]"]):
                    def calc_emittance(comps=comps, d=d):
                        q_x = comps[f"[{d}]"]()
                        q_ux = comps[f"[u{d}]"]()
                        return np.sqrt(np.abs((comps[f"[{d}^2]"]() - q_x**2) \
                            * (comps[f"[u{d}^2]"]() - q_ux**2) \
                            - (comps[f"[{d}*u{d}]"]() - q_x*q_ux)**2))
                    comps[f"emittance_{d}"] = calc_emittance

            # x/y/z/ux/uy/uz std
            for d in ["x", "y", "z", "ux", "uy", "uz"]:
                if all(k in comps for k in [f"[{d}]", f"[{d}^2]"]):
                    comps[f"{d}_std"] = lambda comps=comps, d=d: \
                        np.sqrt(np.maximum(comps[f"[{d}^2]"]() - comps[f"[{d}]"]()**2,0))

            # gamma mean
            if "[ga]" in comps:
                comps["gamma_mean"] = lambda comps=comps: comps["[ga]"]()

            # gamma spread
            if all(k in comps for k in ["[ga]", "[ga^2]"]):
                comps["gamma_spread"] = lambda comps=comps: \
                    np.sqrt(np.maximum(comps["[ga^2]"]() - comps["[ga]"]()**2,0))

            # energy in eV mean and spread
            for s in ["mean", "spread"]:
                if f"gamma_{s}" in comps and \
                    all(k in self.__dict__ for k in ["is_normalized_units", "mass"]):
                    comps[f"energy_{s}_eV"] = lambda comps=comps, s=s: \
                        (constants.m_e if self.is_normalized_units else 1.) \
                        * (constants.c**2 / constants.e) * self.mass * comps[f"gamma_{s}"]()

            # temperature in eV x/y/z
            for d in ["x", "y", "z"]:
                if f"u{d}_std" in comps and \
                    all(k in self.__dict__ for k in ["is_normalized_units", "mass"]):
                    comps[f"temperature_in_eV_{d}"] = lambda comps=comps, d=d: \
                        (constants.m_e if self.is_normalized_units else 1.) \
                        * (constants.c**2 / constants.e) * self.mass * comps[f"u{d}_std"]()**2

            # temperature in eV average
            if all(k in comps for k in ["temperature_in_eV_x", "temperature_in_eV_y",
                                        "temperature_in_eV_z"]):
                comps["temperature_in_eV"] = lambda comps=comps: 1./3. * (
                        comps["temperature_in_eV_x"]() +
                        comps["temperature_in_eV_y"]() +
                        comps["temperature_in_eV_z"]())

            # total_charge and per_slice_charge
            if "sum(w)" in comps and \
                all(k in self.__dict__ for k in ["charge", "normalized_density_factor"]):
                comp_name = "total_charge" if comps is self._avg_comps else "per_slice_charge"
                comps[comp_name] = lambda comps=comps: \
                    self.charge * self.normalized_density_factor * \
                    comps["sum(w)"]()

    def avail(self, type=None):
        """
        List available quantities.

        Parameters
        ----------

        type: None | "avg" | "slice" | "meta"
            None: Print all available quantities.
            "avg" | "slice" | "meta": Return a list of strings of quantities for,
                average, per-slice or metadata respectievly.

        """
        assert type in [None, "avg", "slice", "meta"]
        if type == "avg":
            return list(self._avg_comps.keys())
        elif type == "slice":
            return list(self._slice_comps.keys())
        elif type == "meta":
            return [l for l in self.__dict__.keys() if not l.startswith("_")]
        else:
            print("Available average quantities:")
            print(*self.avail("avg"))
            print("Available per-slice quantities:")
            print(*self.avail("slice"))
            print("Available Metadata:")
            print(*self.avail("meta"))

    def avg_data(self, quantity=None):
        """
        Returns averaged data.
        This function exposes both quantities from the file as well as derived quantities
        that are computed lazily if the required inputs are available.

        emittance_<dir> refers to the normalized projected emittance in direction <dir>.

        Parameters
        ----------

        quantity: str | tuple[str] | list[str]
            The quantity to get. If a list or tuple of strings is input, a list of arrays
            will be output accordingly.

        Returns
        -------

        1D Array over time steps | list[1D Array over time steps]
        """
        if isinstance(quantity, (list, tuple)):
            return [self.avg_data(q) for q in quantity]
        else:
            nl = '\n'
            assert quantity in self._avg_comps, \
                f"Quantity '{quantity}' not found.\n"+\
                f"Available averaged quantities:\n\n{nl.join(self.avail('avg'))}\n\n"+\
                "Please set the `quantity` argument accordingly!"
            return self._avg_comps[quantity]()

    def slice_data(self, quantity=None):
        """
        Returns per-slice data.
        This function exposes both quantities from the file as well as derived quantities
        that are computed lazily if the required inputs are available.

        emittance_<dir> refers to the normalized per-slice emittance in direction <dir>.

        Parameters
        ----------

        quantity: str | tuple[str] | list[str]
            The quantity to get. If a list or tuple of strings is input, a list of arrays
            will be output accordingly.

        Returns
        -------

        2D Array over time steps and slices | list[2D Array over time steps and slices]
        """
        if isinstance(quantity, (list, tuple)):
            return [self.slice_data(q) for q in quantity]
        else:
            nl = '\n'
            assert quantity in self._slice_comps, \
                f"Quantity '{quantity}' not found.\n"+\
                f"Available per-slice quantities:\n\n{nl.join(self.avail('slice'))}\n\n"+\
                "Please set the `quantity` argument accordingly!"
            return self._slice_comps[quantity]()

    def get_all_data(self):
        """
        Get internal all_data NumPy structured array for backwards compatibility.
        """
        return self._all_data


def beam_multiplot(insitu_reader: InSituReader, title: str):
    """
    Plot key metrics for a beam in a 3 by 3 plot.
    This is meant for quick analysis and as an example of how to use the insitu diagnostics.
    Note that some of the axis labels are only correct if the simulation was done in SI units.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 3, figsize=(16,12), dpi=150)

    axs[0][0].plot(insitu_reader.distance, insitu_reader.avg_data("energy_mean_eV"))
    axs[0][0].set_xlabel("Distance in m")
    axs[0][0].set_ylabel("Energy in eV")

    axs[0][1].plot(insitu_reader.distance, insitu_reader.avg_data("[x]"))
    axs[0][1].plot(insitu_reader.distance, insitu_reader.avg_data("[y]"), ":")
    axs[0][1].set_xlabel("Distance in m")
    axs[0][1].set_ylabel("Pos mean in m")

    axs[0][2].plot(insitu_reader.distance, insitu_reader.avg_data("x_std"))
    axs[0][2].plot(insitu_reader.distance, insitu_reader.avg_data("y_std") , ":")
    axs[0][2].set_xlabel("Distance in m")
    axs[0][2].set_ylabel("Pos std in m")

    axs[1][0].plot(insitu_reader.distance, 100 * insitu_reader.avg_data("energy_spread_eV") \
                                           / insitu_reader.avg_data("energy_mean_eV"))
    axs[1][0].set_xlabel("Distance in m")
    axs[1][0].set_ylabel("rel. Energy spread %")

    axs[1][1].plot(insitu_reader.distance, insitu_reader.avg_data("[ux]"))
    axs[1][1].plot(insitu_reader.distance, insitu_reader.avg_data("[uy]"), ":")
    axs[1][1].set_xlabel("Distance in m")
    axs[1][1].set_ylabel("u mean")

    axs[1][2].plot(insitu_reader.distance, insitu_reader.avg_data("ux_std"))
    axs[1][2].plot(insitu_reader.distance, insitu_reader.avg_data("uy_std"), ":")
    axs[1][2].set_xlabel("Distance in m")
    axs[1][2].set_ylabel("u std")

    axs[2][0].plot(insitu_reader.distance, insitu_reader.avg_data("total_charge"))
    axs[2][0].set_xlabel("Distance in m")
    axs[2][0].set_ylabel("Charge in C")

    axs[2][1].plot(insitu_reader.distance, insitu_reader.avg_data("emittance_x"))
    axs[2][1].set_xlabel("Distance in m")
    axs[2][1].set_ylabel("Norm. proj. emittance in x in m")

    axs[2][2].plot(insitu_reader.distance, insitu_reader.avg_data("emittance_y"))
    axs[2][2].set_xlabel("Distance in m")
    axs[2][2].set_ylabel("Norm. proj. emittance in y in m")

    fig.suptitle(title)
    fig.tight_layout()
    fig.show()

    return fig, axs


if __name__ == '__main__':
    # To import this file from another script use (change the path):
    #
    # # console:
    # pip install -U -e ~/hipace/tools
    #
    # # python:
    # import read_insitu_diagnostics as diag
    #
    # ir = diag.InSituReader("diags/insitu/reduced_beam.*.txt")
    import matplotlib.pyplot as plt

    ir = InSituReader("diags/insitu/reduced_beam.*.txt")

    ir.avail()

    beam_multiplot(ir, "Beam")

    plt.figure(figsize=(7,7), dpi=150)
    plt.pcolormesh(ir.zeta, ir.time, ir.slice_data("emittance_x"))
