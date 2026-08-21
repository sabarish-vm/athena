//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file bondi.cpp
//! \brief Problem generator for spherical non-relativistic Bondi problem.
// C headers
#include <limits>
#include <pybind11/embed.h> // for embedding Python
#include <pybind11/numpy.h> // for numpy support
#include <sys/stat.h>

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>  // fopen(), fprintf(), freopen()
#include <cstring> // strcmp()
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../units/units.hpp"

namespace py = pybind11;

// Declarations start here
namespace {
Real mbh, mdot, mdot_python;
Real gamma_idx, inv_gamma, gm1, inv_gm1, polytropic_constant, k_units_cgs;
Real rho_infty, ur_infty, cs2_infty, cs_infty, pres_infty, en_den_infty;
Real GN;
Real rB;
Real sbm_cgs, sbm_code, cgs_cross_section_code;
int ic_mode, bc_mode;
} // namespace
void GravitationalSource(MeshBlock *pmb, const Real time, const Real dt,
                         const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &prim_scalar,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                         AthenaArray<Real> &cons_scalar);

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt, int il, int iu, int jl,
                   int ju, int kl, int ku, int ngh);

void Conductivity(HydroDiffusion *phdif, MeshBlock *pmb,
                  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                  int is, int ie, int js, int je, int ks, int ke);

Real GravityTimeStep(MeshBlock *pmb);
// End of declarations

bool path_exists(const std::string &path) {
  struct stat buffer;
  int result = stat(path.c_str(), &buffer);
  return (result == 0);
}

std::tuple<py::array_t<Real>, py::array_t<Real>>
init_profile(const int &is, const int &ie, Coordinates *pcoords) {
  py::scoped_interpreter guard{};
  // Set paths and check if path exists
  py::module sys = py::module::import("sys");
  sys.attr("path").attr("append")(PGEN_ABSPATH);
  std::string pymod_path = std::string(PGEN_ABSPATH) + std::string("/bondi.py");
  bool module_exists = path_exists(pymod_path);
  if (module_exists) {
  } else {
    std::stringstream err;
    err << "Python module bondi.py for generating analytical profiles does not "
           "exist\n";
    err << "Expected path = " << pymod_path;
    throw std::runtime_error(err.str());
  }

  // Import Python module
  py::module_ mymodule = py::module_::import("bondi");
  // Get Python function
  py::object func = mymodule.attr("soln");

  std::vector<Real> vec;
  // Note that the loop is only over the interior domain
  for (int i = is; i <= ie; i++) {
    // Convert r into bondi units
    const Real r = pcoords->x1v(i) / rB;
    vec.push_back(r);
  }

  // Create a numpy array
  py::array_t<Real> input_arr(vec.size(), vec.data());

  // Python call
  py::tuple result = func(gamma_idx, input_arr);

  // Untuple the python tuple
  mdot_python = result[0].cast<Real>();
  py::array_t<Real> rad_vel = result[1].cast<py::array_t<Real>>();
  py::array_t<Real> rho = result[2].cast<py::array_t<Real>>();

  // Create a cpp tuple of density and velocity profile to return
  auto ret_tuple = std::make_tuple(rad_vel, rho);
  return ret_tuple;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Enroll Source term
  EnrollUserExplicitSourceFunction(GravitationalSource);

  // Enroll Gravitational timestep
  EnrollUserTimeStepFunction(GravityTimeStep);

  bc_mode = pin->GetInteger("problem", "bc_mode");
  // Enroll boundary functions
  if (bc_mode > 0) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, FixedBoundary);
  } else {
  }

  // Setup units
  Units units(pin);
  GN = units.grav_const_code;
  cgs_cross_section_code = units.cm_code * units.cm_code / units.gram_code;

  // Read problem input parameters
  gamma_idx = pin->GetReal("hydro", "gamma");
  mbh = pin->GetReal("problem", "mbh");
  rho_infty = pin->GetReal("problem", "rho_infty");
  cs_infty = pin->GetReal("problem", "cs_infty");
  ur_infty = pin->GetReal("problem", "ur_infty");

  // Choose IC mode
  ic_mode = pin->GetInteger("problem", "ic_mode");

  // Cross-section for Conductivity
  sbm_cgs = pin->GetOrAddReal("problem", "sbm_cgs", 0.0);
  sbm_code = sbm_cgs * cgs_cross_section_code;
  // Enroll Conductivity function
  if (sbm_cgs > 0.0) {
    EnrollConductionCoefficient(Conductivity);
  }

  // Setup derived parameters
  cs2_infty = cs_infty * cs_infty;
  inv_gamma = 1. / gamma_idx;
  gm1 = gamma_idx - 1;
  inv_gm1 = 1. / gm1;

  rB = GN * mbh / cs2_infty; // Bondi radius

  // For polytropic gas : P = k ρ^Γ
  //                    : cs^2 = k Γ ρ^(Γ-1)
  polytropic_constant = cs2_infty * inv_gamma / std::pow(rho_infty, gm1);
  pres_infty = polytropic_constant * std::pow(rho_infty, gamma_idx);

  k_units_cgs = // units for polytropic_constant
      units.code_pressure_cgs / std::pow(units.code_density_cgs, gamma_idx);

  en_den_infty = pres_infty * inv_gm1 + 0.5 * ur_infty * ur_infty * rho_infty;

  mdot = PI * (GN * GN * mbh * mbh) * rho_infty / (cs_infty * cs2_infty);
  { // Print out useful information
#define LEFTSETW(x) std::left << std::setw(x)
    const int num_width = 8;
    std::stringstream msg;
    msg << std::setprecision(2) << '\n';
    msg << "######################################" << '\n';
    msg << "#############  Bondi problem generator" << '\n';
    msg << "###### Setup\n";
    if (ic_mode > 0) {
      msg << LEFTSETW(33) << "Initial conditions " << ": " << "Realistic\n";
    } else {
      msg << LEFTSETW(33) << "Initial conditions :" << ": " << "Uniform\n";
    }
    if (bc_mode > 0) {
      msg << LEFTSETW(33) << "Boundary conditions " << ": " << "Fixed\n";
    } else {
      msg << LEFTSETW(33) << "Boundary conditions " << ": " << "Open\n";
    }

    msg << LEFTSETW(33) << "Polytropic gas index" << ":  " << gamma_idx << '\n';
    msg << LEFTSETW(33) << "Gravitational constant " << ":  " << GN << " [code]"
        << '\n';
    msg << "###### Input parameters : " << '\n';
    msg << LEFTSETW(33) << "## Mass of BH " << ":  " << LEFTSETW(num_width)
        << mbh << " [code] = " << LEFTSETW(num_width)
        << mbh / units.solar_mass_code << " [Msun]" << '\n';
    msg << LEFTSETW(33) << "## Density at infinity " << ":  "
        << LEFTSETW(num_width) << rho_infty
        << " [code] = " << LEFTSETW(num_width)
        << rho_infty * units.code_density_cgs << " [g/cm^3]" << '\n';
    msg << LEFTSETW(33) << "## Radial velocity at infinity " << ":  "
        << LEFTSETW(num_width) << ur_infty
        << " [code] = " << LEFTSETW(num_width) << ur_infty / units.km_s_code
        << " [km/s]" << '\n';
    msg << LEFTSETW(33) << "## Sound speed at infinity " << ":  "
        << LEFTSETW(num_width) << cs_infty
        << " [code] = " << LEFTSETW(num_width) << cs_infty / units.km_s_code
        << " [km/s]" << '\n';
    msg << "#### Heat Conduction :\n";
    msg << LEFTSETW(33) << "## Cross-section " << ":  " << LEFTSETW(num_width)
        << sbm_code << " [code] = " << LEFTSETW(num_width) << sbm_cgs
        << " [cm^2/g]" << '\n';
    msg << "###### Derived parameters" << '\n';
    msg << LEFTSETW(33) << "## Accretion rate " << ":  " << LEFTSETW(num_width)
        << mdot << " [code] = " << LEFTSETW(num_width)
        << mdot / units.solar_mass_code * units.yr_code << " [solmass/yr]"
        << '\n';
    msg << LEFTSETW(33) << "## Bondi radius " << ":  " << LEFTSETW(num_width)
        << rB << " [code] = " << LEFTSETW(num_width) << rB / units.pc_code
        << " [pc]" << '\n';
    msg << LEFTSETW(33) << "## Pressure at infinity " << ":  "
        << LEFTSETW(num_width) << pres_infty
        << " [code] = " << LEFTSETW(num_width)
        << pres_infty * units.code_pressure_cgs << " [dyne/cm^2]" << '\n';
    msg << LEFTSETW(33) << "## Energy density at infinity " << ":  "
        << LEFTSETW(num_width) << en_den_infty
        << " [code] = " << LEFTSETW(num_width)
        << en_den_infty * units.code_energydensity_cgs << " [erg/cm^3]" << '\n';
    msg << LEFTSETW(33) << "## Polytropic constant " << ":  "
        << LEFTSETW(num_width) << polytropic_constant
        << " [code] = " << LEFTSETW(num_width)
        << polytropic_constant * k_units_cgs << " [cgs] " << '\n';
    msg << "######################################" << '\n' << std::endl;
    std::cout << msg.str();
  }
#undef LEFTSETW

  return;
}
//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  if (ic_mode > 0) {
    auto profile_tuple = init_profile(is, ie, pcoord);
    auto rad_vel_vec = std::get<0>(profile_tuple);
    auto rho_vec = std::get<1>(profile_tuple);
    for (int k = ks; k <= ke; k++) {
      for (int j = js; j <= je; j++) {
        for (int i = is; i <= ie; i++) {
          const Real r = pcoord->x1v(i);
          const int id = i - is;
          const Real rho = rho_vec.at(id);
          const Real ur = rad_vel_vec.at(id);
          const Real press = polytropic_constant * std::pow(rho, gamma_idx);
          const Real en_den = press * inv_gm1 + 0.5 * ur * ur * rho;
          phydro->u(IDN, k, j, i) = rho;
          phydro->u(IM1, k, j, i) = rho * ur;
          phydro->u(IEN, k, j, i) = en_den;
          phydro->u(IM2, k, j, i) = 0.0;
          phydro->u(IM3, k, j, i) = 0.0;
        }
      }
    }
  } else if (ic_mode < 0) {
    for (int k = ks; k <= ke; k++) {
      for (int j = js; j <= je; j++) {
        for (int i = is; i <= ie; i++) {
          const Real r = pcoord->x1v(i);
          const Real ur_mag = std::sqrt(GN * mbh / r);
          const Real rho_r = mdot / (4 * PI * r * r * ur_mag);
          const Real press = polytropic_constant * std::pow(rho_r, gamma_idx);
          const Real en_den = press * inv_gm1 + 0.5 * ur_mag * ur_mag * rho_r;
          phydro->u(IDN, k, j, i) = rho_r;
          phydro->u(IM1, k, j, i) = rho_r * ur_mag * -1;
          phydro->u(IEN, k, j, i) = en_den;
          phydro->u(IM2, k, j, i) = 0.0;
          phydro->u(IM3, k, j, i) = 0.0;
        }
      }
    }
  } else {
    for (int k = ks; k <= ke; k++) {
      for (int j = js; j <= je; j++) {
        for (int i = is; i <= ie; i++) {
          phydro->u(IDN, k, j, i) = rho_infty;
          phydro->u(IM1, k, j, i) = rho_infty * ur_infty;
          phydro->u(IEN, k, j, i) = en_den_infty;
          phydro->u(IM2, k, j, i) = 0.0;
          phydro->u(IM3, k, j, i) = 0.0;
        }
      }
    }
  }
  return;
}

void GravitationalSource(MeshBlock *pmb, const Real time, const Real dt,
                         const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &prim_scalar,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                         AthenaArray<Real> &cons_scalar) {
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real rad = pmb->pcoord->x1v(i);
        const Real g_r = GN * mbh / (rad * rad);
        const Real den = prim(IDN, k, j, i);
        const Real src = (rad == 0) ? 0 : dt * den * g_r / rad;
        cons(IM1, k, j, i) -= src * rad;
        cons(IEN, k, j, i) -= src * rad * prim(IVX, k, j, i);
      }
    }
  }
  return;
}

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt, int il, int iu, int jl,
                   int ju, int kl, int ku, int ngh) {
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = 1; i <= ngh; ++i) {
        // prim(IDN, k, j, iu + ngh) = rho_infty;
        // prim(IVX, k, j, iu + ngh) = ur_infty;
        // prim(IPR, k, j, iu + ngh) = pres_infty;
        prim(IDN, k, j, iu + 1) = rho_infty;
        prim(IVX, k, j, iu + 1) = ur_infty;
        prim(IPR, k, j, iu + 1) = pres_infty;
      }
    }
  }
  return;
}

void Conductivity(HydroDiffusion *phdif, MeshBlock *pmb,
                  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc,
                  int is, int ie, int js, int je, int ks, int ke) {
  if (phdif->kappa_iso > 0.0) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          Real den = prim(IDN, k, j, i);
          Real press = prim(IPR, k, j, i);
          Real kappa_smfp =
              2.1 * std::sqrt(press) / std::pow(den, 1.5) / sbm_code;
          phdif->kappa(HydroDiffusion::DiffProcess::iso, k, j, i) = kappa_smfp;
        }
      }
    }
  }
}

Real GravityTimeStep(MeshBlock *pmb) {
  Real min_dt = std::numeric_limits<double>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real rad = pmb->pcoord->x1v(i);
        const Real g_r = GN * mbh / (rad * rad);
        const Real dx = pmb->pcoord->dx1v(i);
        const Real dt = std::sqrt(dx / g_r);
        min_dt = std::min(min_dt, dt);
      }
    }
  }
  return min_dt;
}
