//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file blast.cpp
//! \brief Problem generator for spherical blast wave problem.  Works in
//! Cartesian,
//!        cylindrical, and spherical coordinates.  Contains post-processing
//!        code to check whether blast is spherical for regression tests
//!
//! REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//!   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>  // fopen(), fprintf(), freopen()
#include <cstring> // strcmp()
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../units/units.hpp"

// Declarations start here
namespace {
Real mbh;
Real gamma_idx, inv_gamma, gm1, inv_gm1, polytropic_constant;
Real rho_infty, ur_infty, cs2_infty, cs_infty, pres_infty, en_den_infty;
Real GN;
Real rB;
} // namespace
void GravitationalSource(MeshBlock *pmb, const Real time, const Real dt,
                         const AthenaArray<Real> &prim,
                         const AthenaArray<Real> &prim_scalar,
                         const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                         AthenaArray<Real> &cons_scalar);

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt, int il, int iu, int jl,
                   int ju, int kl, int ku, int ngh);
// End of declarations

void Mesh::InitUserMeshData(ParameterInput *pin) {
  EnrollUserExplicitSourceFunction(GravitationalSource);
  Units units(pin);
  GN = units.grav_const_code;
  // Read problem parameters
  gamma_idx = pin->GetReal("hydro", "gamma");
  mbh = pin->GetReal("problem", "mbh");
  rho_infty = pin->GetReal("problem", "rho_infty");
  cs_infty = pin->GetReal("problem", "cs_infty");
  ur_infty = pin->GetReal("problem", "ur_infty");
  cs2_infty = cs_infty * cs_infty;

  inv_gamma = 1. / gamma_idx;
  gm1 = gamma_idx - 1;
  inv_gm1 = 1. / gm1;

  rB = GN * mbh / cs2_infty;

  // For polytropic gas : P = k ρ^Γ
  //                    : cs^2 = k Γ ρ^(Γ-1)
  polytropic_constant =
      cs_infty * cs_infty * inv_gamma / std::pow(rho_infty, gm1);
  pres_infty = polytropic_constant * std::pow(rho_infty, gamma_idx);

  const Real k_units_cgs =
      units.code_pressure_cgs / std::pow(units.code_density_cgs, gamma_idx);

  en_den_infty = pres_infty / gm1 + 0.5 * ur_infty * ur_infty * rho_infty;

  // Enroll boundary functions
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, FixedBoundary);

  // Print out useful information
#define LEFTSETW(x) std::left << std::setw(x)
  const int num_width = 8;
  std::stringstream msg;
  msg << std::setprecision(2) << '\n';
  msg << "######################################" << '\n';
  msg << "#############  Bondi problem generator" << '\n';
  msg << LEFTSETW(33) << "Polytropic gas index" << ":  " << gamma_idx << '\n';
  msg << LEFTSETW(33) << "Gravitational constant " << ":  " << GN << " [code]"
      << '\n';
  msg << "###### Input parameters : " << '\n';
  msg << LEFTSETW(33) << "## Mass of BH " << ":  " << LEFTSETW(num_width) << mbh
      << " [code] = " << LEFTSETW(num_width) << mbh / units.solar_mass_code
      << " [Msun]" << '\n';
  msg << LEFTSETW(33) << "## Density at infinity " << ":  "
      << LEFTSETW(num_width) << rho_infty << " [code] = " << LEFTSETW(num_width)
      << rho_infty * units.code_density_cgs << " [g/cm^3]" << '\n';
  msg << LEFTSETW(33) << "## Radial velocity at infinity " << ":  "
      << LEFTSETW(num_width) << ur_infty << " [code] = " << LEFTSETW(num_width)
      << ur_infty / units.km_s_code << " [km/s]" << '\n';
  msg << LEFTSETW(33) << "## Sound speed at infinity " << ":  "
      << LEFTSETW(num_width) << cs_infty << " [code] = " << LEFTSETW(num_width)
      << cs_infty / units.km_s_code << " [km/s]" << '\n';
  msg << "###### Derived parameters" << '\n';
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
#undef LEFTSETW

  return;
}
//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Spherical blast wave test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        phydro->u(IDN, k, j, i) = rho_infty;
        phydro->u(IM1, k, j, i) = rho_infty * ur_infty;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
      }
    }
  }
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
}

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt, int il, int iu, int jl,
                   int ju, int kl, int ku, int ngh) {
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = 1; i <= ngh; ++i) {
        prim(IDN, k, j, il - i) = rho_infty;
        prim(IVX, k, j, il - i) = ur_infty;
        prim(IPR, k, j, il - i) = pres_infty;
      }
    }
  }
}
