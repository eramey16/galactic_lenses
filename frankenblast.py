#!/usr/bin/env python
"""
frankenblast.py

Fit stellar population properties for galaxies with DESI Legacy Survey
photometry using FrankenBlast's SBI++ SED fitting pipeline.

Usage (CLI):
    # Single galaxy
    python fit_desi_galaxies.py \
        --name my_galaxy \
        --ra 133.278 \
        --dec -6.329 \
        --redshift 0.041 \
        --mwebv 0.05 \
        --flux_g 100.0 --ivar_g 1e16 \
        --flux_r 200.0 --ivar_r 1e16 \
        --flux_i 180.0 --ivar_i 1e16 \
        --flux_z 150.0 --ivar_z 1e16 \
        --flux_w1 80.0 --ivar_w1 1e16 \
        --flux_w2 40.0 --ivar_w2 1e16

    # From CSV/parquet
    python fit_desi_galaxies.py --input_file galaxies.csv

Usage (Python):
    from fit_desi_galaxies import run_single, run_from_dataframe
    import pandas as pd

    # Single galaxy
    result = run_single(
        name="my_galaxy",
        ra=133.278,
        dec=-6.329,
        redshift=0.041,
        mwebv=0.05,
        flux_g=100.0,  ivar_g=1e16,
        flux_r=200.0,  ivar_r=1e16,
        flux_i=180.0,  ivar_i=1e16,
        flux_z=150.0,  ivar_z=1e16,
        flux_w1=80.0,  ivar_w1=1e16,
        flux_w2=40.0,  ivar_w2=1e16,
    )

    # From DataFrame
    df = pd.read_csv("galaxies.csv")
    results = run_from_dataframe(df)

Expected CSV columns:
    name, ra, dec, redshift (optional), mwebv (optional),
    flux_g, ivar_g, flux_r, ivar_r, flux_i, ivar_i,
    flux_z, ivar_z, flux_w1, ivar_w1, flux_w2, ivar_w2
"""
import os
import sys

franken_root = '/home/jovyan/frankenblast-host'

os.environ['SPS_HOME'] = '/home/jovyan/fsps'
os.environ['SBIPP_ROOT'] = f'{franken_root}/data/SBI'
os.environ['SBIPP_PHOT_ROOT'] = f'{franken_root}/data/sbipp_phot'
os.environ['SBIPP_TRAINING_ROOT'] = f'{franken_root}/data/sbi_training_sets'
os.environ['SED_OUTPUT_ROOT'] = '/home/jovyan/galactic_lenses/data'


import time
import argparse
import traceback

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from classes import Host, Filter
from fit_host_sed import fit_host
from mwebv_host import get_mwebv
from get_host_images import survey_list


# ── Constants ─────────────────────────────────────────────────────────────────

# Maps (flux_key, ivar_key) -> FrankenBlast filter name
DESI_FILTER_MAP = [
    ("flux_g",  "ivar_g",  "PanSTARRS_g"),
    ("flux_r",  "ivar_r",  "PanSTARRS_r"),
    ("flux_i",  "ivar_i",  "PanSTARRS_i"),
    ("flux_z",  "ivar_z",  "PanSTARRS_z"),
    ("flux_w1", "ivar_w1", "WISE_W1"),
    ("flux_w2", "ivar_w2", "WISE_W2"),
]

SBIPP_ROOT       = os.environ.get("SBIPP_ROOT")
SBIPP_PHOT_ROOT  = os.environ.get("SBIPP_PHOT_ROOT")
SED_OUTPUT_ROOT  = os.environ.get("SED_OUTPUT_ROOT", "./data/sed_output")
SURVEY_METADATA  = "/home/jovyan/frankenblast-host/data/survey_frankenblast_metadata.yml"

survey_list(SURVEY_METADATA)

print('Printing filters:\n', [x.name for x in Filter.all()])


# ── Helpers ───────────────────────────────────────────────────────────────────

def nano_to_maggies(flux_nano, ivar_nano):
    """Convert DESI nanomaggies + inverse variance to maggies + error."""
    flux = flux_nano / 1e9
    err  = ivar_nano**-0.5 / 1e9
    return flux, err


def get_mwebv_from_coords(ra, dec):
    """Look up Milky Way E(B-V) from SFD dust maps at (ra, dec)."""
    class _Obj:
        pass
    t = _Obj()
    t.coordinates = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    return get_mwebv(t)


def _build_sbi_params(redshift):
    """Return the SBI++ model parameter dict for spec-z or photo-z mode."""
    if redshift is not None:
        return (
            {
                "anpe_fname_global":  f"{SBIPP_ROOT}/SBI_model_zfix_GPD2W_global.pt",
                "train_fname_global": f"{SBIPP_PHOT_ROOT}/sbi_phot_zfix_GPD2W_global.h5",
                "nhidden": 500,
                "nblocks": 15,
            },
            "zfix_GPD2W",
        )
    else:
        return (
            {
                "anpe_fname_global":  f"{SBIPP_ROOT}/SBI_model_zfree_GPD2W_global.pt",
                "train_fname_global": f"{SBIPP_PHOT_ROOT}/sbi_phot_zfree_GPD2W_global.h5",
                "nhidden": 500,
                "nblocks": 15,
            },
            "zfree_GPD2W",
        )


def _load_results(name):
    """Load the saved .npy output and return summary percentiles."""
    npy_path = os.path.join(SED_OUTPUT_ROOT, name, f"{name}_global.npy")
    if not os.path.exists(npy_path):
        return None
    model_dict = np.load(npy_path, allow_pickle=True).item(0)
    mass_p16, mass_p50, mass_p84 = np.percentile(model_dict["stellar_mass"], [16, 50, 84])
    sfr_p16,  sfr_p50,  sfr_p84  = np.percentile(model_dict["sfr"],          [16, 50, 84])
    return {
        "name":       name,
        "mass_p16":   mass_p16,
        "mass_p50":   mass_p50,
        "mass_p84":   mass_p84,
        "sfr_p16":    sfr_p16,
        "sfr_p50":    sfr_p50,
        "sfr_p84":    sfr_p84,
        "npy_path":   npy_path,
    }


# ── Core fitting function ─────────────────────────────────────────────────────

def run_single(
    name,
    ra,
    dec,
    flux_g,  ivar_g,
    flux_r,  ivar_r,
    flux_i,  ivar_i,
    flux_z,  ivar_z,
    flux_w1, ivar_w1,
    flux_w2, ivar_w2,
    redshift=None,
    mwebv=None,
    all_filters=None,
):
    """
    Fit a single galaxy with DESI photometry using FrankenBlast SBI++.

    Parameters
    ----------
    name : str
        Galaxy identifier (used for output filenames).
    ra, dec : float
        Galaxy coordinates in degrees (ICRS).
    flux_g/r/i/z/w1/w2 : float
        DESI fluxes in nanomaggies.
    ivar_g/r/i/z/w1/w2 : float
        DESI inverse variances in nanomaggies^-2.
    redshift : float or None
        Spectroscopic redshift. If None, photo-z mode is used.
    mwebv : float or None
        Milky Way E(B-V). If None, looked up automatically from SFD maps.
    all_filters : list or None
        Pre-loaded Filter.all() list (pass to avoid reloading for each galaxy).

    Returns
    -------
    dict or None
        Summary results dict, or None if the fit failed.
    """
    prev_dir = os.getcwd()
    os.chdir(franken_root)
    print(f"\n{'='*60}")
    print(f"Fitting: {name}  (ra={ra:.4f}, dec={dec:.4f}, z={redshift})")
    print(f"{'='*60}")

    # Load filters once if not provided
    if all_filters is None:
        all_filters = Filter.all()

    # MW dust
    if mwebv is None:
        print("Looking up MW E(B-V) from SFD maps...")
        mwebv = get_mwebv_from_coords(ra, dec)
    print(f"MW E(B-V) = {mwebv:.4f}")

    # Pack raw inputs for easy lookup
    raw_phot = {
        "flux_g":  (flux_g,  ivar_g),
        "flux_r":  (flux_r,  ivar_r),
        "flux_i":  (flux_i,  ivar_i),
        "flux_z":  (flux_z,  ivar_z),
        "flux_w1": (flux_w1, ivar_w1),
        "flux_w2": (flux_w2, ivar_w2),
    }

    # Build Host
    host = Host(
        name=name,
        redshift=redshift,
        photometric_redshift=None,
        host_prob=1.0,
        missedcat_prob=0.0,
        smallcone_prob=0.0,
        association_catalog="desi_legacy",
        milkyway_dust_reddening=mwebv,
        coordinates=SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs"),
    )

    # Build container object
    class GalaxyContainer:
        pass

    galaxy              = GalaxyContainer()
    galaxy.name         = name
    galaxy.host         = host
    galaxy.global_apertures = None

    # Build photometry arrays
    available_filters = []
    phot_list = []

    for flux_key, ivar_key, filter_name in DESI_FILTER_MAP:
        flux_nano, ivar_nano = raw_phot[flux_key]

        # Skip missing or invalid values
        if flux_nano is None or ivar_nano is None:
            print(f"  Skipping {filter_name}: missing value")
            continue
        if not np.isfinite(flux_nano) or not np.isfinite(ivar_nano):
            print(f"  Skipping {filter_name}: non-finite value")
            continue
        if ivar_nano <= 0:
            print(f"  Skipping {filter_name}: ivar <= 0")
            continue

        flux, flux_err = nano_to_maggies(flux_nano, ivar_nano)

        if flux <= 0:
            print(f"  Skipping {filter_name}: non-detection (flux={flux:.3e})")
            continue

        filter_obj = next((f for f in all_filters if f.name == filter_name), None)
        if filter_obj is None:
            print(f"  WARNING: '{filter_name}' not in survey metadata — skipping")
            continue

        available_filters.append({"filter": filter_obj})
        phot_list.append({"flux": flux, "flux_error": flux_err})
        print(f"  {filter_name:12s}: flux={flux:.3e}, err={flux_err:.3e} maggies")

    if len(phot_list) == 0:
        print(f"  ERROR: No valid photometry for {name} — skipping fit.")
        return None

    galaxy.host_phot_filters = np.array(available_filters)
    galaxy.host_photometry   = np.array(phot_list)

    # SBI++ parameters
    sbi_params, train_fname = _build_sbi_params(redshift)

    # Run fit
    try:
        start = time.time()
        fit_host(
            galaxy,
            sbi_params=sbi_params,
            fname=train_fname,
            all_filters=all_filters,
            mode="test",
            sbipp=True,
            aperture_type="global",
            aperture=galaxy.global_apertures,
            save=True,
        )
        elapsed = (time.time() - start) / 60
        print(f"  Fit completed in {elapsed:.2f} min")
    except Exception as e:
        print(f"  ERROR during fit for {name}: {e}")
        traceback.print_exc()
        return None

    # Load and return results
    results = _load_results(name)
    if results:
        print(f"  log M* = {results['mass_p50']:.2f} [{results['mass_p16']:.2f}, {results['mass_p84']:.2f}]")
        print(f"  SFR    = {results['sfr_p50']:.3f}  [{results['sfr_p16']:.3f}, {results['sfr_p84']:.3f}] Msun/yr")

    os.chdir(prev_dir)
    return results


# ── DataFrame batch runner ────────────────────────────────────────────────────

def run_from_dataframe(df, skip_existing=True):
    """
    Run SED fits for all galaxies in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: name, ra, dec, flux_g, ivar_g, flux_r, ivar_r,
        flux_i, ivar_i, flux_z, ivar_z, flux_w1, ivar_w1, flux_w2, ivar_w2.
        Optional columns: redshift, mwebv.
    skip_existing : bool
        If True, skip galaxies that already have a .npy output file.

    Returns
    -------
    pd.DataFrame
        Summary results for all successfully fitted galaxies.
    """
    required_cols = [
        "name", "ra", "dec",
        "flux_g", "ivar_g", "flux_r", "ivar_r",
        "flux_i", "ivar_i", "flux_z", "ivar_z",
        "flux_w1", "ivar_w1", "flux_w2", "ivar_w2",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Load filters once for the whole batch
    all_filters = Filter.all()
    print([f.name for f in all_filters])

    results_list = []
    n = len(df)

    for idx, row in df.iterrows():
        name = str(row["name"])
        print(f"\n[{idx+1}/{n}] {name}")

        # Skip if output already exists
        if skip_existing:
            npy_path = os.path.join(SED_OUTPUT_ROOT, name, f"{name}_global.npy")
            if os.path.exists(npy_path):
                print(f"  Output already exists — skipping. (set skip_existing=False to refit)")
                existing = _load_results(name)
                if existing:
                    results_list.append(existing)
                continue

        result = run_single(
            name    = name,
            ra      = float(row["ra"]),
            dec     = float(row["dec"]),
            redshift = float(row["redshift"]) if "redshift" in df.columns and pd.notna(row.get("redshift")) else None,
            mwebv    = float(row["mwebv"])    if "mwebv"    in df.columns and pd.notna(row.get("mwebv"))    else None,
            flux_g  = float(row["flux_g"]),   ivar_g  = float(row["ivar_g"]),
            flux_r  = float(row["flux_r"]),   ivar_r  = float(row["ivar_r"]),
            flux_i  = float(row["flux_i"]),   ivar_i  = float(row["ivar_i"]),
            flux_z  = float(row["flux_z"]),   ivar_z  = float(row["ivar_z"]),
            flux_w1 = float(row["flux_w1"]),  ivar_w1 = float(row["ivar_w1"]),
            flux_w2 = float(row["flux_w2"]),  ivar_w2 = float(row["ivar_w2"]),
            all_filters = all_filters,
        )

        if result:
            results_list.append(result)

    results_df = pd.DataFrame(results_list)
    print(f"\nCompleted {len(results_df)}/{n} fits successfully.")
    return results_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fit DESI galaxy photometry with FrankenBlast SBI++",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_file", type=str,
        help="Path to CSV or Parquet file with galaxy photometry table.")
    input_group.add_argument("--name", type=str,
        help="Galaxy name (single-object mode).")

    # Single-object arguments
    single = parser.add_argument_group("Single galaxy options")
    single.add_argument("--ra",       type=float, help="RA in degrees")
    single.add_argument("--dec",      type=float, help="Dec in degrees")
    single.add_argument("--redshift", type=float, default=None, help="Spectroscopic redshift (omit for photo-z)")
    single.add_argument("--mwebv",    type=float, default=None, help="MW E(B-V) (auto-looked up if omitted)")
    single.add_argument("--flux_g",   type=float); single.add_argument("--ivar_g",  type=float)
    single.add_argument("--flux_r",   type=float); single.add_argument("--ivar_r",  type=float)
    single.add_argument("--flux_i",   type=float); single.add_argument("--ivar_i",  type=float)
    single.add_argument("--flux_z",   type=float); single.add_argument("--ivar_z",  type=float)
    single.add_argument("--flux_w1",  type=float); single.add_argument("--ivar_w1", type=float)
    single.add_argument("--flux_w2",  type=float); single.add_argument("--ivar_w2", type=float)

    # Batch options
    batch = parser.add_argument_group("Batch options")
    batch.add_argument("--no_skip_existing", action="store_true",
        help="Refit galaxies that already have output files.")
    batch.add_argument("--output_summary", type=str, default=None,
        help="Path to save summary CSV of results (batch mode only).")

    return parser.parse_args()


def main():
    args = _parse_args()

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.input_file:
        fpath = args.input_file
        if fpath.endswith(".parquet"):
            df = pd.read_parquet(fpath)
        else:
            df = pd.read_csv(fpath)
        print(f"Loaded {len(df)} galaxies from {fpath}")

        results_df = run_from_dataframe(df, skip_existing=not args.no_skip_existing)

        # Save summary
        out_path = args.output_summary or "frankenblast_results.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")

    # ── Single-object mode ────────────────────────────────────────────────────
    else:
        required_single = ["ra", "dec", "flux_g", "ivar_g", "flux_r", "ivar_r",
                           "flux_i", "ivar_i", "flux_z", "ivar_z",
                           "flux_w1", "ivar_w1", "flux_w2", "ivar_w2"]
        missing = [f"--{k}" for k in required_single if getattr(args, k) is None]
        if missing:
            print(f"ERROR: Single-object mode requires: {', '.join(missing)}")
            sys.exit(1)

        result = run_single(
            name     = args.name,
            ra       = args.ra,
            dec      = args.dec,
            redshift = args.redshift,
            mwebv    = args.mwebv,
            flux_g   = args.flux_g,  ivar_g  = args.ivar_g,
            flux_r   = args.flux_r,  ivar_r  = args.ivar_r,
            flux_i   = args.flux_i,  ivar_i  = args.ivar_i,
            flux_z   = args.flux_z,  ivar_z  = args.ivar_z,
            flux_w1  = args.flux_w1, ivar_w1 = args.ivar_w1,
            flux_w2  = args.flux_w2, ivar_w2 = args.ivar_w2,
        )

        if result is None:
            print("Fit failed.")
            sys.exit(1)


if __name__ == "__main__":
    main()