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

from classes import Transient, Host, Filter
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
    redshift=None,
    mwebv=None,
    all_filters=None,
    download_cutouts=True,
):
    """
    Run the full FrankenBlast pipeline for a single galaxy with known coordinates,
    skipping only the host association (prost) step.
    """
    from get_host_images import download_and_save_cutouts, get_cutouts
    from create_apertures import construct_aperture
    from do_photometry import do_global_photometry

    prev_dir = os.getcwd()
    os.chdir(franken_root)

    print(f"\n{'='*60}")
    print(f"Fitting: {name}  (ra={ra:.4f}, dec={dec:.4f}, z={redshift})")
    print(f"{'='*60}")

    if all_filters is None:
        survey_list(SURVEY_METADATA)
        all_filters = Filter.all()

    # MW dust
    if mwebv is None:
        print("Looking up MW E(B-V) from SFD maps...")
        mwebv = get_mwebv_from_coords(ra, dec)
    print(f"MW E(B-V) = {mwebv:.4f}")

    # ── Build host & container ────────────────────────────────────────────────
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
    
    galaxy = Transient(
        name=name,
        coordinates=SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs"),
        transient_redshift=redshift,
        milkyway_dust_reddening=mwebv,
    )
    galaxy.host = host

    # ── Download cutout images ────────────────────────────────────────────────
    if download_cutouts:
        print("Downloading cutout images...")
        download_and_save_cutouts(galaxy, filters=Filter.all())
    else:
        print("Skipping download, loading existing cutouts...")

    # ── Aperture photometry ───────────────────────────────────────────────────
    cutouts = galaxy.cutouts  # populated by download_and_save_cutouts
    global_apertures = []

    print("Constructing apertures and doing photometry...")
    for cutout in cutouts:
        aperture = construct_aperture(cutout, galaxy.host.coordinates)
        global_apertures.append(aperture)

    galaxy.global_apertures = global_apertures

    all_phot = []
    for i in np.arange(0, len(cutouts), 1):
        filt = cutouts[i]['filter']
        apr  = galaxy.global_apertures[i]
        phot = do_global_photometry(
            galaxy, filter=filt, aperture=apr,
            fwhm_correction=False, show_plot=False,  # set True to inspect apertures
        )
        all_phot.append(phot)

    galaxy.host_photometry = all_phot

    # ── Clean photometry (same as notebook) ──────────────────────────────────
    phot_filters = np.array([cutout['filter'].name for cutout in cutouts])

    available_filters = []
    update_phot = []
    for filtername in phot_filters:
        filter_obj = next((f for f in all_filters if f.name == filtername), None)
        if filter_obj:
            phot_flux = np.array(galaxy.host_photometry)[
                np.where(phot_filters == filtername)][0]['flux']
            phot_err  = np.array(galaxy.host_photometry)[
                np.where(phot_filters == filtername)][0]['flux_error']
            if phot_flux is not None and phot_flux > 0:
                available_filters.append({"filter": filter_obj})
                update_phot.append(
                    np.array(galaxy.host_photometry)[
                        np.where(phot_filters == filtername)][0]
                )

    galaxy.host_phot_filters = np.array(available_filters)
    galaxy.host_photometry   = np.array(update_phot)

    # ── SED fit ───────────────────────────────────────────────────────────────
    sbi_params, train_fname = _build_sbi_params(redshift)

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
        print(f"Fit completed in {elapsed:.2f} min")
    except Exception as e:
        print(f"ERROR during fit for {name}: {e}")
        traceback.print_exc()
        return None

    res = _load_results(name)

    os.chdir(prev_dir)

    return res


# ── DataFrame batch runner ────────────────────────────────────────────────────

def run_from_dataframe(df, skip_existing=True, download_cutouts=True):
    """
    Run full FrankenBlast pipeline for all galaxies in a DataFrame.
    Only needs: name, ra, dec, and optionally redshift, mwebv.
    """
    required_cols = ["name", "ra", "dec"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    results_list = []
    n = len(df)

    for idx, row in df.iterrows():
        name = str(row["name"])
        print(f"\n[{idx+1}/{n}] {name}")

        if skip_existing:
            npy_path = os.path.join(SED_OUTPUT_ROOT, name, f"{name}_global.npy")
            if os.path.exists(npy_path):
                print(f"  Output already exists — skipping.")
                existing = _load_results(name)
                if existing:
                    results_list.append(existing)
                continue

        # Check if cutouts already downloaded
        cutout_dir = os.path.join("./cutouts", name)
        already_downloaded = os.path.exists(cutout_dir) and len(os.listdir(cutout_dir)) > 0

        result = run_single_with_photometry(
            name     = name,
            ra       = float(row["ra"]),
            dec      = float(row["dec"]),
            redshift = float(row["redshift"]) if "redshift" in df.columns and pd.notna(row.get("redshift")) else None,
            mwebv    = float(row["mwebv"])    if "mwebv"    in df.columns and pd.notna(row.get("mwebv"))    else None,
            all_filters     = ALL_FILTERS,
            download_cutouts = not already_downloaded,  # skip if already have images
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