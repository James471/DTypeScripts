#!/usr/bin/env python3
"""
Analyse D-type ionization front simulation output from Quokka.

Usage examples
--------------
# All plots and videos for the whole directory:
    python analyse_dtype.py run.toml /path/to/sims/ --all

# Specific products over the whole directory:
    python analyse_dtype.py run.toml /path/to/sims/ --maps x_HI temperature --videos x_HI --radius --eff-radius

# Single plotfile (no videos):
    python analyse_dtype.py run.toml /path/to/sims/ --snapshot DType0000100 --maps x_HI temperature --radius

# From the dtype_front_radii.csv the test problem writes directly, with no plotfiles
# required (e.g. to re-check the radius solution after a verification-code change,
# without re-running with plotfile_interval turned on):
    python analyse_dtype.py run.toml /path/to/sims/ --from-csv --eff-radius --eff-radius-error

# Same, comparing against the Krumholz & Matzner (2009) radiation-pressure solution
# instead of the pure Spitzer solution (requires a CSV written by DTypeFrontRadPres):
    python analyse_dtype.py run.toml /path/to/sims/ --from-csv --radiation-pressure --eff-radius-error
"""

import argparse
import os
import sys

try:
    import tomllib          # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # pip install tomli


def load_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_analytical(cfg, radiation_pressure=False, mg=False):
    from IFrontAnalysis import DTypeAnalytical
    Q   = cfg["stromgen"]["Q"]
    n_H = cfg["stromgen"]["primary_species_2"]
    optical_to_ionizing_fraction = cfg["stromgen"].get("optical_to_ionizing_fraction", 0.1)
    return DTypeAnalytical(Q, n_H, radiation_pressure=radiation_pressure, mg=mg,
                            optical_to_ionizing_fraction=optical_to_ionizing_fraction)


def get_dx_pc(cfg):
    """Smallest cell size in pc, from geometry.prob_lo/prob_hi and amr.n_cell (assumes
    the finest level equals the base grid, i.e. max_level = 0)."""
    from astropy import units as u
    prob_lo = cfg["geometry"]["prob_lo"]
    prob_hi = cfg["geometry"]["prob_hi"]
    n_cell  = cfg["amr"]["n_cell"]
    dx_cm = min((hi - lo) / n for lo, hi, n in zip(prob_lo, prob_hi, n_cell))
    return (dx_cm * u.cm).to(u.pc).value


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse Quokka D-type ionization front simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("toml",   help="Path to the Quokka input TOML file")
    p.add_argument("simdir", help="Directory containing the plotfiles (or, with --from-csv, "
                                  "the directory containing dtype_front_radii.csv)")

    # -- scope --
    scope = p.add_mutually_exclusive_group()
    scope.add_argument(
        "--snapshot", metavar="PLOTFILE",
        help="Analyse a single named plotfile only (no videos produced)",
    )
    scope.add_argument(
        "--from-csv", nargs="?", const="dtype_front_radii.csv", default=None, metavar="CSV",
        help="Read r_effective/r_spitzer[/r_radpres] directly from the CSV the test problem "
             "writes (default filename: dtype_front_radii.csv, resolved relative to simdir "
             "unless an absolute/relative path is given) instead of computing them from "
             "plotfiles. No plotfiles are required. Only --eff-radius, --eff-radius-error, "
             "and --norm-radius apply in this mode (maps/videos/--radius need plotfiles).",
    )

    p.add_argument(
        "--radiation-pressure", action="store_true",
        help="Compare against the Krumholz & Matzner (2009) radiation-pressure solution "
             "(the r_radpres CSV column, or DTypeAnalytical's radiation-pressure mode) "
             "instead of the pure Spitzer gas-pressure solution.",
    )
    p.add_argument(
        "--mg", action="store_true",
        help="Boost r_ch for the extra momentum deposited by the reprocessed optical "
             "band (testDTypeFrontMG.cpp / stromgen.optical_to_ionizing_fraction). "
             "Only affects --radiation-pressure.",
    )

    # -- what to make --
    p.add_argument(
        "--all", action="store_true",
        help="Make everything: all field maps, all videos, radius and effective-radius plots",
    )
    p.add_argument(
        "--maps", nargs="+", metavar="FIELD",
        choices=["x_HI", "temperature", "n_e", "n_HI", "n_HII",
                 "n_photon", "gasDensity", "velocity", "cs", "pressure",
                 "E_IR", "E_optical", "E_ion"],
        help="Field maps to create (slice plots). E_IR/E_optical/E_ion (radiation "
             "energy density of the IR/optical/ionizing bands) require a "
             "multi-group plotfile (e.g. DTypeFrontMG) and are not included by --all.",
    )
    p.add_argument(
        "--videos", nargs="+", metavar="FIELD",
        choices=["x_HI", "temperature", "n_e", "n_HI", "n_HII",
                 "n_photon", "gasDensity", "velocity", "cs", "pressure",
                 "E_IR", "E_optical", "E_ion"],
        help="Fields to animate as videos (ignored with --snapshot). "
             "E_IR/E_optical/E_ion require a multi-group plotfile and are not "
             "included by --all.",
    )
    p.add_argument("--radius",     action="store_true", help="Plot front radius history")
    p.add_argument("--eff-radius", action="store_true", help="Plot effective radius history")
    p.add_argument("--eff-radius-error", action="store_true",
                   help="Plot effective radius error (r_eff - r_analytical) / dx")
    p.add_argument("--norm-radius", action="store_true",
                   help="Plot normalised effective radius r_eff / r_s vs t / t_s")
    p.add_argument("--max-velocity", action="store_true",
                   help="Plot max velocity in the domain as a function of time")
    p.add_argument("--compare-analytical", action="store_true",
                   help="Plot r_effective alongside all three analytical solutions "
                        "(Spitzer, KM09, KM09+mg) vs. time on one axis")

    # -- map options --
    p.add_argument("--plot-analytical", action="store_true",
                   help="Overlay analytical front circle on maps")
    p.add_argument("--plot-eff",        action="store_true",
                   help="Overlay effective radius circle on maps")
    p.add_argument("--plot-front",      action="store_true",
                   help="Overlay median front radius circle on maps")
    p.add_argument("--cmap",  default="viridis", metavar="CMAP",
                   help="Matplotlib colormap for all field maps (default: viridis)")
    p.add_argument("--vmin",  type=float, default=None, metavar="VMIN",
                   help="Colorbar minimum (applied to all requested fields)")
    p.add_argument("--vmax",  type=float, default=None, metavar="VMAX",
                   help="Colorbar maximum (applied to all requested fields)")
    p.add_argument("--nolog", action="store_true",
                   help="Use linear (not log) colorbar scale")
    p.add_argument("--redo", action="store_true",
                   help="Recreate plots even if output files already exist")
    p.add_argument("--outdir", metavar="DIR",
                   help="Output directory for plots (default: <simdir>/Plots)")

    # -- steps --
    p.add_argument("--step", type=int, default=1,
                   help="Use every Nth plotfile (default: 1, i.e. all)")
    p.add_argument("--end", type=int, default=None,
                   help="Stop at plotfile number N (inclusive)")
    p.add_argument("--fps", type=int, default=10, help="Video frame rate (default: 10)")
    p.add_argument("--pdf", action="store_true",
                   help="Also write each field map as a PDF alongside the PNG. Off by "
                        "default because PDFs roughly double the render time and are "
                        "unused for the videos assembled from the PNG frames.")

    args = p.parse_args()

    if not args.all and not any([args.maps, args.videos, args.radius,
                                  args.eff_radius, args.eff_radius_error,
                                  args.norm_radius, args.max_velocity,
                                  args.compare_analytical]):
        p.error("Nothing to do — specify --all or at least one of "
                "--maps / --videos / --radius / --eff-radius / --eff-radius-error / "
                "--norm-radius / --max-velocity / --compare-analytical")

    if args.from_csv is not None:
        unsupported = args.all or args.maps or args.videos or args.radius or args.max_velocity
        if unsupported:
            p.error("--from-csv only supports --eff-radius / --eff-radius-error / --norm-radius "
                    "(maps, videos, --radius, and --max-velocity require plotfiles)")

    return args


_ALL_FIELDS = ["x_HI", "temperature", "n_e", "n_HI", "n_HII",
               "n_photon", "gasDensity", "velocity", "cs", "pressure"]


def main():
    args = parse_args()
    cfg  = load_toml(args.toml)

    prefix  = cfg.get("plotfile_prefix", "plt")
    outdir  = args.outdir or os.path.join(args.simdir, "Plots")
    analytical = build_analytical(cfg, radiation_pressure=args.radiation_pressure, mg=args.mg)

    map_kwargs = dict(
        plot_analytical=args.plot_analytical,
        plot_eff=args.plot_eff,
        plot_front=args.plot_front,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        nolog=args.nolog,
        redo=args.redo,
    )

    # ── CSV mode (no plotfiles required) ─────────────────────────────────────
    if args.from_csv is not None:
        import matplotlib.pyplot as pl
        from IFrontAnalysis import IonizationFrontFromCSV

        csv_path = (args.from_csv if os.path.isabs(args.from_csv) or os.path.exists(args.from_csv)
                    else os.path.join(args.simdir, args.from_csv))
        if not os.path.isfile(csv_path):
            sys.exit(f"Error: CSV not found: {csv_path}")

        os.makedirs(outdir, exist_ok=True)
        dx_pc = get_dx_pc(cfg)
        sim = IonizationFrontFromCSV(
            csv_path, dx_pc, analytical=analytical, use_radpres=args.radiation_pressure,
        )

        def savefig(fig, filename):
            fig.tight_layout()
            for ext in ("png", "pdf"):
                path = os.path.join(outdir, filename.rsplit(".", 1)[0] + f".{ext}")
                fig.savefig(path, dpi=150)
                print(f"Saved: {path}")
            pl.close(fig)

        if args.eff_radius:
            fig, ax = sim.plot_effective_radius_history(plot_analytical=True)
            ax.set_title("Effective radius history")
            savefig(fig, "eff_radius_history.png")

        if args.eff_radius_error:
            fig, ax = sim.plot_effective_radius_error_history()
            ax.set_title(r"Effective radius error $\Delta r / \Delta x$")
            savefig(fig, "eff_radius_error.png")

        if args.norm_radius:
            fig, ax = sim.plot_normalized_effective_radius_history()
            ax.set_title("Normalised effective radius")
            savefig(fig, "norm_radius_history.png")

        if args.compare_analytical:
            fig, ax = sim.plot_analytical_comparison_history()
            ax.set_title("Analytical solution comparison")
            savefig(fig, "analytical_comparison.png")
        return

    # ── single-snapshot mode ──────────────────────────────────────────────────
    if args.snapshot:
        from IFrontAnalysis import IonizationFrontSnapshot
        snap_path = os.path.join(args.simdir, args.snapshot)
        if not os.path.isdir(snap_path):
            sys.exit(f"Error: plotfile not found: {snap_path}")

        snap = IonizationFrontSnapshot(snap_path, outdir=outdir, analytical=analytical)
        fields = _ALL_FIELDS if args.all else (args.maps or [])
        for field in fields:
            snap.create_quantity_map(field, **map_kwargs)

        if args.all or args.radius:
            r_med, r_16, r_84 = snap.get_front_radius()
            print(f"Front radius: {r_med:.3f} pc  [{r_16:.3f}, {r_84:.3f}]")
        if args.all or args.eff_radius:
            r_eff = snap.get_effective_radius()
            print(f"Effective radius: {r_eff:.3f} pc")
        return

    # ── whole-directory mode ──────────────────────────────────────────────────
    import matplotlib.pyplot as pl
    from IFrontAnalysis import IonizationFront

    sim = IonizationFront(
        args.simdir,
        outdir=outdir,
        start_pattern=prefix,
        ending_number=args.end,
        step=args.step,
        analytical=analytical,
        save_pdf=args.pdf,
    )

    fields_maps   = _ALL_FIELDS if args.all else (args.maps   or [])
    fields_videos = _ALL_FIELDS if args.all else (args.videos or [])

    for field in fields_maps:
        sim.create_quantity_plots(field, **map_kwargs)

    for field in fields_videos:
        sim.create_quantity_video(field, fps=args.fps, **map_kwargs)

    def savefig(fig, filename):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(outdir, filename.rsplit(".", 1)[0] + f".{ext}")
            fig.savefig(path, dpi=150)
            print(f"Saved: {path}")
        pl.close(fig)

    if args.all or args.radius:
        fig, ax = sim.plot_radius_history(plot_analytical=True)
        ax.set_title("Front radius history")
        savefig(fig, "radius_history.png")

    if args.all or args.eff_radius:
        fig, ax = sim.plot_effective_radius_history(plot_analytical=True)
        ax.set_title("Effective radius history")
        savefig(fig, "eff_radius_history.png")

    if args.all or args.eff_radius_error:
        fig, ax = sim.plot_effective_radius_error_history()
        ax.set_title(r"Effective radius error $\Delta r / \Delta x$")
        savefig(fig, "eff_radius_error.png")

    if args.all or args.norm_radius:
        fig, ax = sim.plot_normalized_effective_radius_history()
        ax.set_title("Normalised effective radius")
        savefig(fig, "norm_radius_history.png")

    if args.all or args.max_velocity:
        fig, ax = sim.plot_max_velocity_history(nolog=args.nolog)
        ax.set_title("Max velocity history")
        savefig(fig, "max_velocity_history.png")

    if args.all or args.compare_analytical:
        fig, ax = sim.plot_analytical_comparison_history()
        ax.set_title("Analytical solution comparison")
        savefig(fig, "analytical_comparison.png")


if __name__ == "__main__":
    main()
