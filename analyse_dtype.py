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


def build_analytical(cfg):
    from IFrontAnalysis import DTypeAnalytical
    Q   = cfg["stromgen"]["Q"]
    n_H = cfg["stromgen"]["primary_species_2"]
    return DTypeAnalytical(Q, n_H)


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse Quokka D-type ionization front simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("toml",   help="Path to the Quokka input TOML file")
    p.add_argument("simdir", help="Directory containing the plotfiles")

    # -- scope --
    scope = p.add_mutually_exclusive_group()
    scope.add_argument(
        "--snapshot", metavar="PLOTFILE",
        help="Analyse a single named plotfile only (no videos produced)",
    )

    # -- what to make --
    p.add_argument(
        "--all", action="store_true",
        help="Make everything: all field maps, all videos, radius and effective-radius plots",
    )
    p.add_argument(
        "--maps", nargs="+", metavar="FIELD",
        choices=["x_HI", "temperature", "n_e", "n_HI", "n_HII",
                 "n_photon", "gasDensity", "velocity", "cs", "pressure"],
        help="Field maps to create (slice plots)",
    )
    p.add_argument(
        "--videos", nargs="+", metavar="FIELD",
        choices=["x_HI", "temperature", "n_e", "n_HI", "n_HII",
                 "n_photon", "gasDensity", "velocity", "cs", "pressure"],
        help="Fields to animate as videos (ignored with --snapshot)",
    )
    p.add_argument("--radius",     action="store_true", help="Plot front radius history")
    p.add_argument("--eff-radius", action="store_true", help="Plot effective radius history")
    p.add_argument("--eff-radius-error", action="store_true",
                   help="Plot effective radius error (r_eff - r_analytical) / dx")
    p.add_argument("--norm-radius", action="store_true",
                   help="Plot normalised effective radius r_eff / r_s vs t / t_s")

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

    args = p.parse_args()

    if not args.all and not any([args.maps, args.videos, args.radius,
                                  args.eff_radius, args.eff_radius_error,
                                  args.norm_radius]):
        p.error("Nothing to do — specify --all or at least one of "
                "--maps / --videos / --radius / --eff-radius / --eff-radius-error / --norm-radius")
    return args


_ALL_FIELDS = ["x_HI", "temperature", "n_e", "n_HI", "n_HII",
               "n_photon", "gasDensity", "velocity", "cs", "pressure"]


def main():
    args = parse_args()
    cfg  = load_toml(args.toml)

    prefix  = cfg.get("plotfile_prefix", "plt")
    outdir  = args.outdir or os.path.join(args.simdir, "Plots")
    analytical = build_analytical(cfg)

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


if __name__ == "__main__":
    main()
