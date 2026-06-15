#!/usr/bin/env python3
"""
oeuvre.__main__ — allows ``python -m oeuvre [target]``

With no arguments: opens the GUI.
With a target directory: opens the GUI pre-selected on that target.
With --headless: runs CLI-only (no GUI).
"""

import os
import argparse

from .pipeline import PipelineConfig, run_pipeline
from .natural_narrowband import STRETCH_TARGET, SCNR_AMOUNT, STAR_DESAT, SAT_BOOST


def main():
    parser = argparse.ArgumentParser(
        prog='oeuvre',
        description='Oeuvre — SHO Narrowband Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Launch modes:
  oeuvre                  Open GUI, pick a target
  oeuvre IC_1805          Open GUI pre-selected on IC_1805
  oeuvre IC_1805 --headless   CLI-only processing (no GUI)
        """,
    )

    parser.add_argument('target', nargs='?', default=None,
                        help='Target directory (optional — opens GUI if omitted)')

    # Mode
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI (CLI-only)')

    # Data location
    parser.add_argument('--workspace', default=None,
                        help='Data root (targets, darks/, StarNetv2CLI_MacOS/). '
                             'Overrides OEUVRE_WORKSPACE; default ~/oeuvre-astro')

    # Mosaic prep
    parser.add_argument('--cluster-radius', type=float, default=0.15,
                        help='Cluster radius in degrees (default: 0.15)')

    # Siril control
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Skip calibrate/register/stack (reuse existing masters)')

    # SHO pipeline
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: target root)')
    parser.add_argument('--render', choices=['balanced', 'truthful'],
                        default='balanced',
                        help='Render intent: balanced (default) or strict truthful')
    parser.add_argument('--stretch-target', type=float, default=STRETCH_TARGET)
    parser.add_argument('--scnr-amount', type=float, default=SCNR_AMOUNT)
    parser.add_argument('--star-desat', type=float, default=STAR_DESAT)
    parser.add_argument('--sat-boost', type=float, default=SAT_BOOST)
    parser.add_argument('--clear-cache', action='store_true',
                        help='Delete _sho_work cache before SHO processing')
    parser.add_argument('--recolor-only', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--flatten-background', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--hue-strength', type=float, default=0.40,
                        help=argparse.SUPPRESS)
    parser.add_argument('--oiii-factor', type=float, default=0.32,
                        help=argparse.SUPPRESS)
    parser.set_defaults(truthful_mode=False, hubbleize=True)
    parser.add_argument('--truthful-mode', dest='truthful_mode', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--artistic-mode', dest='truthful_mode', action='store_false',
                        help=argparse.SUPPRESS)
    parser.add_argument('--hubbleize', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--hubbleize-strength', type=float, default=0.45,
                        help=argparse.SUPPRESS)
    parser.add_argument('--star-consensus', choices=['auto', 'off', 'soft', 'strict'],
                        default='auto',
                        help=argparse.SUPPRESS)

    # Preview
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--interactive', action='store_true')

    args = parser.parse_args()

    # A --workspace flag is just a convenient way to set OEUVRE_WORKSPACE, so the
    # single source of truth in oeuvre.config flows to every module.
    if args.workspace:
        os.environ['OEUVRE_WORKSPACE'] = os.path.abspath(
            os.path.expanduser(args.workspace))

    if args.headless:
        # CLI mode — target required
        if not args.target:
            parser.error('--headless requires a target directory')

        cfg = PipelineConfig(
            target=args.target,
            cluster_radius=args.cluster_radius,
            no_preprocess=args.no_preprocess,
            output_dir=args.output_dir,
            stretch_target=args.stretch_target,
            scnr_amount=args.scnr_amount,
            star_desat=args.star_desat,
            sat_boost=args.sat_boost,
            clear_cache=args.clear_cache,
            recolor_only=args.recolor_only,
            flatten_background=args.flatten_background,
            hue_strength=args.hue_strength,
            oiii_factor=args.oiii_factor,
            truthful_mode=((args.render == 'truthful')
                           or (args.truthful_mode and not args.hubbleize)),
            hubbleize=(args.render != 'truthful') and args.hubbleize,
            hubbleize_strength=args.hubbleize_strength,
            star_consensus=args.star_consensus,
            no_preview=args.no_preview,
            interactive=args.interactive,
        )
        output = run_pipeline(cfg)
        return output
    else:
        # GUI mode
        from .gui import launch_gui
        launch_gui(auto_target=args.target)


if __name__ == '__main__':
    main()
