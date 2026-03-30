"""
main_pbpk.py
============
Entry point for the Single-Cell Resolution PBPK simulation.

Usage — single simulation
─────────────────────────
  python main_pbpk.py
  python main_pbpk.py --Drug cetuximab --Target egfr --Days 120
  python main_pbpk.py --File data/Final_EGFR_Data.csv --Dose 400 --Regimen single

Usage — multi-dose (independent single-dose runs)
──────────────────────────────────────────────────
  python main_pbpk.py --multi --doses 0.1,1,10,100
  python main_pbpk.py --multi --doses 1,10,100,400 --Drug cetuximab --Target EGFR

  Each dose is simulated independently as a single-dose experiment.
  Output CSVs are saved to data/ with the pattern:
    pbpk_{target}_{drug}_{dose}mg.csv
"""

import argparse
from pbpk import Config, PBPKSimulator


def run_single_simulation(file: str, drug: str, target: str,
                           dose: float, regimen: str, days: int) -> str:
    """
    Run one PBPK simulation and return the output filename (relative to data/).
    """
    config = Config(
        drug_name=drug,
        target_name=target,
        dose_override=dose,
        regimen_override=regimen,
    )
    sim = PBPKSimulator(data_path=file, config=config)
    output_filename = (
        f"pbpk_{config.target_name}_{config.drug_name}"
        f"_{config.DOSE_MG_M2}mg.csv"
    )
    sim.run_all_and_save(output_filename=output_filename, days=days)
    return output_filename


def main():
    parser = argparse.ArgumentParser(description="Single-Cell PBPK Platform")
    parser.add_argument('--File',    type=str,   default='data/Final_EGFR_Data.csv',
                        help='Input CSV from Bayesian pipeline')
    parser.add_argument('--Drug',    type=str,   default='Cetuximab',
                        help='Drug name (must exist in config.py DRUG_DB)')
    parser.add_argument('--Target',  type=str,   default='EGFR',
                        help='Target name (must exist in config.py TARGET_DB)')
    parser.add_argument('--Dose',    type=float, default=None,
                        help='Custom dose mg/m² (overrides drug default)')
    parser.add_argument('--Regimen', type=str,   default='single',
                        help='Dosing regimen (default: single). '
                             'Options: single, q1w, q2w, loading_q1w, …')
    parser.add_argument('--Days',    type=int,   default=120,
                        help='Simulation duration (days)')

    # ── Multi-dose arguments ───────────────────────────────────────────────
    parser.add_argument('--multi',   action='store_true',
                        help='Run multiple independent single-dose simulations')
    parser.add_argument('--doses',   type=str,   default='0.1,1,10,100',
                        help='Comma-separated dose list for --multi (mg/m²). '
                             'Default: "0.1,1,10,100"')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f" Single-Cell Resolution PBPK Simulation")
    print(f"{'='*60}")

    # ── Multi-dose mode ────────────────────────────────────────────────────
    if args.multi:
        doses = [float(d.strip()) for d in args.doses.split(',')]
        if not doses:
            print("[ERROR] --doses produced an empty list. "
                  "Provide e.g. --doses 0.1,1,10,100")
            return

        print(f"\n[*] Multi-dose mode")
        print(f"    Drug    : {args.Drug}")
        print(f"    Target  : {args.Target}")
        print(f"    Doses   : {doses} mg/m²")
        print(f"    Days    : {args.Days}")
        print(f"    Regimen : single (forced — each dose is independent)")

        saved_files = []
        failed_doses = []

        for i, dose in enumerate(doses, 1):
            print(f"\n{'─'*60}")
            print(f"  Simulation {i}/{len(doses)}  —  Dose: {dose} mg/m²")
            print(f"{'─'*60}")
            try:
                fname = run_single_simulation(
                    file=args.File,
                    drug=args.Drug,
                    target=args.Target,
                    dose=dose,
                    regimen='single',
                    days=args.Days,
                )
                saved_files.append((dose, fname))
                print(f"  [✓] Done → data/{fname}")
            except Exception as exc:
                print(f"  [✗] Dose {dose} failed: {exc}")
                failed_doses.append(dose)

        # Summary
        print(f"\n{'='*60}")
        print(f" Multi-dose simulations complete")
        print(f" {len(saved_files)}/{len(doses)} successful")
        print(f"\n Output files:")
        for dose, fname in saved_files:
            print(f"   [{dose:>8.2f} mg/m²]  data/{fname}")
        if failed_doses:
            print(f"\n Failed doses: {failed_doses}")
        print(f"\n Visualise with:")
        doses_str = ','.join(str(d) for d in doses)
        print(f"   python visualisation.py "
              f"--Target {args.Target} --Drug {args.Drug} "
              f"--multi --doses {doses_str}")
        print(f"{'='*60}")

    # ── Single simulation mode ─────────────────────────────────────────────
    else:
        regimen = args.Regimen or 'single'
        config = Config(
            drug_name=args.Drug,
            target_name=args.Target,
            dose_override=args.Dose,
            regimen_override=regimen,
        )
        sim = PBPKSimulator(data_path=args.File, config=config)
        output_filename = (
            f"pbpk_{config.target_name}_{config.drug_name}"
            f"_{config.DOSE_MG_M2}mg.csv"
        )
        sim.run_all_and_save(output_filename=output_filename, days=args.Days)

        print(f"\n{'='*60}")
        print(f" Simulation Complete")
        print(f" Output: data/{output_filename}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()