"""
main_pbpk.py
============
Entry point for the Single-Cell Resolution PBPK simulation.

Usage
─────
python main_pbpk.py
python main_pbpk.py --Drug cetuximab --Target egfr --Days 120
python main_pbpk.py --File data/Final_EGFR_Data.csv --Dose 400
"""

import argparse
from pbpk import Config, PBPKSimulator


def main():
    parser = argparse.ArgumentParser(description="Single-Cell PBPK Platform")
    parser.add_argument('--File', type=str, default='data/Final_EGFR_Data.csv')
    parser.add_argument('--Drug', type=str, default='Cetuximab')
    parser.add_argument('--Target', type=str, default='EGFR')
    parser.add_argument('--Dose', type=float, default=None,
    help='Custom dose mg/m² (overrides drug default)')
    parser.add_argument('--Regimen', type=str, default=None,
    help='Override regimen (e.g., single, q1w, loading_q1w)')
    parser.add_argument('--Days', type=int, default=120)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f" Single-Cell Resolution PBPK Simulation")
    print(f"{'='*60}")

    config = Config(
    drug_name=args.Drug,
    target_name=args.Target,
    dose_override=args.Dose,
    regimen_override=args.Regimen,
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
