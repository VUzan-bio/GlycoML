"""Extract glycan-only PDBs from full glycoprotein structures."""
from __future__ import annotations

import argparse
from pathlib import Path

GLYCAN_RESNAMES = "NAG+MAN+GAL+FUC+SIA+BMA+NDG+A2G"


def extract_glycans(input_dir: Path, output_dir: Path) -> int:
    try:
        import pymol
        from pymol import cmd
    except Exception as exc:
        raise RuntimeError("PyMOL is required.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    pymol.finish_launching(["pymol", "-cq"])

    extracted = 0
    for pdb_file in sorted(input_dir.glob("fc_fcgr_*_full.pdb")):
        glycan_name = pdb_file.stem.replace("fc_fcgr_", "").replace("_full", "")

        cmd.reinitialize()
        cmd.load(str(pdb_file), "complex")
        cmd.select("glycan", f"complex and resn {GLYCAN_RESNAMES}")

        if cmd.count_atoms("glycan") == 0:
            print(f"WARN: No glycan residues found in {pdb_file.name}")
            continue

        cmd.create("glycan_only", "glycan")
        output_path = output_dir / f"{glycan_name}.pdb"
        cmd.save(str(output_path), "glycan_only")
        print(f"OK: Extracted {output_path.name}")
        extracted += 1

    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract glycan-only PDBs from glycoprotein PDBs.")
    parser.add_argument("--input-dir", required=True, help="Directory with full glycoprotein PDBs")
    parser.add_argument("--output-dir", default="data/structures/glycans")
    args = parser.parse_args()

    total = extract_glycans(Path(args.input_dir), Path(args.output_dir))
    print(f"Done. Extracted {total} glycan PDBs.")


if __name__ == "__main__":
    main()
