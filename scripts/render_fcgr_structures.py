from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

LOG = logging.getLogger("glycoml.phase3.render")


def parse_residue_list(spec: str) -> List[int]:
    values: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            values.extend(list(range(int(start), int(end) + 1)))
        else:
            values.append(int(chunk))
    return values


def render_structures(
    data_path: Path,
    pdb_template: Path,
    output_dir: Path,
    glycan_dir: Path | None,
    contact_residues: List[int],
) -> None:
    try:
        import pymol
        from pymol import cmd
    except Exception as exc:
        raise RuntimeError("PyMOL is required to render structures.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict[str, Any]] = {}

    pymol.finish_launching(["pymol", "-cq"])

    df = pd.read_csv(data_path)
    glycan_values = df.get("glycan_name")
    if glycan_values is not None:
        non_empty = (
            glycan_values.astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": "", "none": "", "unknown": ""})
        )
        if glycan_dir is None and (non_empty != "").any():
            LOG.warning(
                "No glycan directory provided; glycan-specific coordinates will not be loaded and "
                "all structures will look identical."
            )
    for _, row in df.iterrows():
        fcgr = str(row.get("fcgr_name", "unknown"))
        glycan = str(row.get("glycan_name", "unknown"))
        key = f"{fcgr}_{glycan}"
        safe_key = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key)

        cmd.reinitialize()
        cmd.load(str(pdb_template), "complex")
        cmd.hide("everything", "all")
        cmd.show("cartoon", "complex")
        cmd.color("slate", "complex and chain A")
        cmd.color("orange", "complex and chain B")

        if contact_residues:
            contact_sel = "complex and chain A and resi " + "+".join(map(str, contact_residues))
            cmd.show("surface", contact_sel)
            cmd.color("green", contact_sel)

        has_glycan = False
        if glycan_dir is not None:
            glycan_file = glycan_dir / f"{glycan}.pdb"
            if glycan_file.exists():
                cmd.load(str(glycan_file), "glycan")
                cmd.show("sticks", "glycan")
                cmd.color("cyan", "glycan and elem C")
                cmd.color("red", "glycan and elem O")
                has_glycan = True
            else:
                LOG.warning("Missing glycan PDB: %s", glycan_file)

        png_path = output_dir / f"{safe_key}.png"
        pdb_path = output_dir / f"{safe_key}.pdb"
        pse_path = output_dir / f"{safe_key}.pse"

        cmd.png(str(png_path), width=1200, height=1200, dpi=300, ray=1)
        cmd.save(str(pdb_path))
        cmd.save(str(pse_path))

        manifest[key] = {
            "fcgr_name": fcgr,
            "glycan_name": glycan,
            "png_path": str(png_path),
            "pdb_path": str(pdb_path),
            "pse_path": str(pse_path),
            "has_glycan": has_glycan,
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    LOG.info("Rendered %d structures", len(manifest))
    LOG.info("Manifest written to %s", manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Fcgr structures with glycans.")
    parser.add_argument("--data", required=True, help="CSV with fcgr_name and glycan_name.")
    parser.add_argument("--pdb-template", required=True, help="PDB template for Fc-Fcgr complex.")
    parser.add_argument("--output-dir", default="outputs/phase3_pymol", help="Output directory.")
    parser.add_argument("--glycan-dir", default="", help="Optional directory with glycan PDBs.")
    parser.add_argument(
        "--contact-residues",
        default="234-239,265-271,296-299",
        help="Comma-separated residue ranges for Fc contact surface.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_path = Path(args.data)
    pdb_template = Path(args.pdb_template)
    output_dir = Path(args.output_dir)
    glycan_dir = Path(args.glycan_dir) if args.glycan_dir else None
    residues = parse_residue_list(args.contact_residues)

    render_structures(
        data_path=data_path,
        pdb_template=pdb_template,
        output_dir=output_dir,
        glycan_dir=glycan_dir,
        contact_residues=residues,
    )


if __name__ == "__main__":
    main()
