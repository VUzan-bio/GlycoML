#!/usr/bin/env python
"""Generate FcγR-Fc training data from glycan structure-function heuristics."""

from __future__ import annotations

import numpy as np
import pandas as pd


FCGR_SEQUENCES = {
    "FcγRIIA-H131": "MWQLLLPTALLLLVSAGMRTEDLPKAVVFLEPQWYRVLEKDSVTLKCQGAYSPEDNSTQWFHNESLISSQASSYFIDAATVDDSGEYRCQTNLSTLSDPVQLEVHIGWLLLQAPRWVFKEEDPIHLRCHSWKNTALHKVTYLQNGKGRKYFHHNSDFYIPKATLKDSGSYFCRGLVGSKNVSSETVNITITQGLAVSTISSFFPPGYQ",
    "FcγRIIA-R131": "MWQLLLPTALLLLVSAGMRTEDLPKAVVFLEPQWYRVLEKDSVTLKCQGAYSPEDNSTQWFHNESLISSQASSYFIDAATVDDSGEYRCQTNLSTLSDPVQLEVHIGWLLLQAPRWVFKEEDPIRLRCHSWKNTALHKVTYLQNGKGRKYFHHNSDFYIPKATLKDSGSYFCRGLVGSKNVSSETVNITITQGLAVSTISSFFPPGYQ",
    "FcγRIIIA-V158": "MWLLHPALLLLASAGPMSVFHSGGQMHHPPPPILPHHHHPLLPPDQTYYRFVENKGKSPWPVAYNVTYRMKKPLQFTTQEDYSHDYSTLNRVLQLENATSSNFSRDPKYVYQFKGMWNYFHAGLHDKQKHKSWSAWGVFGQGEIPDLLFIFHYRETDLQSVYFVKATVVSPSESKSQPVTCRVLGFVPRDKYFDYMDLSLPRIHFVVVVSENDGKGDKASNEKVIFNTDSHPRIRYQVRLFFQHLGEISAREFHRQGFMFKNRTLNFSKDIQVSNLTSISNVDGEFYCDPGYEYY",
    "FcγRIIIA-F158": "MWLLHPALLLLASAGPMSVFHSGGQMHHPPPPILPHHHHPLLPPDQTYYRFVENKGKSPWPVAYNVTYRMKKPLQFTTQEDYSHDYSTLNRVLQLENATSSNFSRDPKYVYQFKGMWNYFHAGLHDKQKHKSWSAWGVFGQGEIPDLLFIFHYRETDLQSVYFVKATVVSPSESKSQPVTCRVLGFVPRDKYFDYMDLSLPRIHFVVVVSENDGKGDKASNEKVIFNTDSHPRIRYQVRLFFQHLGEISAREFHRQGFMFKNRTLNFSKDIQVSNLTSISNVDGEFYCDPGYEYY",
    "FcγRIIIB": "MWLLHPALLLLASAGPMSVFHSGGQMHHPPPPILPHHHHPLLPPDQTYYRFVENKGKSPWPVAYNVTYRMKKPLQFTTQEDYSHDYSTLNRVLQLENATSSNFSRDPKYVYQFKGMWNYFHAGLHDKQKHKSWSAWGVFGQGEIPDLLFIFHYRETDLQSVYFVKATVVSPSESKSQPVTCRVLGFVPRDKYFDYMDLSLPRIHFVVVVSENDGKGDKASNEKVIFNTDSHPRIRYQVRLFFQHLGEISAREFHRQGFMFKNRTLNFSKDIQVSNLTSISNVDGEFYCDPGYEYY",
}

IGG1_FC = (
    "APELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYN"
    "STYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDELTKNQV"
    "SLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMH"
    "EALHNHYTQKSLSLSPGK"
)

GLYCAN_STRUCTURES = {
    "G0": ("Glc2Man3GlcNAc4", 1.0),
    "G0F": ("Glc2Man3GlcNAc4Fuc1", 8.0),
    "G1F": ("Glc2Man3GlcNAc4Fuc1Gal1", 4.0),
    "G2F": ("Glc2Man3GlcNAc4Fuc1Gal2", 2.0),
    "G0FS": ("Glc2Man3GlcNAc4Fuc1Neu5Ac1", 0.3),
    "G1FS": ("Glc2Man3GlcNAc4Fuc1Gal1Neu5Ac1", 0.2),
    "G2FS": ("Glc2Man3GlcNAc4Fuc1Gal2Neu5Ac2", 0.1),
    "G0FB": ("Glc2Man3GlcNAc5Fuc1", 12.0),
    "G1FB": ("Glc2Man3GlcNAc5Fuc1Gal1", 8.0),
    "G0-Fuc": ("Glc2Man3GlcNAc4", 15.0),
    "G1-Fuc": ("Glc2Man3GlcNAc4Gal1", 10.0),
    "Man5": ("Man5GlcNAc2", 0.5),
    "Man8": ("Man8GlcNAc2", 0.3),
}


def generate_training_data() -> pd.DataFrame:
    data = []

    fcgr_name = "FcγRIIIA-V158"
    fcgr_seq = FCGR_SEQUENCES[fcgr_name]
    for glycan_name, (glycan_struct, rel_affinity) in GLYCAN_STRUCTURES.items():
        base_kd = 450.0
        kd = base_kd / rel_affinity
        kd_noisy = kd * np.random.uniform(0.8, 1.2)
        data.append(
            {
                "fcgr_name": fcgr_name,
                "fcgr_sequence": fcgr_seq,
                "fc_sequence": IGG1_FC,
                "glycan_name": glycan_name,
                "glycan_structure": glycan_struct,
                "binding_kd_nm": kd_noisy,
                "log_kd": np.log10(kd_noisy),
                "binding_affinity": 1.0 / kd_noisy,
                "source": "synthetic_literature",
            }
        )

    fcgr_name = "FcγRIIIA-F158"
    fcgr_seq = FCGR_SEQUENCES[fcgr_name]
    for glycan_name, (glycan_struct, rel_affinity) in GLYCAN_STRUCTURES.items():
        base_kd = 1500.0
        kd = base_kd / rel_affinity
        kd_noisy = kd * np.random.uniform(0.8, 1.2)
        data.append(
            {
                "fcgr_name": fcgr_name,
                "fcgr_sequence": fcgr_seq,
                "fc_sequence": IGG1_FC,
                "glycan_name": glycan_name,
                "glycan_structure": glycan_struct,
                "binding_kd_nm": kd_noisy,
                "log_kd": np.log10(kd_noisy),
                "binding_affinity": 1.0 / kd_noisy,
                "source": "synthetic_literature",
            }
        )

    for variant in ["H131", "R131"]:
        fcgr_name = f"FcγRIIA-{variant}"
        fcgr_seq = FCGR_SEQUENCES[fcgr_name]
        for glycan_name, (glycan_struct, rel_affinity) in GLYCAN_STRUCTURES.items():
            base_kd = 2000.0 if variant == "H131" else 5000.0
            kd = base_kd / (rel_affinity * 0.5)
            kd_noisy = kd * np.random.uniform(0.8, 1.2)
            data.append(
                {
                    "fcgr_name": fcgr_name,
                    "fcgr_sequence": fcgr_seq,
                    "fc_sequence": IGG1_FC,
                    "glycan_name": glycan_name,
                    "glycan_structure": glycan_struct,
                    "binding_kd_nm": kd_noisy,
                    "log_kd": np.log10(kd_noisy),
                    "binding_affinity": 1.0 / kd_noisy,
                    "source": "synthetic_literature",
                }
            )

    df = pd.DataFrame(data)
    return df


def main() -> None:
    np.random.seed(42)
    print("Generating FcγR-Fc training data from literature relationships...")
    df = generate_training_data()

    print(f"✓ Generated {len(df)} training examples")
    print(f"  - {df['fcgr_name'].nunique()} FcγR variants")
    print(f"  - {df['glycan_name'].nunique()} glycan structures")
    print(f"  - KD range: {df['binding_kd_nm'].min():.1f} - {df['binding_kd_nm'].max():.1f} nM")

    output = "data/processed/fcgr_fc_training_data.csv"
    df.to_csv(output, index=False)
    print(f"✓ Saved to {output}")

    print("\nBinding affinity by FcγR (mean KD in nM):")
    summary = df.groupby("fcgr_name")["binding_kd_nm"].agg(["mean", "min", "max"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
