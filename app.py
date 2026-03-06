import json
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Module 3 (Compression + Search)
# -----------------------------
def generate_bwt(sequence: str):
    """Generate Burrows-Wheeler Transform and sorted rotation table."""
    seq = sequence.upper() + "$"
    rotations = [seq[i:] + seq[:i] for i in range(len(seq))]
    rotations.sort()
    bwt_string = "".join(rotation[-1] for rotation in rotations)
    return bwt_string, rotations


def inverse_bwt(bwt_string: str) -> str:
    """Reconstruct original string (without terminal '$') from BWT."""
    table = [""] * len(bwt_string)
    for _ in range(len(bwt_string)):
        table = [bwt_string[i] + table[i] for i in range(len(bwt_string))]
        table.sort()

    for row in table:
        if row.endswith("$"):
            return row[:-1]
    return ""


def build_fm_index(sequence: str):
    """Build a simple FM-index (suffix array + BWT + C + Occ prefix table)."""
    text = sequence.upper() + "$"
    suffix_array = sorted(range(len(text)), key=lambda i: text[i:])
    bwt = "".join(text[i - 1] if i > 0 else "$" for i in suffix_array)

    alphabet = sorted(set(text))
    char_counts = {c: 0 for c in alphabet}
    for c in text:
        char_counts[c] += 1

    c_table = {}
    running_total = 0
    for c in alphabet:
        c_table[c] = running_total
        running_total += char_counts[c]

    occ = {c: [0] * (len(bwt) + 1) for c in alphabet}
    for i, ch in enumerate(bwt, start=1):
        for c in alphabet:
            occ[c][i] = occ[c][i - 1]
        occ[ch][i] += 1

    return {
        "text": text,
        "suffix_array": suffix_array,
        "bwt": bwt,
        "alphabet": alphabet,
        "c_table": c_table,
        "occ": occ,
    }


def fm_backward_search(pattern: str, fm_index: dict):
    """Return sorted start positions of pattern occurrences using backward search."""
    pattern = pattern.upper()
    if not pattern:
        return []

    bwt = fm_index["bwt"]
    c_table = fm_index["c_table"]
    occ = fm_index["occ"]
    suffix_array = fm_index["suffix_array"]

    if any(ch not in c_table for ch in pattern):
        return []

    l, r = 0, len(bwt)
    for ch in reversed(pattern):
        l = c_table[ch] + occ[ch][l]
        r = c_table[ch] + occ[ch][r]
        if l >= r:
            return []

    return sorted(suffix_array[l:r])


# -----------------------------
# Module 1 (Graph-based Pangenome)
# -----------------------------
def build_pangenome_graph(sequences, k=3):
    """Build weighted directed De Bruijn-style graph from DNA sequences."""
    graph = nx.DiGraph()

    for seq in sequences:
        seq = seq.upper()
        if len(seq) < k + 1:
            continue

        for i in range(len(seq) - k):
            kmer_1 = seq[i:i + k]
            kmer_2 = seq[i + 1:i + k + 1]

            if graph.has_edge(kmer_1, kmer_2):
                graph[kmer_1][kmer_2]["weight"] += 1
            else:
                graph.add_edge(kmer_1, kmer_2, weight=1)

    return graph


def compute_conservation_profile(sequences):
    """Compute per-position conservation as major allele frequency (0..1)."""
    if not sequences:
        return pd.DataFrame(columns=["Position", "Conservation", "MajorBase"])

    sequences = [s.upper() for s in sequences if s]
    if not sequences:
        return pd.DataFrame(columns=["Position", "Conservation", "MajorBase"])

    min_len = min(len(s) for s in sequences)
    if min_len == 0:
        return pd.DataFrame(columns=["Position", "Conservation", "MajorBase"])

    rows = []
    for i in range(min_len):
        column = [s[i] for s in sequences]
        series = pd.Series(column)
        counts = series.value_counts()
        major_base = counts.index[0]
        conservation = counts.iloc[0] / len(column)
        rows.append({"Position": i + 1, "Conservation": conservation, "MajorBase": major_base})

    return pd.DataFrame(rows)


# -----------------------------
# Module 2 (DeepNCV + Explorer)
# -----------------------------
class SimpleDNA_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


def encode_sequence(seq: str) -> torch.Tensor:
    """Convert DNA string to one-hot tensor shape: (1, 4, L)."""
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }
    encoded = [mapping.get(base.upper(), [0.25, 0.25, 0.25, 0.25]) for base in seq]
    return torch.tensor(encoded, dtype=torch.float32).T.unsqueeze(0)


def get_model(seed=42):
    torch.manual_seed(seed)
    model = SimpleDNA_CNN()
    model.eval()
    return model


def predict_functional_impact(sequence: str, seed: int = 42) -> float:
    model = get_model(seed=seed)
    with torch.no_grad():
        input_tensor = encode_sequence(sequence)
        output = model(input_tensor)
    return output.item()


def compute_saliency(sequence: str, seed: int = 42):
    """Simple gradient saliency: max channel gradient per position."""
    model = get_model(seed=seed)
    input_tensor = encode_sequence(sequence)
    input_tensor.requires_grad_(True)

    output = model(input_tensor)
    output.backward(torch.ones_like(output))

    grads = input_tensor.grad.detach().abs().squeeze(0)  # shape [4, L]
    saliency = grads.max(dim=0).values.cpu().numpy()  # [L]
    return saliency


def sanitize_dna_sequence(text: str) -> str:
    return "".join(text.split()).upper()


def mutation_scan(sequence: str, seed: int = 42):
    """All single-base substitutions with impact and delta vs baseline."""
    sequence = sanitize_dna_sequence(sequence)
    bases = ["A", "C", "G", "T"]
    baseline = predict_functional_impact(sequence, seed=seed)

    rows = []
    for idx, ref in enumerate(sequence):
        for alt in bases:
            if alt == ref:
                continue
            mutant = sequence[:idx] + alt + sequence[idx + 1:]
            score = predict_functional_impact(mutant, seed=seed)
            rows.append(
                {
                    "Position": idx + 1,
                    "Ref": ref,
                    "Alt": alt,
                    "MutantSequence": mutant,
                    "ImpactScore": score,
                    "DeltaVsBaseline": score - baseline,
                }
            )

    return baseline, pd.DataFrame(rows)


def make_heatmap_matrix(scan_df: pd.DataFrame, sequence: str):
    bases = ["A", "C", "G", "T"]
    length = len(sequence)
    matrix = np.full((4, length), np.nan, dtype=float)

    for _, row in scan_df.iterrows():
        b_idx = bases.index(row["Alt"])
        p_idx = int(row["Position"]) - 1
        matrix[b_idx, p_idx] = row["DeltaVsBaseline"]

    return bases, matrix


# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(
    page_title="PanGen-AI Suite | Computational Genomics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("PanGen-AI Suite")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Select a Module:",
    [
        "Home - Overview",
        "Module 1: Pangenome Explorer",
        "Module 2: DeepNCV (AI Variant Caller)",
        "Module 3: Geno-Compressor (BWT)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Developed by: **Yashwant Nama**\n\n"
    "Computational Researcher\n\n"
    "Target: Advanced Genomic Data Structures & Deep Learning"
)


if page == "Home - Overview":
    st.title("PanGen-AI Suite: Integrated Computational Genomics")
    st.markdown(
        """
Welcome to the **PanGen-AI Suite**. This toolkit bridges raw genomic data,
evolutionary conservation, and artificial intelligence.

### Available Modules
1. **Pangenome Explorer:** Build graph-based k-mer pangenome structure + conservation profile.
2. **DeepNCV:** Score non-coding DNA and explore mutation impact with heatmaps.
3. **Geno-Compressor:** BWT transform + inverse + FM-index pattern search.
"""
    )

elif page == "Module 1: Pangenome Explorer":
    st.title("Module 1: Pangenome Graph Explorer")
    st.subheader("Graph-Based Sequence Assembly + Conservation Analysis")
    st.markdown(
        """
Modern pangenomics represents genomes as **sequence graphs** instead of linear strings.
This module builds a weighted directed k-mer graph and computes a per-position conservation profile.
"""
    )

    default_seqs = "ATGCGTAC\nATGCATAC\nATGCGTAC\nATGCCTAC"
    seq_input = st.text_area("Enter DNA Sequences (one per line):", value=default_seqs, height=120)
    kmer_size = st.slider("Select k-mer size (Resolution):", min_value=2, max_value=7, value=3)

    if st.button("Generate Pangenome Graph"):
        sequences = [sanitize_dna_sequence(s) for s in seq_input.split("\n") if s.strip()]

        if sequences:
            with st.spinner("Constructing graph and conservation profile..."):
                graph = build_pangenome_graph(sequences, k=kmer_size)
                num_nodes = graph.number_of_nodes()
                num_edges = graph.number_of_edges()
                conservation_df = compute_conservation_profile(sequences)

            st.success("Pangenome analysis complete.")
            col1, col2, col3 = st.columns(3)
            col1.metric("Input Sequences", len(sequences))
            col2.metric("Graph Nodes (k-mers)", num_nodes)
            col3.metric("Graph Edges", num_edges)

            if num_nodes > 0 and num_edges > 0:
                fig, ax = plt.subplots(figsize=(11, 6))
                pos = nx.spring_layout(graph, seed=42)
                weights = [graph[u][v]["weight"] * 1.8 for u, v in graph.edges()]

                nx.draw_networkx_nodes(graph, pos, node_size=720, node_color="#4CAF50", alpha=0.92, ax=ax)
                nx.draw_networkx_labels(graph, pos, font_size=9, font_color="white", font_weight="bold", ax=ax)
                nx.draw_networkx_edges(graph, pos, width=weights, edge_color="#555555", arrowsize=20, ax=ax)

                ax.set_title(f"Pangenome Sequence Graph (k={kmer_size})", fontsize=14)
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

                st.info(
                    "💡 Nodes = k-mers, edges = adjacency. Thicker edges indicate shared/conserved paths. "
                    "Branches suggest SNPs/variants/alternative sequence paths."
                )
            else:
                st.warning("No graph could be built. Ensure each sequence has length at least k+1.")

            if not conservation_df.empty:
                cfig, cax = plt.subplots(figsize=(11, 3.8))
                cax.plot(
                    conservation_df["Position"],
                    conservation_df["Conservation"],
                    marker="o",
                    linewidth=1.8,
                    color="#1f77b4",
                )
                cax.set_ylim(0, 1.05)
                cax.set_xlabel("Position")
                cax.set_ylabel("Conservation")
                cax.set_title("Per-position Conservation Profile (Major Allele Frequency)")
                cax.grid(alpha=0.3)
                st.pyplot(cfig)
                plt.close(cfig)

                with st.expander("Show conservation table"):
                    st.dataframe(conservation_df, use_container_width=True)
            else:
                st.warning("Could not compute conservation profile.")
        else:
            st.warning("Please enter at least one DNA sequence.")

elif page == "Module 2: DeepNCV (AI Variant Caller)":
    st.title("Module 2: DeepNCV")
    st.subheader("Prediction + Variant Effect Explorer + Explainability")

    tabs = st.tabs([
        "Single Prediction",
        "Variant Effect Explorer",
        "Batch/Saturation CSV",
    ])

    with tabs[0]:
        dna_sequence = st.text_area(
            "Enter DNA Sequence (A, T, C, G):",
            height=130,
            value="ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAG",
            key="single_seq",
        )
        seed = st.number_input("Reproducibility Seed", min_value=0, max_value=100000, value=42, step=1)
        show_explainability = st.checkbox("Show explainability (saliency per nucleotide)", value=True)

        if st.button("Predict Functional Impact", key="predict_single"):
            clean_seq = sanitize_dna_sequence(dna_sequence)
            if len(clean_seq) < 10:
                st.error("Please enter a valid DNA sequence (minimum 10 base pairs).")
            else:
                with st.spinner("Running model inference..."):
                    impact_score = predict_functional_impact(clean_seq, seed=int(seed))

                confidence_pct = impact_score * 100
                st.success("Inference complete.")
                st.metric("Functional Impact Probability", f"{confidence_pct:.2f}%")

                if confidence_pct > 50:
                    st.info("🧠 Model Prediction: **Active Functional Region**")
                else:
                    st.warning("🧠 Model Prediction: **Inactive/Neutral Region**")

                st.write(f"Sequence Length: {len(clean_seq)} bp")
                st.write(f"Input Tensor Shape: `[1, 4, {len(clean_seq)}]`")

                if show_explainability:
                    saliency = compute_saliency(clean_seq, seed=int(seed))
                    sfig, sax = plt.subplots(figsize=(11, 3.8))
                    sax.plot(np.arange(1, len(clean_seq) + 1), saliency, color="#d62728", linewidth=1.8)
                    sax.set_xlabel("Position")
                    sax.set_ylabel("Saliency")
                    sax.set_title("Nucleotide Importance (Gradient Saliency)")
                    sax.grid(alpha=0.3)
                    st.pyplot(sfig)
                    plt.close(sfig)

                report = {
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                    "seed": int(seed),
                    "sequence_length": len(clean_seq),
                    "sequence": clean_seq,
                    "impact_score": impact_score,
                    "impact_percent": confidence_pct,
                    "model": "SimpleDNA_CNN (demo, untrained)",
                }
                st.download_button(
                    "Download reproducibility JSON",
                    data=json.dumps(report, indent=2),
                    file_name="deepncv_reproducible_run.json",
                    mime="application/json",
                )

    with tabs[1]:
        seq_for_scan = st.text_area(
            "Reference DNA Sequence for mutation scan:",
            height=130,
            value="ATGCGTACGTAGCTAGCTAG",
            key="scan_seq",
        )
        scan_seed = st.number_input("Seed", min_value=0, max_value=100000, value=42, step=1, key="scan_seed")

        if st.button("Run Single-Mutation Scan", key="run_scan"):
            ref_seq = sanitize_dna_sequence(seq_for_scan)
            if len(ref_seq) < 10:
                st.error("Please provide at least 10 bp for mutation scanning.")
            else:
                with st.spinner("Generating all single-base substitutions..."):
                    baseline, scan_df = mutation_scan(ref_seq, seed=int(scan_seed))

                st.success("Mutation scan complete.")
                st.metric("Baseline Impact Score", f"{baseline:.4f}")
                st.dataframe(scan_df[["Position", "Ref", "Alt", "ImpactScore", "DeltaVsBaseline"]], use_container_width=True)

                bases, heatmap_matrix = make_heatmap_matrix(scan_df, ref_seq)
                hfig, hax = plt.subplots(figsize=(12, 3.8))
                im = hax.imshow(heatmap_matrix, aspect="auto", cmap="coolwarm")
                hax.set_yticks(range(len(bases)))
                hax.set_yticklabels(bases)
                hax.set_xticks(range(len(ref_seq)))
                hax.set_xticklabels(range(1, len(ref_seq) + 1), fontsize=8)
                hax.set_xlabel("Position")
                hax.set_ylabel("Alternative Base")
                hax.set_title("Variant Effect Heatmap (Δ vs baseline)")
                plt.colorbar(im, ax=hax, label="Delta Impact")
                st.pyplot(hfig)
                plt.close(hfig)

    with tabs[2]:
        batch_sequences = st.text_area(
            "Batch DNA sequences (one per line):",
            height=120,
            value="ATGCGTACGTAG\nATGCATACGTAG\nATGCGTACCTAG",
            key="batch_seq",
        )
        batch_seed = st.number_input("Batch Seed", min_value=0, max_value=100000, value=42, step=1, key="batch_seed")

        if st.button("Run Batch Predictions", key="batch_predict"):
            seqs = [sanitize_dna_sequence(s) for s in batch_sequences.split("\n") if s.strip()]
            valid = [s for s in seqs if len(s) >= 10]

            if not valid:
                st.error("No valid sequences found (need length >= 10).")
            else:
                rows = []
                for i, seq in enumerate(valid, start=1):
                    score = predict_functional_impact(seq, seed=int(batch_seed))
                    rows.append(
                        {
                            "SequenceID": f"Seq_{i}",
                            "Length": len(seq),
                            "ImpactScore": score,
                            "ImpactPercent": score * 100,
                            "Sequence": seq,
                        }
                    )

                batch_df = pd.DataFrame(rows)
                st.dataframe(batch_df, use_container_width=True)

                csv_data = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Batch Predictions CSV",
                    data=csv_data,
                    file_name="deepncv_batch_predictions.csv",
                    mime="text/csv",
                )

elif page == "Module 3: Geno-Compressor (BWT)":
    st.title("Module 3: Geno-Compressor + FM-index Search")
    st.subheader("BWT Compression + Inverse + Fast Pattern Search")

    sequence_input = st.text_input("Enter DNA sequence:", value="GATTACAGATTACA")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Compress Sequence (BWT)"):
            clean_seq = sanitize_dna_sequence(sequence_input)
            if clean_seq:
                bwt_result, sorted_rotations = generate_bwt(clean_seq)
                reconstructed = inverse_bwt(bwt_result)

                st.success("Transformation complete.")
                st.metric("Original Sequence", clean_seq)
                st.metric("BWT Output", bwt_result)
                st.metric("Reconstructed Sequence", reconstructed)

                with st.expander("Show sorted rotation matrix"):
                    st.code("\n".join(sorted_rotations))
            else:
                st.warning("Please enter a DNA sequence.")

    with col_b:
        pattern = st.text_input("Pattern for FM-index search:", value="TACA")
        if st.button("Run FM-index Search"):
            clean_seq = sanitize_dna_sequence(sequence_input)
            clean_pattern = sanitize_dna_sequence(pattern)

            if not clean_seq or not clean_pattern:
                st.warning("Please provide both sequence and pattern.")
            else:
                fm = build_fm_index(clean_seq)
                positions = [p for p in fm_backward_search(clean_pattern, fm) if p < len(clean_seq)]

                st.metric("Matches Found", len(positions))
                st.write("Match Start Positions (0-based):", positions)

                with st.expander("Show FM-index internals"):
                    st.write("BWT:", fm["bwt"])
                    st.write("Suffix Array:", fm["suffix_array"])
                    st.write("C-table:", fm["c_table"])

                    occ_df = pd.DataFrame({c: fm["occ"][c] for c in fm["alphabet"]})
                    st.dataframe(occ_df, use_container_width=True)
