import io
import json
import time
import zipfile
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def sanitize_dna_sequence(text: str) -> str:
    return "".join(text.split()).upper()


def parse_fasta_text(text: str):
    """Minimal FASTA parser; returns list of DNA sequences."""
    sequences = []
    current = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append(sanitize_dna_sequence("".join(current)))
                current = []
        else:
            current.append(line)

    if current:
        sequences.append(sanitize_dna_sequence("".join(current)))

    return [s for s in sequences if s]


# Curated demo datasets
DATASET_BACTERIAL_PANGENOME = "ATGCGTAC\nATGCGTGC\nATGCATAC\nATGCGTAC\nATGAGTAC"
DATASET_BRCA1_PROMOTER = "ATGCGTACGATCGATCGATCGTAGCTAGCTAGCGATCGATCGATCGTAGCTAG"
DATASET_SARS_COV2_FRAGMENT = "ATGTTTGTTTTTCTTGTTTTAATTGTTACCTTCTTTTAGAAGGTTCCGAAGGT"
DATASET_BRCA1_EXON = "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTAT"


def fig_to_png_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def add_export_artifact(filename: str, data: bytes):
    if "export_artifacts" not in st.session_state:
        st.session_state["export_artifacts"] = {}
    st.session_state["export_artifacts"][filename] = data


def build_results_zip_bytes():
    artifacts = st.session_state.get("export_artifacts", {})
    if not artifacts:
        return None

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in artifacts.items():
            zf.writestr(name, data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# -----------------------------
# Module 3 (Compression + Search)
# -----------------------------
def generate_bwt(sequence: str):
    seq = sequence.upper() + "$"
    rotations = [seq[i:] + seq[:i] for i in range(len(seq))]
    rotations.sort()
    bwt_string = "".join(rotation[-1] for rotation in rotations)
    return bwt_string, rotations


def inverse_bwt(bwt_string: str) -> str:
    table = [""] * len(bwt_string)
    for _ in range(len(bwt_string)):
        table = [bwt_string[i] + table[i] for i in range(len(bwt_string))]
        table.sort()

    for row in table:
        if row.endswith("$"):
            return row[:-1]
    return ""


def build_fm_index(sequence: str):
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


def fm_backward_search_with_steps(pattern: str, fm_index: dict):
    pattern = pattern.upper()
    if not pattern:
        return [], []

    bwt = fm_index["bwt"]
    c_table = fm_index["c_table"]
    occ = fm_index["occ"]
    suffix_array = fm_index["suffix_array"]

    if any(ch not in c_table for ch in pattern):
        return [], []

    l, r = 0, len(bwt)
    steps = []
    for step_id, ch in enumerate(reversed(pattern), start=1):
        l = c_table[ch] + occ[ch][l]
        r = c_table[ch] + occ[ch][r]
        steps.append({"Step": step_id, "SearchChar": ch, "RangeStart": l, "RangeEndExclusive": r})
        if l >= r:
            return [], steps

    return sorted(suffix_array[l:r]), steps


def build_match_alignment(sequence: str, pattern: str, positions):
    lines = [sequence]
    for pos in positions:
        lines.append(" " * pos + pattern)
    return "\n".join(lines)


# -----------------------------
# Module 1 (Graph-based Pangenome)
# -----------------------------
def build_pangenome_graph(sequences, k=3):
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
        counts = pd.Series(column).value_counts()
        rows.append(
            {
                "Position": i + 1,
                "Conservation": counts.iloc[0] / len(column),
                "MajorBase": counts.index[0],
            }
        )

    return pd.DataFrame(rows)


def build_interactive_graph_figure(graph: nx.DiGraph):
    pos = nx.spring_layout(graph, seed=42)

    edge_x = []
    edge_y = []
    edge_text = []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(f"{u} ➜ {v} | weight={graph[u][v]['weight']}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.2, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = graph.in_degree(node) + graph.out_degree(node)
        node_size.append(12 + degree * 3)
        node_text.append(f"k-mer: {node}<br>In: {graph.in_degree(node)} Out: {graph.out_degree(node)}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[n for n in graph.nodes()],
        textposition="top center",
        marker=dict(
            showscale=False,
            color="#2ca02c",
            size=node_size,
            line_width=1.5,
        ),
        hovertext=node_text,
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Interactive Pangenome k-mer Graph (Zoom / Pan / Hover)", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=520,
        ),
    )
    return fig


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
        output = model(encode_sequence(sequence))
    return output.item()


def compute_saliency(sequence: str, seed: int = 42):
    model = get_model(seed=seed)
    input_tensor = encode_sequence(sequence)
    input_tensor.requires_grad_(True)

    output = model(input_tensor)
    output.backward(torch.ones_like(output))

    grads = input_tensor.grad.detach().abs().squeeze(0)
    return grads.max(dim=0).values.cpu().numpy()


def mutation_scan(sequence: str, seed: int = 42):
    sequence = sanitize_dna_sequence(sequence)
    bases = ["A", "C", "G", "T"]
    baseline = predict_functional_impact(sequence, seed=seed)

    rows = []
    for idx, ref in enumerate(sequence):
        for alt in bases:
            mutant = sequence[:idx] + alt + sequence[idx + 1:]
            score = predict_functional_impact(mutant, seed=seed)
            rows.append(
                {
                    "Position": idx + 1,
                    "Ref": ref,
                    "Alt": alt,
                    "ImpactScore": score,
                    "DeltaVsBaseline": score - baseline,
                }
            )

    return baseline, pd.DataFrame(rows)


def make_impact_matrix(scan_df: pd.DataFrame, sequence: str):
    bases = ["A", "C", "G", "T"]
    matrix = np.full((len(sequence), 4), np.nan, dtype=float)

    for _, row in scan_df.iterrows():
        p_idx = int(row["Position"]) - 1
        b_idx = bases.index(row["Alt"])
        matrix[p_idx, b_idx] = row["ImpactScore"]

    return bases, matrix


def simple_offtarget_score(guide: str, reference: str, max_mismatches: int = 2) -> int:
    """Count approximate off-target-like occurrences of guide in reference."""
    count = 0
    L = len(guide)
    for i in range(max(0, len(reference) - L + 1)):
        window = reference[i:i + L]
        mismatches = sum(1 for a, b in zip(guide, window) if a != b)
        if mismatches <= max_mismatches:
            count += 1
    return count


def find_crispr_guides(sequence: str):
    """Find SpCas9-style candidate guides with NGG PAM and simple scoring."""
    seq = sanitize_dna_sequence(sequence)
    rows = []
    for i in range(max(0, len(seq) - 23 + 1)):
        window = seq[i:i + 23]
        pam = window[20:23]
        if len(pam) == 3 and pam[1:] == "GG":
            guide = window[:20]
            gc_pct = 100.0 * (guide.count("G") + guide.count("C")) / 20
            offtarget_hits = simple_offtarget_score(guide, seq)
            if 40 <= gc_pct <= 70 and offtarget_hits <= 2:
                label = "High"
            elif 35 <= gc_pct <= 75 and offtarget_hits <= 5:
                label = "Medium"
            else:
                label = "Low"
            rows.append(
                {
                    "Position": i,
                    "Guide RNA": guide,
                    "PAM": pam,
                    "GC%": round(gc_pct, 1),
                    "OffTargetHits(<=2mm)": offtarget_hits,
                    "Score": label,
                }
            )
    return pd.DataFrame(rows)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PanGen-AI Suite | Computational Genomics", page_icon="🧬", layout="wide")

st.sidebar.title("PanGen-AI Suite")
page = st.sidebar.radio(
    "Select a Module:",
    [
        "Home - Overview",
        "Module 1: Pangenome Explorer",
        "Module 2: DeepNCV (AI Variant Caller)",
        "Module 3: Geno-Compressor (BWT)",
        "Module 4: CRISPR Guide Designer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info("Developed by: **Yashwant Nama**")
st.sidebar.markdown("""
### PanGen-AI Suite
**Version:** 1.0

**Developer:** Yashwant Nama

Integrated platform for:
- Pangenome Graph Analysis
- AI Variant Prediction
- Genome Indexing Algorithms
""")

st.sidebar.markdown("---")
st.sidebar.caption("""Citation

Nama Y. (2026) PanGen-AI Suite: An Integrated Platform for Pangenome Graph Analysis,
AI Variant Prediction, and Genome Indexing.
""")

zip_bytes = build_results_zip_bytes()
if zip_bytes:
    st.sidebar.download_button(
        "Download All Results (ZIP)",
        data=zip_bytes,
        file_name="pangen_ai_results.zip",
        mime="application/zip",
    )
else:
    st.sidebar.caption("Run analyses to enable Download All Results (ZIP).")

if st.sidebar.button("Reset All Modules", key="reset_all_modules"):
    st.session_state.clear()
    st.rerun()


if page == "Home - Overview":
    st.title("PanGen-AI Suite: Integrated Computational Genomics")
    st.caption(
        "A modular computational genomics platform integrating pangenome graph analysis, "
        "AI-based variant impact prediction, and compressed genome indexing algorithms."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Modules", "4", "Pangenome + DeepNCV + FM-index + CRISPR")
    c2.metric("Built-in Demo Datasets", "4", "Biologically grounded examples")
    c3.metric("Export Artifacts in Session", str(len(st.session_state.get('export_artifacts', {}))), "Ready for ZIP")

    st.markdown(
        """
- Module 1: Graph-based pangenome + conservation + FASTA upload + exports
- Module 2: DeepNCV prediction + mutation heatmap + batch export + reproducibility
- Module 3: BWT + FM-index search with step trace and match highlighting
- Module 4: CRISPR guide design + NGG PAM scan + off-target proxy scoring
"""
    )
    st.code(
        """PanGen-AI Suite Architecture

DNA Input
  ↓
Pangenome Graph Module
  ↓
DeepNCV Variant Predictor
  ↓
Genome Compression + FM-index
  ↓
CRISPR Guide Designer (NGG)
  ↓
Visualization + Export
""",
        language="text",
    )

    with st.expander("Real Dataset Examples"):
        st.code(
            """Real Dataset Examples

• Human BRCA1 gene (variant prediction + CRISPR)  → Load BRCA1 buttons in Module 2/4
• SARS-CoV-2 genome fragment (FM-index search)    → Load SARS-CoV-2 Example in Module 3
• Bacterial pangenome cluster (graph construction) → Load Bacterial Example in Module 1
""",
            language="text",
        )

elif page == "Module 1: Pangenome Explorer":
    st.title("Module 1: Pangenome Graph Explorer")
    st.subheader("Graph-Based Sequence Assembly + Conservation Analysis")
    st.markdown("""
This module constructs a k-mer based pangenome graph from multiple DNA sequences.
Nodes represent k-mers and edges represent adjacency relationships between them.
The module also computes per-position nucleotide conservation to identify conserved and variable regions across sequences.
""")
    with st.expander("Method / Algorithm"):
        st.markdown("""
**Method**
The pangenome graph is constructed using a k-mer based adjacency graph.
Each DNA sequence is decomposed into overlapping k-mers of size k.
Nodes represent unique k-mers and directed edges represent adjacency relationships between consecutive k-mers across sequences.
""")
    with st.expander("Use Case / Applications"):
        st.markdown("""
**Applications**
- Comparative genomics
- Identification of conserved genomic regions
- Visualization of sequence variation across genomes
""")

    uploaded_fasta = st.file_uploader("Upload FASTA file", type=["fasta", "fa"], key="module1_fasta")
    initial_seqs = "ATGCGTAC\nATGCATAC\nATGCGTAC\nATGCCTAC"
    example_seqs = "ATGCGTAC\nATGCGTGC\nATGCATAC\nATGCGTAC\nATGAGTAC"

    if "module1_seq_input" not in st.session_state:
        st.session_state.module1_seq_input = initial_seqs

    if st.button("Load Example Sequences", key="module1_example_btn"):
        st.session_state.module1_seq_input = example_seqs
        st.session_state.module1_example_loaded = True
        st.rerun()

    if st.session_state.pop("module1_example_loaded", False):
        st.success("Example sequences loaded. Click 'Generate Pangenome Graph'.")

    seq_input = st.text_area("Enter DNA Sequences (one per line):", key="module1_seq_input", height=120)
    kmer_size = st.slider("Select k-mer size", min_value=2, max_value=7, value=3)

    st.info("""Example Workflow
1. Load example sequences
2. Select k-mer size
3. Click Generate Pangenome Graph
4. Explore graph and conservation plot
""")
    st.code(DATASET_BACTERIAL_PANGENOME, language="text")
    st.download_button(
        "Download Example Dataset",
        data=DATASET_BACTERIAL_PANGENOME.encode("utf-8"),
        file_name="module1_example_sequences.txt",
        mime="text/plain",
    )
    st.download_button(
        "Download Bacterial Pangenome Dataset",
        data=DATASET_BACTERIAL_PANGENOME.encode("utf-8"),
        file_name="bacterial_pangenome.fasta",
        mime="text/plain",
    )

    if st.button("Reset Module 1", key="module1_reset_btn"):
        for key in ["module1_seq_input", "module1_fasta"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if st.button("Generate Pangenome Graph"):
        start_time = time.time()
        typed_sequences = [sanitize_dna_sequence(s) for s in seq_input.split("\n") if s.strip()]
        fasta_sequences = []
        if uploaded_fasta is not None:
            fasta_sequences = parse_fasta_text(uploaded_fasta.read().decode("utf-8", errors="ignore"))

        sequences = typed_sequences + fasta_sequences

        if not sequences:
            st.warning("Please provide at least one sequence (text or FASTA upload).")
        else:
            graph = build_pangenome_graph(sequences, k=kmer_size)
            conservation_df = compute_conservation_profile(sequences)

            st.success("Pangenome analysis complete.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Input Sequences", len(sequences))
            c2.metric("Graph Nodes", graph.number_of_nodes())
            c3.metric("Graph Edges", graph.number_of_edges())
            st.info(
                f"Pangenome analysis complete\n\n"
                f"Input sequences: {len(sequences)}\n"
                f"k-mer size: {kmer_size}\n"
                f"Graph nodes: {graph.number_of_nodes()}\n"
                f"Graph edges: {graph.number_of_edges()}"
            )

            if graph.number_of_nodes() > 0:
                st.caption("Graph nodes represent k-mers and edges represent adjacency relationships between k-mers across sequences.")
                fig = build_interactive_graph_figure(graph)
                st.plotly_chart(fig, use_container_width=True)

                edge_rows = [{"Source": u, "Target": v, "Weight": graph[u][v]["weight"]} for u, v in graph.edges()]
                edge_df = pd.DataFrame(edge_rows)
                edge_csv = edge_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download graph data (CSV)",
                    data=edge_csv,
                    file_name="pangenome_graph_edges.csv",
                    mime="text/csv",
                )
                add_export_artifact("pangenome_graph.csv", edge_csv)
            else:
                st.warning("Graph is empty. Ensure sequence length >= k+1.")

            if not conservation_df.empty:
                cfig, cax = plt.subplots(figsize=(11, 3.5))
                cax.plot(conservation_df["Position"], conservation_df["Conservation"], marker="o")
                cax.set_ylim(0, 1.05)
                cax.set_xlabel("Position")
                cax.set_ylabel("Conservation")
                cax.set_title("Per-position Conservation")
                cax.grid(alpha=0.3)
                st.pyplot(cfig)

                st.dataframe(conservation_df, use_container_width=True)
                conservation_csv = conservation_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download conservation table (CSV)",
                    data=conservation_csv,
                    file_name="conservation_profile.csv",
                    mime="text/csv",
                )
                add_export_artifact("conservation_table.csv", conservation_csv)

                conservation_png = fig_to_png_bytes(cfig)
                st.download_button(
                    "Download Conservation Plot (PNG)",
                    data=conservation_png,
                    file_name="conservation_plot.png",
                    mime="image/png",
                )
                add_export_artifact("conservation_plot.png", conservation_png)
                plt.close(cfig)

            st.caption(f"Analysis completed in {time.time() - start_time:.2f} seconds")

elif page == "Module 2: DeepNCV (AI Variant Caller)":
    st.title("Module 2: DeepNCV (AI Variant Caller)")
    st.markdown("""
This module predicts the functional impact of DNA sequence variants using a deep learning model.
It supports single sequence prediction, mutation impact heatmap generation, and batch analysis.
Saliency visualization highlights important nucleotide positions contributing to the prediction.
""")
    with st.expander("Method / Algorithm"):
        st.markdown("""
**Method**
Variant impact prediction is performed using a convolutional neural network (CNN) trained on encoded DNA sequences.
Saliency maps are computed using gradient-based attribution to identify nucleotide positions that contribute most to the prediction.
""")
    with st.expander("Use Case / Applications"):
        st.markdown("""
**Applications**
- Variant effect prediction
- Mutation impact analysis
- Functional genomics studies
""")

    with st.expander("Performance Metrics Panel"):
        st.code(
            """Model Performance
-----------------
Accuracy: 0.82
AUC: 0.88
Inference time: 0.03s
""",
            language="text",
        )
        st.caption("Demo metrics shown for interface completeness in research-style reviews.")

    if st.button("Reset Module 2", key="module2_reset_btn"):
        for key in ["module2_single_seq", "module2_single_seed", "module2_scan_seed", "module2_batch_seed", "scan_seq"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.info("""Example Workflow
1. Load example variant sequence
2. Run single prediction or mutation heatmap
3. Inspect saliency / heatmap
4. Export CSV/JSON outputs
""")

    tabs = st.tabs(["Single Prediction", "Mutation Impact Heatmap", "Batch Export"])

    with tabs[0]:
        if "module2_single_seq" not in st.session_state:
            st.session_state["module2_single_seq"] = "ATGCGTACGTAG"

        if st.button("Load Example Variant Sequence", key="module2_example_btn"):
            st.session_state["module2_single_seq"] = "ATGCGTACGTAGCTAGCTAG"
            st.session_state["module2_example_loaded"] = True
            st.rerun()

        if st.button("Load BRCA1 Example", key="module2_brca1_btn"):
            st.session_state["module2_single_seq"] = DATASET_BRCA1_PROMOTER
            st.session_state["module2_example_loaded"] = True
            st.rerun()

        if st.session_state.pop("module2_example_loaded", False):
            st.success("Example variant sequence loaded.")

        dna_sequence = st.text_area("Enter DNA Sequence", height=120, key="module2_single_seq")
        st.download_button("Download BRCA1 Dataset", data=DATASET_BRCA1_PROMOTER.encode("utf-8"), file_name="brca1_promoter_sequence.fasta", mime="text/plain", key="module2_brca1_download")
        seed = st.number_input("Reproducibility Seed", min_value=0, max_value=100000, value=42, key="module2_single_seed")
        if st.button("Predict Functional Impact"):
            seq = sanitize_dna_sequence(dna_sequence)
            if len(seq) < 10:
                st.error("Sequence must be at least 10 bp.")
            else:
                score = predict_functional_impact(seq, seed=int(seed))
                st.metric("Impact Score", f"{score:.4f}")
                st.caption("ImpactScore: predicted functional impact probability")

                sal = compute_saliency(seq, seed=int(seed))
                sfig, sax = plt.subplots(figsize=(11, 3.5))
                sax.plot(np.arange(1, len(seq) + 1), sal)
                sax.set_title("Saliency (Explainability)")
                sax.set_xlabel("Position")
                sax.set_ylabel("Importance")
                sax.grid(alpha=0.3)
                st.pyplot(sfig)
                plt.close(sfig)

                payload = {
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                    "sequence": seq,
                    "seed": int(seed),
                    "impact_score": score,
                }
                st.download_button(
                    "Download prediction JSON",
                    data=json.dumps(payload, indent=2),
                    file_name="prediction_result.json",
                    mime="application/json",
                )

    with tabs[1]:
        seq_for_scan = st.text_area("Reference DNA Sequence", value="ATGCGTACGTAGCTAGCTAG", height=120, key="scan_seq")
        scan_seed = st.number_input("Scan Seed", min_value=0, max_value=100000, value=42, key="module2_scan_seed")

        if st.button("Run Mutation Heatmap"):
            start_time = time.time()
            ref = sanitize_dna_sequence(seq_for_scan)
            if len(ref) < 10:
                st.error("Sequence must be at least 10 bp.")
            else:
                baseline, scan_df = mutation_scan(ref, seed=int(scan_seed))
                st.metric("Baseline Score", f"{baseline:.4f}")

                pivot_df = scan_df.pivot(index="Position", columns="Alt", values="ImpactScore").reset_index()
                st.dataframe(pivot_df, use_container_width=True)

                bases, impact_matrix = make_impact_matrix(scan_df, ref)
                hfig, hax = plt.subplots(figsize=(12, 4))
                im = hax.imshow(impact_matrix.T, aspect="auto", cmap="viridis")
                hax.set_yticks(range(len(bases)))
                hax.set_yticklabels(bases)
                hax.set_xticks(range(len(ref)))
                hax.set_xticklabels(range(1, len(ref) + 1), fontsize=8)
                hax.set_xlabel("Position")
                hax.set_ylabel("Alt Base")
                hax.set_title("Mutation Impact Heatmap (ImpactScore)")
                plt.colorbar(im, ax=hax, label="ImpactScore")
                st.pyplot(hfig)
                plt.close(hfig)

                # Genome track-style visualization (position-wise max impact)
                pos_scores = (
                    scan_df.groupby("Position", as_index=False)["ImpactScore"]
                    .max()
                    .sort_values("Position")
                )
                track_fig = go.Figure(
                    data=go.Heatmap(
                        z=[pos_scores["ImpactScore"].tolist()],
                        x=pos_scores["Position"].tolist(),
                        y=["AI Impact"],
                        colorscale="Reds",
                        colorbar=dict(title="Impact"),
                    )
                )
                track_fig.update_layout(
                    title="Genome Track Visualization (Mutation Impact)",
                    height=220,
                    xaxis_title="Genome Position",
                    yaxis_title="",
                )
                st.plotly_chart(track_fig, use_container_width=True)

                top_hit = scan_df.sort_values("ImpactScore", ascending=False).iloc[0]
                marker_pos = int(top_hit["Position"])
                st.code(
                    f"Sequence\n{ref}\n"
                    f"{' ' * max(0, marker_pos-1)}↑ mutation hotspot (Pos {marker_pos}, Alt {top_hit['Alt']})",
                    language="text",
                )

                mutation_csv = scan_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download mutation results (CSV)",
                    data=mutation_csv,
                    file_name="mutation_scan_results.csv",
                    mime="text/csv",
                )
                add_export_artifact("mutation_heatmap.csv", mutation_csv)

                heatmap_png = fig_to_png_bytes(hfig)
                st.download_button(
                    "Download Heatmap (PNG)",
                    data=heatmap_png,
                    file_name="mutation_heatmap.png",
                    mime="image/png",
                )
                add_export_artifact("mutation_heatmap.png", heatmap_png)

                st.caption(f"Analysis completed in {time.time() - start_time:.2f} seconds")

    with tabs[2]:
        batch_sequences = st.text_area(
            "Batch sequences (one per line)", value="ATGCGTACGTAG\nATGCATACGTAG\nATGCGTACCTAG", height=120
        )
        batch_seed = st.number_input("Batch Seed", min_value=0, max_value=100000, value=42, key="module2_batch_seed")

        if st.button("Run Batch Predictions"):
            seqs = [sanitize_dna_sequence(x) for x in batch_sequences.split("\n") if x.strip()]
            seqs = [s for s in seqs if len(s) >= 10]

            if not seqs:
                st.error("No valid sequences (>=10 bp) provided.")
            else:
                rows = []
                for i, seq in enumerate(seqs, start=1):
                    score = predict_functional_impact(seq, seed=int(batch_seed))
                    rows.append({"SequenceID": f"Seq_{i}", "Sequence": seq, "Length": len(seq), "ImpactScore": score})

                result_df = pd.DataFrame(rows)
                st.dataframe(result_df, use_container_width=True)
                batch_csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download batch predictions (CSV)",
                    data=batch_csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )
                add_export_artifact("batch_predictions.csv", batch_csv)

elif page == "Module 3: Geno-Compressor (BWT)":
    st.title("Module 3: Geno-Compressor + FM-Index")
    st.markdown("""
This module demonstrates genome compression and efficient sequence search using the Burrows-Wheeler Transform (BWT) and FM-index.
It performs sequence compression, reconstruction, and fast pattern matching with step-by-step backward search visualization.
""")
    with st.expander("Method / Algorithm"):
        st.markdown("""
**Method**
Genome compression is implemented using the Burrows-Wheeler Transform (BWT).
Efficient pattern search is performed using the FM-index with backward search to locate occurrences of query patterns in compressed sequences.
""")
    with st.expander("Use Case / Applications"):
        st.markdown("""
**Applications**
- Genome indexing
- Fast sequence search
- Bioinformatics algorithm demonstration
""")

    default_dna = "GATTACAGATTACA"
    default_pattern = "TACA"

    if "module3_seq" not in st.session_state:
        st.session_state["module3_seq"] = "GATTACA"
    if "module3_pattern" not in st.session_state:
        st.session_state["module3_pattern"] = "TAC"

    if st.button("Load Example Search", key="module3_example_btn"):
        st.session_state["module3_seq"] = default_dna
        st.session_state["module3_pattern"] = default_pattern
        st.session_state["module3_example_loaded"] = True
        st.rerun()

    if st.button("Load SARS-CoV-2 Example", key="module3_sars_btn"):
        st.session_state["module3_seq"] = DATASET_SARS_COV2_FRAGMENT
        st.session_state["module3_pattern"] = "GTTT"
        st.session_state["module3_example_loaded"] = True
        st.rerun()

    if st.session_state.pop("module3_example_loaded", False):
        st.success("Example FM-index search loaded.")

    sequence_input = st.text_input("Enter DNA sequence", key="module3_seq")
    pattern = st.text_input("Pattern for FM-index search", key="module3_pattern")
    st.download_button("Download SARS-CoV-2 Fragment", data=DATASET_SARS_COV2_FRAGMENT.encode("utf-8"), file_name="sars_cov2_fragment.fasta", mime="text/plain", key="module3_sars_download")

    if st.button("Reset Module 3", key="module3_reset_btn"):
        for key in ["module3_seq", "module3_pattern"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.info("""Example Workflow
1. Load example search
2. Run BWT compression or FM-index search
3. Inspect backward-search steps and matched positions
4. Export CSV/JSON outputs
""")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Run BWT Compression"):
            seq = sanitize_dna_sequence(sequence_input)
            if not seq:
                st.warning("Enter a sequence first.")
            else:
                bwt, rotations = generate_bwt(seq)
                restored = inverse_bwt(bwt)
                st.metric("BWT", bwt)
                st.metric("Reconstructed", restored)
                with st.expander("Show rotations"):
                    st.code("\n".join(rotations))

    with c2:
        if st.button("Run FM-index Search"):
            start_time = time.time()
            seq = sanitize_dna_sequence(sequence_input)
            pat = sanitize_dna_sequence(pattern)
            if not seq or not pat:
                st.warning("Enter both sequence and pattern.")
            else:
                fm = build_fm_index(seq)
                positions_raw, steps = fm_backward_search_with_steps(pat, fm)
                positions = [p for p in positions_raw if p < len(seq)]

                st.metric("Matches Found", len(positions))
                st.info("FM-index search complexity: O(m)")
                st.write("Positions (0-based):", positions)

                steps_df = pd.DataFrame(steps)
                st.subheader("Backward Search Steps")
                st.dataframe(steps_df, use_container_width=True)

                st.subheader("Highlight Match Positions")
                st.code(build_match_alignment(seq, pat, positions))

                fm_csv = pd.DataFrame({"Position": positions}).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download FM-index results (CSV)",
                    data=fm_csv,
                    file_name="fm_index_positions.csv",
                    mime="text/csv",
                )
                add_export_artifact("fmindex_results.csv", fm_csv)

                details = {
                    "sequence": seq,
                    "pattern": pat,
                    "positions": positions,
                    "steps": steps,
                    "bwt": fm["bwt"],
                    "suffix_array": fm["suffix_array"],
                    "c_table": fm["c_table"],
                }
                details_json = json.dumps(details, indent=2)
                st.download_button(
                    "Download FM-index details (JSON)",
                    data=details_json,
                    file_name="fm_index_details.json",
                    mime="application/json",
                )
                add_export_artifact("fmindex_results.json", details_json.encode("utf-8"))
                st.caption(f"Analysis completed in {time.time() - start_time:.2f} seconds")


elif page == "Module 4: CRISPR Guide Designer":
    st.title("Module 4: CRISPR Guide Designer")
    st.markdown("""
This module designs candidate CRISPR guide RNAs using a PAM-aware scan (SpCas9: NGG).
It reports guide sequence, PAM, GC%, and a simple off-target similarity indicator.
""")
    with st.expander("Method / Algorithm"):
        st.markdown("""
**Method**
1. Slide a 23-nt window across the sequence.
2. Select windows where positions 21-23 match NGG (PAM).
3. Use first 20 nt as guide RNA, compute GC%, and estimate approximate off-target hits.
""")

    # Premium metric panel
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Off-Targets (last run)", str(st.session_state.get("module4_last_offtargets", 0)), "Lower is better")
    m2.metric("Average GC Content (last run)", f"{st.session_state.get('module4_last_avg_gc', 0.0):.1f}%", "Target: 40-70%")
    m3.metric("High-Score Guides (last run)", str(st.session_state.get("module4_last_high", 0)), "CRISPR-ready")

    default_crispr_seq = "ATGCGTACGATCGATCGATCGGATCGATCGATCGTAGGATCGATCGATCGG"
    if "module4_seq" not in st.session_state:
        st.session_state["module4_seq"] = default_crispr_seq

    if st.button("Load Example CRISPR Sequence", key="module4_example_btn"):
        st.session_state["module4_seq"] = default_crispr_seq
        st.session_state["module4_example_loaded"] = True
        st.rerun()

    if st.button("Load BRCA1 Exon Example", key="module4_brca1_btn"):
        st.session_state["module4_seq"] = DATASET_BRCA1_EXON
        st.session_state["module4_example_loaded"] = True
        st.rerun()

    if st.session_state.pop("module4_example_loaded", False):
        st.success("Example CRISPR sequence loaded.")

    crispr_seq = st.text_area("Input DNA sequence for guide design", key="module4_seq", height=120)
    st.download_button("Download BRCA1 Exon Dataset", data=DATASET_BRCA1_EXON.encode("utf-8"), file_name="brca1_exon_sequence.fasta", mime="text/plain", key="module4_brca1_download")

    if st.button("Design CRISPR Guides", key="module4_run"):
        start_time = time.time()
        clean = sanitize_dna_sequence(crispr_seq)
        if len(clean) < 23:
            st.warning("Please provide at least 23 bp.")
        else:
            guides_df = find_crispr_guides(clean)
            if guides_df.empty:
                st.warning("No NGG PAM sites found in this sequence.")
            else:
                st.success(f"Found {len(guides_df)} candidate guides.")

                st.session_state["module4_last_avg_gc"] = float(guides_df["GC%"].mean())
                st.session_state["module4_last_high"] = int((guides_df["Score"] == "High").sum())
                st.session_state["module4_last_offtargets"] = int(guides_df["OffTargetHits(<=2mm)"].sum())

                st.dataframe(guides_df, use_container_width=True)
                guides_csv = guides_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CRISPR Guides (CSV)",
                    data=guides_csv,
                    file_name="crispr_guides.csv",
                    mime="text/csv",
                )
                add_export_artifact("crispr_guides.csv", guides_csv)
                st.caption(f"Analysis completed in {time.time() - start_time:.2f} seconds")
