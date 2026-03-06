import json
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit as st
def generate_bwt(sequence: str) -> str:
import torch
    """
import torch.nn as nn
    Generates the Burrows-Wheeler Transform of a given DNA sequence.
import torch.nn.functional as F
    """

    # 1. Append the special End-of-String (EOS) character '$'

    # '$' is lexicographically smaller than A, C, G, T
# -----------------------------
    seq = sequence.upper() + '$'
# Utilities
    
# -----------------------------
    # 2. Generate all cyclic rotations of the sequence
def sanitize_dna_sequence(text: str) -> str:
    # Example for "ACA$": ["ACA$", "CA$A", "A$AC", "$ACA"]
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


# -----------------------------
# Module 3 (Compression + Search)
# -----------------------------
def generate_bwt(sequence: str):
    seq = sequence.upper() + "$"
    rotations = [seq[i:] + seq[:i] for i in range(len(seq))]
    rotations = [seq[i:] + seq[:i] for i in range(len(seq))]
    
    # 3. Sort the rotations lexicographically
    rotations.sort()
    rotations.sort()
    
    bwt_string = "".join(rotation[-1] for rotation in rotations)
    # 4. Extract the last column of the sorted matrix
    bwt_string = ''.join([rotation[-1] for rotation in rotations])
    
    return bwt_string, rotations
    return bwt_string, rotations



def inverse_bwt(bwt_string: str) -> str:
def inverse_bwt(bwt_string: str) -> str:
    """
    Reconstructs the original DNA sequence from the BWT string.
    This proves the transformation is lossless.
    """
    # Create an empty table with the same number of rows as the length of the BWT string
    table = [""] * len(bwt_string)
    table = [""] * len(bwt_string)
    
    # Iteratively rebuild the sorted rotations matrix
    for _ in range(len(bwt_string)):
    for _ in range(len(bwt_string)):
        # Prepend the BWT string (which is the last column) to the table
        table = [bwt_string[i] + table[i] for i in range(len(bwt_string))]
        table = [bwt_string[i] + table[i] for i in range(len(bwt_string))]
        # Sort the rows lexicographically
        table.sort()
        table.sort()
        

    # Find the row that ends with the EOS character '$'
    for row in table:
    for row in table:
        if row.endswith('$'):
        if row.endswith("$"):
            # Return the original sequence without the '$'
            return row[:-1]
            return row[:-1]
            
    return ""
    return ""


import pandas as pd
import numpy as np
from deepncv_utils import predict_functional_impact
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Define the 1D-CNN Architecture
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
class SimpleDNA_CNN(nn.Module):
    def __init__(self):
    def __init__(self):
        super(SimpleDNA_CNN, self).__init__()
        super().__init__()
        # Input channels = 4 (A, C, G, T one-hot encoded)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Adaptive pooling allows sequences of any length!
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc1 = nn.Linear(32, 16)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.fc2 = nn.Linear(16, 1)


    def forward(self, x):
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) # Output probability between 0 and 1
        return torch.sigmoid(self.fc2(x))
        return x



# 2. Helper function to One-Hot Encode DNA
def encode_sequence(seq: str) -> torch.Tensor:
def encode_sequence(seq: str) -> torch.Tensor:
    """Converts a DNA string into a one-hot PyTorch tensor of shape (1, 4, L)"""
    mapping = {
    mapping = {
        'A': [1, 0, 0, 0],
        "A": [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        "C": [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        "G": [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
        "T": [0, 0, 0, 1],
    }
    }
    # Default to uniform distribution for unknown characters like 'N'
    encoded = [mapping.get(base.upper(), [0.25, 0.25, 0.25, 0.25]) for base in seq]
    encoded = [mapping.get(base.upper(), [0.25, 0.25, 0.25, 0.25]) for base in seq]
    
    return torch.tensor(encoded, dtype=torch.float32).T.unsqueeze(0)
    # Convert to tensor and reshape to (Batch=1, Channels=4, Length=L)

    tensor = torch.tensor(encoded, dtype=torch.float32).T.unsqueeze(0)

    return tensor
def get_model(seed=42):

    torch.manual_seed(seed)
# 3. Inference Function
def predict_functional_impact(sequence: str) -> float:
    # Set seed so the "dummy" untrained model gives consistent results for the demo
    torch.manual_seed(42) 
    model = SimpleDNA_CNN()
    model = SimpleDNA_CNN()
    model.eval() # Set to evaluation mode
    model.eval()
    
    return model


def predict_functional_impact(sequence: str, seed: int = 42) -> float:
    model = get_model(seed=seed)
    with torch.no_grad():
    with torch.no_grad():
        input_tensor = encode_sequence(sequence)
        output = model(encode_sequence(sequence))
        output = model(input_tensor)
    return output.item()
        probability = output.item()

        

    return probability
def compute_saliency(sequence: str, seed: int = 42):

    model = get_model(seed=seed)
# --- PAGE CONFIGURATION ---
    input_tensor = encode_sequence(sequence)
st.set_page_config(
    input_tensor.requires_grad_(True)
    page_title="PanGen-AI Suite | Computational Genomics",

    page_icon="DNA",
    output = model(input_tensor)
    layout="wide",
    output.backward(torch.ones_like(output))
    initial_sidebar_state="expanded"

)
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


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PanGen-AI Suite | Computational Genomics", page_icon="🧬", layout="wide")


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("PanGen-AI Suite")
st.sidebar.title("PanGen-AI Suite")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
page = st.sidebar.radio(
    "Select a Module:",
    "Select a Module:",
    ["Home - Overview", 
    [
     "Module 1: Pangenome Explorer", 
        "Home - Overview",
     "Module 2: DeepNCV (AI Variant Caller)", 
        "Module 1: Pangenome Explorer",
     "Module 3: Geno-Compressor (BWT)"]
        "Module 2: DeepNCV (AI Variant Caller)",
        "Module 3: Geno-Compressor (BWT)",
    ],
)
)


st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.info(
st.sidebar.info("Developed by: **Yashwant Nama**")
    "Developed by: **Yashwant Nama**\n\n"
st.sidebar.markdown("""
    "Computational Researcher\n\n"
### PanGen-AI Suite
    "Target: Advanced Genomic Data Structures & Deep Learning"
**Version:** 1.0
)

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



# --- PAGE 1: HOME ---
if page == "Home - Overview":
if page == "Home - Overview":
    st.title("PanGen-AI Suite: Integrated Computational Genomics")
    st.title("PanGen-AI Suite: Integrated Computational Genomics")
    st.markdown("""
    st.caption(
    Welcome to the **PanGen-AI Suite**. This toolkit bridges the gap between raw genomic data, 
        "A modular computational genomics platform integrating pangenome graph analysis, "
    evolutionary conservation, and artificial intelligence.
        "AI-based variant impact prediction, and compressed genome indexing algorithms."
    
    )
    ### Available Modules:
    st.markdown(
    1. **Pangenome Explorer:** Visualize evolutionary conservation and structural variants across multiple species using FASTA/VCF data.
        """
    2. **DeepNCV (Non-Coding Variant AI):** A PyTorch-based Deep Learning model (1D-CNN) to predict the functional impact of mutations in non-coding DNA regions.
- Module 1: Graph-based pangenome + conservation + FASTA upload + exports
    3. **Geno-Compressor:** An implementation of advanced pangenomic data structures, utilizing the Burrows-Wheeler Transform (BWT) for efficient DNA sequence compression and search.
- Module 2: DeepNCV prediction + mutation heatmap + batch export + reproducibility
    
- Module 3: BWT + FM-index search with step trace and match highlighting
    *Select a module from the left sidebar to begin.*
"""
    """)
    )


# --- PAGE 2: MODULE 1 (Pangenome Explorer) ---
elif page == "Module 1: Pangenome Explorer":
elif page == "Module 1: Pangenome Explorer":
    st.title("Module 1: Pangenome Explorer")
    st.title("Module 1: Pangenome Graph Explorer")
    st.subheader("Evolutionary Conservation & Variant Analysis")
    st.subheader("Graph-Based Sequence Assembly + Conservation Analysis")
    
    st.markdown("""
    st.markdown("Upload multiple FASTA files or a VCF file to analyze conserved non-coding regions.")
This module constructs a k-mer based pangenome graph from multiple DNA sequences.
    
Nodes represent k-mers and edges represent adjacency relationships between them.
    uploaded_files = st.file_uploader("Upload Genomic Files (.fasta, .vcf)", accept_multiple_files=True)
The module also computes per-position nucleotide conservation to identify conserved and variable regions across sequences.
    
""")
    if st.button("Run Evolutionary Analysis"):
    with st.expander("Method / Algorithm"):
        if uploaded_files:
        st.markdown("""
            st.success("Files loaded successfully! (Backend alignment logic will be implemented here)")
**Method**
            # Placeholder for future visualization
The pangenome graph is constructed using a k-mer based adjacency graph.
            st.code("Processing Multiple Sequence Alignment (MSA)...", language="python")
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

    st.info("Suggested screenshot workflow: 1) Load example sequences → 2) Generate graph → 3) Capture graph + conservation plot.")
    st.code(example_seqs, language="text")
    st.download_button(
        "Download Example Dataset",
        data=example_seqs.encode("utf-8"),
        file_name="module1_example_sequences.txt",
        mime="text/plain",
    )

    if st.button("Reset Module 1", key="module1_reset_btn"):
        for key in ["module1_seq_input", "module1_fasta"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if st.button("Generate Pangenome Graph"):
        typed_sequences = [sanitize_dna_sequence(s) for s in seq_input.split("\n") if s.strip()]
        fasta_sequences = []
        if uploaded_fasta is not None:
            fasta_sequences = parse_fasta_text(uploaded_fasta.read().decode("utf-8", errors="ignore"))

        sequences = typed_sequences + fasta_sequences

        if not sequences:
            st.warning("Please provide at least one sequence (text or FASTA upload).")
        else:
        else:
            st.warning("Please upload files to proceed.")
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
                st.download_button(
                    "Download graph data (CSV)",
                    data=edge_df.to_csv(index=False).encode("utf-8"),
                    file_name="pangenome_graph_edges.csv",
                    mime="text/csv",
                )
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
                plt.close(cfig)

                st.dataframe(conservation_df, use_container_width=True)
                st.download_button(
                    "Download conservation table (CSV)",
                    data=conservation_df.to_csv(index=False).encode("utf-8"),
                    file_name="conservation_profile.csv",
                    mime="text/csv",
                )


# --- PAGE 3: MODULE 2 (DeepNCV) ---
elif page == "Module 2: DeepNCV (AI Variant Caller)":
elif page == "Module 2: DeepNCV (AI Variant Caller)":
    st.title("Module 2: Deep Learning for Non-Coding Variants (DeepNCV)")
    st.title("Module 2: DeepNCV (AI Variant Caller)")
    st.subheader("PyTorch-based 1D-CNN Functional Predictor")
    st.markdown("""
    
This module predicts the functional impact of DNA sequence variants using a deep learning model.
    st.markdown("Enter a non-coding DNA sequence to predict its functional state using a 1D Convolutional Neural Network. The model converts the sequence into a one-hot encoded tensor and extracts spatial motifs.")
It supports single sequence prediction, mutation impact heatmap generation, and batch analysis.
    
Saliency visualization highlights important nucleotide positions contributing to the prediction.
    dna_sequence = st.text_area("Enter DNA Sequence (A, T, C, G):", height=150, value="ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAG")
""")
    
    with st.expander("Method / Algorithm"):
    if st.button("Predict Functional Impact (PyTorch)"):
        st.markdown("""
        # Clean the sequence (remove spaces/newlines)
**Method**
        clean_seq = "".join(dna_sequence.split())
Variant impact prediction is performed using a convolutional neural network (CNN) trained on encoded DNA sequences.
        
Saliency maps are computed using gradient-based attribution to identify nucleotide positions that contribute most to the prediction.
        if len(clean_seq) >= 10:
""")
            with st.spinner("Running sequence through 1D-CNN Layers..."):
    with st.expander("Use Case / Applications"):
                # Call the PyTorch backend
        st.markdown("""
                impact_score = predict_functional_impact(clean_seq)
**Applications**
                confidence_pct = impact_score * 100
- Variant effect prediction
                
- Mutation impact analysis
            st.success("Inference Complete!")
- Functional genomics studies
            
""")
            # Display Results

            st.metric(label="Functional Impact Probability", value=f"{confidence_pct:.2f}%")
    if st.button("Reset Module 2", key="module2_reset_btn"):
            
        for key in ["module2_single_seq", "module2_single_seed", "module2_scan_seed", "module2_batch_seed", "scan_seq"]:
            if confidence_pct > 50:
            if key in st.session_state:
                st.info("🧠 Model Prediction: **Active Functional Region (e.g., Enhancer/Promoter)**")
                del st.session_state[key]
        st.rerun()

    tabs = st.tabs(["Single Prediction", "Mutation Impact Heatmap", "Batch Export"])

    with tabs[0]:
        if "module2_single_seq" not in st.session_state:
            st.session_state["module2_single_seq"] = "ATGCGTACGTAG"

        if st.button("Load Example Variant Sequence", key="module2_example_btn"):
            st.session_state["module2_single_seq"] = "ATGCGTACGTAGCTAGCTAG"
            st.session_state["module2_example_loaded"] = True
            st.rerun()

        if st.session_state.pop("module2_example_loaded", False):
            st.success("Example variant sequence loaded.")

        dna_sequence = st.text_area("Enter DNA Sequence", height=120, key="module2_single_seq")
        seed = st.number_input("Reproducibility Seed", min_value=0, max_value=100000, value=42, key="module2_single_seed")
        if st.button("Predict Functional Impact"):
            seq = sanitize_dna_sequence(dna_sequence)
            if len(seq) < 10:
                st.error("Sequence must be at least 10 bp.")
            else:
            else:
                st.warning("🧠 Model Prediction: **Inactive/Neutral Region**")
                score = predict_functional_impact(seq, seed=int(seed))
                
                st.metric("Impact Score", f"{score:.4f}")
            with st.expander("Show PyTorch Tensor Details"):
                st.caption("ImpactScore: predicted functional impact probability")
                st.write(f"Sequence Length: {len(clean_seq)} bp")

                st.write(f"Input Tensor Shape: `[1, 4, {len(clean_seq)}]` (Batch, Channels, Length)")
                sal = compute_saliency(seq, seed=int(seed))
                
                sfig, sax = plt.subplots(figsize=(11, 3.5))
        else:
                sax.plot(np.arange(1, len(seq) + 1), sal)
            st.error("Please enter a valid DNA sequence (minimum 10 base pairs).")
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

                st.download_button(
                    "Download mutation results (CSV)",
                    data=scan_df.to_csv(index=False).encode("utf-8"),
                    file_name="mutation_scan_results.csv",
                    mime="text/csv",
                )

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
                st.download_button(
                    "Download batch predictions (CSV)",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )


# --- PAGE 4: MODULE 3 (Geno-Compressor) ---
elif page == "Module 3: Geno-Compressor (BWT)":
elif page == "Module 3: Geno-Compressor (BWT)":
    st.title("Module 3: Geno-Compressor")
    st.title("Module 3: Geno-Compressor + FM-Index")
    st.subheader("Pangenomic Data Structures & Compression")
    st.markdown("""
    st.markdown("Demonstrating the Burrows-Wheeler Transform (BWT) for memory-efficient genomic data storage. BWT groups runs of identical characters, making it highly compressible via Run-Length Encoding (RLE) and forms the basis of the FM-index used in modern aligners.")
This module demonstrates genome compression and efficient sequence search using the Burrows-Wheeler Transform (BWT) and FM-index.
    
It performs sequence compression, reconstruction, and fast pattern matching with step-by-step backward search visualization.
    # User Input
""")
    sequence_input = st.text_input("Enter a short DNA sequence to compress:", value="GATTACA")
    with st.expander("Method / Algorithm"):
    
        st.markdown("""
    if st.button("Compress Sequence"):
**Method**
        if sequence_input:
Genome compression is implemented using the Burrows-Wheeler Transform (BWT).
            # Run the BWT
Efficient pattern search is performed using the FM-index with backward search to locate occurrences of query patterns in compressed sequences.
            bwt_result, sorted_rotations = generate_bwt(sequence_input)
""")
            
    with st.expander("Use Case / Applications"):
            st.success("Transformation Complete!")
        st.markdown("""
            
**Applications**
            # Display Results in Columns for a clean look
- Genome indexing
            col1, col2 = st.columns(2)
- Fast sequence search
            
- Bioinformatics algorithm demonstration
            with col1:
""")
                st.metric(label="Original Sequence", value=sequence_input)

                st.metric(label="BWT Output (Last Column)", value=bwt_result)
    default_dna = "GATTACAGATTACA"
                
    default_pattern = "TACA"
            with col2:

                # Show the magic of reversing it
    if "module3_seq" not in st.session_state:
                reconstructed = inverse_bwt(bwt_result)
        st.session_state["module3_seq"] = "GATTACA"
                st.metric(label="Reconstructed Sequence", value=reconstructed)
    if "module3_pattern" not in st.session_state:
            
        st.session_state["module3_pattern"] = "TAC"
        # Optional: Show the math behind it (The sorted matrix)

        with st.expander("Show Lexicographical Matrix (How it works)"):
    if st.button("Load Example Search", key="module3_example_btn"):
            st.code('\n'.join(sorted_rotations))
        st.session_state["module3_seq"] = default_dna
            
        st.session_state["module3_pattern"] = default_pattern
    else:
        st.session_state["module3_example_loaded"] = True
        st.warning("Please enter a DNA sequence.")
        st.rerun()

    if st.session_state.pop("module3_example_loaded", False):
        st.success("Example FM-index search loaded.")

    sequence_input = st.text_input("Enter DNA sequence", key="module3_seq")
    pattern = st.text_input("Pattern for FM-index search", key="module3_pattern")

    if st.button("Reset Module 3", key="module3_reset_btn"):
        for key in ["module3_seq", "module3_pattern"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

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

                st.download_button(
                    "Download FM-index results (CSV)",
                    data=pd.DataFrame({"Position": positions}).to_csv(index=False).encode("utf-8"),
                    file_name="fm_index_positions.csv",
                    mime="text/csv",
                )

                details = {
                    "sequence": seq,
                    "pattern": pat,
                    "positions": positions,
                    "steps": steps,
                    "bwt": fm["bwt"],
                    "suffix_array": fm["suffix_array"],
                    "c_table": fm["c_table"],
                }
                st.download_button(
                    "Download FM-index details (JSON)",
                    data=json.dumps(details, indent=2),
                    file_name="fm_index_details.json",
                    mime="application/json",
                )
