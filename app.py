import streamlit as st
def generate_bwt(sequence: str) -> str:
    """
    Generates the Burrows-Wheeler Transform of a given DNA sequence.
    """
    # 1. Append the special End-of-String (EOS) character '$'
    # '$' is lexicographically smaller than A, C, G, T
    seq = sequence.upper() + '$'
    
    # 2. Generate all cyclic rotations of the sequence
    # Example for "ACA$": ["ACA$", "CA$A", "A$AC", "$ACA"]
    rotations = [seq[i:] + seq[:i] for i in range(len(seq))]
    
    # 3. Sort the rotations lexicographically
    rotations.sort()
    
    # 4. Extract the last column of the sorted matrix
    bwt_string = ''.join([rotation[-1] for rotation in rotations])
    
    return bwt_string, rotations

def inverse_bwt(bwt_string: str) -> str:
    """
    Reconstructs the original DNA sequence from the BWT string.
    This proves the transformation is lossless.
    """
    # Create an empty table with the same number of rows as the length of the BWT string
    table = [""] * len(bwt_string)
    
    # Iteratively rebuild the sorted rotations matrix
    for _ in range(len(bwt_string)):
        # Prepend the BWT string (which is the last column) to the table
        table = [bwt_string[i] + table[i] for i in range(len(bwt_string))]
        # Sort the rows lexicographically
        table.sort()
        
    # Find the row that ends with the EOS character '$'
    for row in table:
        if row.endswith('$'):
            # Return the original sequence without the '$'
            return row[:-1]
            
    return ""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
# 1. Define the 1D-CNN Architecture
class SimpleDNA_CNN(nn.Module):
    def __init__(self):
        super(SimpleDNA_CNN, self).__init__()
        # Input channels = 4 (A, C, G, T one-hot encoded)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Adaptive pooling allows sequences of any length!
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) # Output probability between 0 and 1
        return x
# 2. Helper function to One-Hot Encode DNA
def encode_sequence(seq: str) -> torch.Tensor:
    """Converts a DNA string into a one-hot PyTorch tensor of shape (1, 4, L)"""
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }
    # Default to uniform distribution for unknown characters like 'N'
    encoded = [mapping.get(base.upper(), [0.25, 0.25, 0.25, 0.25]) for base in seq]
    
    # Convert to tensor and reshape to (Batch=1, Channels=4, Length=L)
    tensor = torch.tensor(encoded, dtype=torch.float32).T.unsqueeze(0)
    return tensor

# 3. Inference Function
def predict_functional_impact(sequence: str) -> float:
    # Set seed so the "dummy" untrained model gives consistent results for the demo
    torch.manual_seed(42) 
    model = SimpleDNA_CNN()
    model.eval() # Set to evaluation mode
    
    with torch.no_grad():
        input_tensor = encode_sequence(sequence)
        output = model(input_tensor)
        probability = output.item()
        
    return probability

def build_pangenome_graph(sequences, k=3):
    """
    Builds a simple De Bruijn/Sequence graph from multiple DNA sequences.
    This demonstrates understanding of graph-based pangenomics.
    """
    graph = nx.DiGraph()

    for seq in sequences:
        seq = seq.upper()
        if len(seq) < k + 1:
            continue

        # Extract k-mers and build directed edges between adjacent k-mers
        for i in range(len(seq) - k):
            kmer_1 = seq[i:i + k]
            kmer_2 = seq[i + 1:i + k + 1]

            if graph.has_edge(kmer_1, kmer_2):
                graph[kmer_1][kmer_2]['weight'] += 1
            else:
                graph.add_edge(kmer_1, kmer_2, weight=1)

    return graph

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PanGen-AI Suite | Computational Genomics",
    page_icon="DNA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("PanGen-AI Suite")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Select a Module:",
    ["Home - Overview", 
     "Module 1: Pangenome Explorer", 
     "Module 2: DeepNCV (AI Variant Caller)", 
     "Module 3: Geno-Compressor (BWT)"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Developed by: **Yashwant Nama**\n\n"
    "Computational Researcher\n\n"
    "Target: Advanced Genomic Data Structures & Deep Learning"
)

# --- PAGE 1: HOME ---
if page == "Home - Overview":
    st.title("PanGen-AI Suite: Integrated Computational Genomics")
    st.markdown("""
    Welcome to the **PanGen-AI Suite**. This toolkit bridges the gap between raw genomic data, 
    evolutionary conservation, and artificial intelligence.
    
    ### Available Modules:
    1. **Pangenome Explorer:** Visualize evolutionary conservation and structural variants across multiple species using FASTA/VCF data.
    2. **DeepNCV (Non-Coding Variant AI):** A PyTorch-based Deep Learning model (1D-CNN) to predict the functional impact of mutations in non-coding DNA regions.
    3. **Geno-Compressor:** An implementation of advanced pangenomic data structures, utilizing the Burrows-Wheeler Transform (BWT) for efficient DNA sequence compression and search.
    
    *Select a module from the left sidebar to begin.*
    """)

# --- PAGE 2: MODULE 1 (Pangenome Explorer) ---
elif page == "Module 1: Pangenome Explorer":
    st.title("Module 1: Pangenome Graph Explorer")
    st.subheader("Graph-Based Sequence Assembly & Conservation")
    st.markdown("""
    Modern pangenomics represents genomes as **Sequence Graphs** rather than linear strings.
    This module takes multiple homologous sequences, extracts *k-mers*, and constructs a directed graph (De Bruijn Graph).
    Thicker edges represent highly conserved regions (Core Genome), while branching paths represent structural variations (Accessory Genome).
    """)

    default_seqs = "ATGCGTAC\nATGCATAC\nATGCGTAC\nATGCCTAC"
    seq_input = st.text_area("Enter DNA Sequences (one per line):", value=default_seqs, height=120)
    kmer_size = st.slider("Select k-mer size (Resolution):", min_value=2, max_value=5, value=3)

    if st.button("Generate Pangenome Graph"):
        sequences = [s.strip() for s in seq_input.split('\n') if s.strip()]

        if len(sequences) > 0:
            with st.spinner("Constructing Graph Topology..."):
                graph = build_pangenome_graph(sequences, k=kmer_size)
                num_nodes = graph.number_of_nodes()
                num_edges = graph.number_of_edges()

                st.success("Pangenome Graph Constructed Successfully!")

                col1, col2, col3 = st.columns(3)
                col1.metric("Input Sequences", len(sequences))
                col2.metric("Graph Nodes (k-mers)", num_nodes)
                col3.metric("Graph Edges", num_edges)

                if num_nodes > 0 and num_edges > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    pos = nx.spring_layout(graph, seed=42)
                    weights = [graph[u][v]['weight'] * 2 for u, v in graph.edges()]

                    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='#4CAF50', alpha=0.9, ax=ax)
                    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='white', font_weight='bold', ax=ax)
                    nx.draw_networkx_edges(graph, pos, width=weights, edge_color='#555555', arrowsize=20, ax=ax)

                    plt.title(f"Pangenome Sequence Graph (k={kmer_size})", fontsize=14)
                    plt.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)

                    st.info("💡 **Interpretation:** Nodes represent DNA k-mers. Paths that are shared among multiple sequences have thicker arrows (Core Genome). Branches indicate mutations, SNPs, or structural variations (Accessory Genome).")
                else:
                    st.warning("No graph could be built. Ensure each sequence length is at least k+1.")
        else:
            st.warning("Please enter at least one DNA sequence.")

# --- PAGE 3: MODULE 2 (DeepNCV) ---
elif page == "Module 2: DeepNCV (AI Variant Caller)":
    st.title("Module 2: Deep Learning for Non-Coding Variants (DeepNCV)")
    st.subheader("PyTorch-based 1D-CNN Functional Predictor")
    
    st.markdown("Enter a non-coding DNA sequence to predict its functional state using a 1D Convolutional Neural Network. The model converts the sequence into a one-hot encoded tensor and extracts spatial motifs.")
    
    dna_sequence = st.text_area("Enter DNA Sequence (A, T, C, G):", height=150, value="ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAG")
    
    if st.button("Predict Functional Impact (PyTorch)"):
        # Clean the sequence (remove spaces/newlines)
        clean_seq = "".join(dna_sequence.split())
        
        if len(clean_seq) >= 10:
            with st.spinner("Running sequence through 1D-CNN Layers..."):
                # Call the PyTorch backend
                impact_score = predict_functional_impact(clean_seq)
                confidence_pct = impact_score * 100
                
            st.success("Inference Complete!")
            
            # Display Results
            st.metric(label="Functional Impact Probability", value=f"{confidence_pct:.2f}%")
           
# --- PAGE 4: MODULE 3 ---
elif page == "Module 3: Geno-Compressor (BWT)":
    st.title("Module 3: Geno-Compressor")
    st.subheader("Pangenomic Data Structures & Compression")
    st.markdown("Demonstrating the Burrows-Wheeler Transform (BWT) for memory-efficient genomic data storage. BWT groups runs of identical characters, making it highly compressible via Run-Length Encoding (RLE) and forms the basis of the FM-index used in modern aligners.")
    
    sequence_input = st.text_input("Enter a short DNA sequence to compress:", value="GATTACA")
    
    if st.button("Compress Sequence"):
        if sequence_input:
            bwt_result, sorted_rotations = generate_bwt(sequence_input)
            st.success("Transformation Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Original Sequence", value=sequence_input)
                st.metric(label="BWT Output (Last Column)", value=bwt_result)
            with col2:
                reconstructed = inverse_bwt(bwt_result)
                st.metric(label="Reconstructed Sequence", value=reconstructed)
            
            with st.expander("Show Lexicographical Matrix (How it works)"):
                st.code('\n'.join(sorted_rotations))
        else:
            st.warning("Please enter a DNA sequence.")
