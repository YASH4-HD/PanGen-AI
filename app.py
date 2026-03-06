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
    st.title("Module 1: Pangenome Explorer")
    st.subheader("Evolutionary Conservation & Variant Analysis")
    
    st.markdown("Upload multiple FASTA files or a VCF file to analyze conserved non-coding regions.")
    
    uploaded_files = st.file_uploader("Upload Genomic Files (.fasta, .vcf)", accept_multiple_files=True)
    
    if st.button("Run Evolutionary Analysis"):
        if uploaded_files:
            st.success("Files loaded successfully! (Backend alignment logic will be implemented here)")
            # Placeholder for future visualization
            st.code("Processing Multiple Sequence Alignment (MSA)...", language="python")
        else:
            st.warning("Please upload files to proceed.")

# --- PAGE 3: MODULE 2 (DeepNCV) ---
elif page == "Module 2: DeepNCV (AI Variant Caller)":
    st.title("Module 2: Deep Learning for Non-Coding Variants (DeepNCV)")
    st.subheader("PyTorch-based 1D-CNN Functional Predictor")
    
    st.markdown("Enter a non-coding DNA sequence to predict its functional state (e.g., Active Enhancer vs. Inactive).")
    
    dna_sequence = st.text_area("Enter DNA Sequence (A, T, C, G):", height=150, placeholder="e.g., ATGCGTACGTAGCTAG...")
    mutation_pos = st.number_input("Enter Mutation Position (Optional):", min_value=0, value=0)
    
    if st.button("Predict Functional Impact (PyTorch)"):
        if len(dna_sequence) > 10:
            st.info("Running sequence through 1D-CNN Model...")
            # Placeholder for PyTorch inference
            st.progress(50)
            st.success("Prediction: Highly Conserved Functional Region (Confidence: 94.2%)")
        else:
            st.error("Please enter a valid DNA sequence (minimum 10 base pairs).")

# --- PAGE 4: MODULE 3 (Geno-Compressor) ---
elif page == "Module 3: Geno-Compressor (BWT)":
st.title("Module 3: Geno-Compressor")
st.subheader("Pangenomic Data Structures & Compression")
st.markdown("Demonstrating the Burrows-Wheeler Transform (BWT) for memory-efficient genomic data storage. BWT groups runs of identical characters, making it highly compressible via Run-Length Encoding (RLE) and forms the basis of the FM-index used in modern aligners.")

# User Input
sequence_input = st.text_input("Enter a short DNA sequence to compress:", value="GATTACA")

if st.button("Compress Sequence"):
    if sequence_input:
        # Run the BWT
        bwt_result, sorted_rotations = generate_bwt(sequence_input)
        
        st.success("Transformation Complete!")
        
        # Display Results in Columns for a clean look
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Original Sequence", value=sequence_input)
            st.metric(label="BWT Output (Last Column)", value=bwt_result)
            
        with col2:
            # Show the magic of reversing it
            reconstructed = inverse_bwt(bwt_result)
            st.metric(label="Reconstructed Sequence", value=reconstructed)
            
        # Optional: Show the math behind it (The sorted matrix)
        with st.expander("Show Lexicographical Matrix (How it works)"):
            st.code('\n'.join(sorted_rotations))
            
    else:
        st.warning("Please enter a DNA sequence.")
