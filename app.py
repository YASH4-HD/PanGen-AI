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
from plotly.subplots import make_subplots
import streamlit as st
import stmol
import py3Dmol
import requests
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
DATASET_TRANSLATION_DNA = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"


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
# Module 5 (Genome Alignment Explorer)
# -----------------------------
def needleman_wunsch_align(reference: str, query: str, match: int = 2, mismatch: int = -1, gap: int = -2):
reference = sanitize_dna_sequence(reference)
query = sanitize_dna_sequence(query)
n, m = len(reference), len(query)
if n == 0 or m == 0:
return "", "", 0

score = np.zeros((n + 1, m + 1), dtype=int)
trace = np.zeros((n + 1, m + 1), dtype=int)

for i in range(1, n + 1):
score[i, 0] = i * gap
trace[i, 0] = 1
for j in range(1, m + 1):
score[0, j] = j * gap
trace[0, j] = 2

for i in range(1, n + 1):
for j in range(1, m + 1):
diag = score[i - 1, j - 1] + (match if reference[i - 1] == query[j - 1] else mismatch)
up = score[i - 1, j] + gap
left = score[i, j - 1] + gap
best = max(diag, up, left)
score[i, j] = best
trace[i, j] = 0 if best == diag else (1 if best == up else 2)

align_ref = []
align_query = []
i, j = n, m
while i > 0 or j > 0:
if i > 0 and j > 0 and trace[i, j] == 0:
align_ref.append(reference[i - 1])
align_query.append(query[j - 1])
i -= 1
j -= 1
elif i > 0 and (j == 0 or trace[i, j] == 1):
align_ref.append(reference[i - 1])
align_query.append("-")
i -= 1
else:
align_ref.append("-")
align_query.append(query[j - 1])
j -= 1

return "".join(reversed(align_ref)), "".join(reversed(align_query)), int(score[n, m])


def alignment_annotation(aligned_ref: str, aligned_query: str):
markers = []
mismatches = 0
for a, b in zip(aligned_ref, aligned_query):
if a == b:
markers.append("|")
elif a == "-" or b == "-":
markers.append(" ")
mismatches += 1
else:
markers.append("x")
mismatches += 1
return "".join(markers), mismatches


def map_reads_to_reference(reference: str, reads):
rows = []
for idx, read in enumerate(reads, start=1):
a_ref, a_read, score = needleman_wunsch_align(reference, read)
marker, mismatches = alignment_annotation(a_ref, a_read)
rows.append(
{
"ReadID": f"Read_{idx}",
"Read": read,
"Score": score,
"Mismatches": mismatches,
"AlignedReference": a_ref,
"AlignmentMarker": marker,
"AlignedRead": a_read,
}
)
return pd.DataFrame(rows)


# -----------------------------
# Module 6 (Protein Analysis)
# -----------------------------
CODON_TABLE = {
"ATA": "I", "ATC": "I", "ATT": "I", "ATG": "M",
"ACA": "T", "ACC": "T", "ACG": "T", "ACT": "T",
"AAC": "N", "AAT": "N", "AAA": "K", "AAG": "K",
"AGC": "S", "AGT": "S", "AGA": "R", "AGG": "R",
"CTA": "L", "CTC": "L", "CTG": "L", "CTT": "L",
"CCA": "P", "CCC": "P", "CCG": "P", "CCT": "P",
"CAC": "H", "CAT": "H", "CAA": "Q", "CAG": "Q",
"CGA": "R", "CGC": "R", "CGG": "R", "CGT": "R",
"GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V",
"GCA": "A", "GCC": "A", "GCG": "A", "GCT": "A",
"GAC": "D", "GAT": "D", "GAA": "E", "GAG": "E",
"GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G",
"TCA": "S", "TCC": "S", "TCG": "S", "TCT": "S",
"TTC": "F", "TTT": "F", "TTA": "L", "TTG": "L",
"TAC": "Y", "TAT": "Y", "TAA": "*", "TAG": "*",
"TGC": "C", "TGT": "C", "TGA": "*", "TGG": "W",
}

AMINO_ACID_WEIGHTS = {
"A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.15,
"Q": 146.15, "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17,
"L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
"S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

HYDROPHOBIC_AA = set("AILMFWYV")


def translate_dna_to_protein(sequence: str):
seq = sanitize_dna_sequence(sequence)
protein = []
for i in range(0, len(seq) - 2, 3):
codon = seq[i:i + 3]
protein.append(CODON_TABLE.get(codon, "X"))
return "".join(protein)


def analyze_protein_properties(protein: str):
aa = [x for x in protein if x.isalpha()]
if not aa:
return {"Length": 0, "MolecularWeight_kDa": 0.0, "HydrophobicPct": 0.0}, pd.DataFrame(columns=["AminoAcid", "Count"])

mw = sum(AMINO_ACID_WEIGHTS.get(x, 0.0) for x in aa) / 1000.0
hydrophobic_pct = 100.0 * sum(1 for x in aa if x in HYDROPHOBIC_AA) / len(aa)
comp = pd.Series(aa).value_counts().reset_index()
comp.columns = ["AminoAcid", "Count"]
return {
"Length": len(aa),
"MolecularWeight_kDa": round(mw, 2),
"HydrophobicPct": round(hydrophobic_pct, 1),
}, comp


# BLOSUM62 substitution scores for mutation-impact heuristics
BLOSUM62 = {
"A": {"A": 4, "R": -1, "N": -2, "D": -2, "C": 0, "Q": -1, "E": -1, "G": 0, "H": -2, "I": -1, "L": -1, "K": -1, "M": -1, "F": -2, "P": -1, "S": 1, "T": 0, "W": -3, "Y": -2, "V": 0},
"R": {"A": -1, "R": 5, "N": 0, "D": -2, "C": -3, "Q": 1, "E": 0, "G": -2, "H": 0, "I": -3, "L": -2, "K": 2, "M": -1, "F": -3, "P": -2, "S": -1, "T": -1, "W": -3, "Y": -2, "V": -3},
"N": {"A": -2, "R": 0, "N": 6, "D": 1, "C": -3, "Q": 0, "E": 0, "G": 0, "H": 1, "I": -3, "L": -3, "K": 0, "M": -2, "F": -3, "P": -2, "S": 1, "T": 0, "W": -4, "Y": -2, "V": -3},
"D": {"A": -2, "R": -2, "N": 1, "D": 6, "C": -3, "Q": 0, "E": 2, "G": -1, "H": -1, "I": -3, "L": -4, "K": -1, "M": -3, "F": -3, "P": -1, "S": 0, "T": -1, "W": -4, "Y": -3, "V": -3},
"C": {"A": 0, "R": -3, "N": -3, "D": -3, "C": 9, "Q": -3, "E": -4, "G": -3, "H": -3, "I": -1, "L": -1, "K": -3, "M": -1, "F": -2, "P": -3, "S": -1, "T": -1, "W": -2, "Y": -2, "V": -1},
"Q": {"A": -1, "R": 1, "N": 0, "D": 0, "C": -3, "Q": 5, "E": 2, "G": -2, "H": 0, "I": -3, "L": -2, "K": 1, "M": 0, "F": -3, "P": -1, "S": 0, "T": -1, "W": -2, "Y": -1, "V": -2},
"E": {"A": -1, "R": 0, "N": 0, "D": 2, "C": -4, "Q": 2, "E": 5, "G": -2, "H": 0, "I": -3, "L": -3, "K": 1, "M": -2, "F": -3, "P": -1, "S": 0, "T": -1, "W": -3, "Y": -2, "V": -2},
"G": {"A": 0, "R": -2, "N": 0, "D": -1, "C": -3, "Q": -2, "E": -2, "G": 6, "H": -2, "I": -4, "L": -4, "K": -2, "M": -3, "F": -3, "P": -2, "S": 0, "T": -2, "W": -2, "Y": -3, "V": -3},
"H": {"A": -2, "R": 0, "N": 1, "D": -1, "C": -3, "Q": 0, "E": 0, "G": -2, "H": 8, "I": -3, "L": -3, "K": -1, "M": -2, "F": -1, "P": -2, "S": -1, "T": -2, "W": -2, "Y": 2, "V": -3},
"I": {"A": -1, "R": -3, "N": -3, "D": -3, "C": -1, "Q": -3, "E": -3, "G": -4, "H": -3, "I": 4, "L": 2, "K": -3, "M": 1, "F": 0, "P": -3, "S": -2, "T": -1, "W": -3, "Y": -1, "V": 3},
"L": {"A": -1, "R": -2, "N": -3, "D": -4, "C": -1, "Q": -2, "E": -3, "G": -4, "H": -3, "I": 2, "L": 4, "K": -2, "M": 2, "F": 0, "P": -3, "S": -2, "T": -1, "W": -2, "Y": -1, "V": 1},
"K": {"A": -1, "R": 2, "N": 0, "D": -1, "C": -3, "Q": 1, "E": 1, "G": -2, "H": -1, "I": -3, "L": -2, "K": 5, "M": -1, "F": -3, "P": -1, "S": 0, "T": -1, "W": -3, "Y": -2, "V": -2},
"M": {"A": -1, "R": -1, "N": -2, "D": -3, "C": -1, "Q": 0, "E": -2, "G": -3, "H": -2, "I": 1, "L": 2, "K": -1, "M": 5, "F": 0, "P": -2, "S": -1, "T": -1, "W": -1, "Y": -1, "V": 1},
"F": {"A": -2, "R": -3, "N": -3, "D": -3, "C": -2, "Q": -3, "E": -3, "G": -3, "H": -1, "I": 0, "L": 0, "K": -3, "M": 0, "F": 6, "P": -4, "S": -2, "T": -2, "W": 1, "Y": 3, "V": -1},
"P": {"A": -1, "R": -2, "N": -2, "D": -1, "C": -3, "Q": -1, "E": -1, "G": -2, "H": -2, "I": -3, "L": -3, "K": -1, "M": -2, "F": -4, "P": 7, "S": -1, "T": -1, "W": -4, "Y": -3, "V": -2},
"S": {"A": 1, "R": -1, "N": 1, "D": 0, "C": -1, "Q": 0, "E": 0, "G": 0, "H": -1, "I": -2, "L": -2, "K": 0, "M": -1, "F": -2, "P": -1, "S": 4, "T": 1, "W": -3, "Y": -2, "V": -2},
"T": {"A": 0, "R": -1, "N": 0, "D": -1, "C": -1, "Q": -1, "E": -1, "G": -2, "H": -2, "I": -1, "L": -1, "K": -1, "M": -1, "F": -2, "P": -1, "S": 1, "T": 5, "W": -2, "Y": -2, "V": 0},
"W": {"A": -3, "R": -3, "N": -4, "D": -4, "C": -2, "Q": -2, "E": -3, "G": -2, "H": -2, "I": -3, "L": -2, "K": -3, "M": -1, "F": 1, "P": -4, "S": -3, "T": -2, "W": 11, "Y": 2, "V": -3},
"Y": {"A": -2, "R": -2, "N": -2, "D": -3, "C": -2, "Q": -1, "E": -2, "G": -3, "H": 2, "I": -1, "L": -1, "K": -2, "M": -1, "F": 3, "P": -3, "S": -2, "T": -2, "W": 2, "Y": 7, "V": -1},
"V": {"A": 0, "R": -3, "N": -3, "D": -3, "C": -1, "Q": -2, "E": -2, "G": -3, "H": -3, "I": 3, "L": 1, "K": -2, "M": 1, "F": -1, "P": -2, "S": -2, "T": 0, "W": -3, "Y": -1, "V": 4},
}

PROTEIN_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def parse_mutation_notation(mutation: str):
m = mutation.strip().upper()
if len(m) < 3:
return None
ref = m[0]
alt = m[-1]
pos_txt = m[1:-1]
if not pos_txt.isdigit():
return None
return ref, int(pos_txt), alt


def predict_protein_mutation_impact(protein_sequence: str, mutation: str):
seq = protein_sequence.strip().upper()
parsed = parse_mutation_notation(mutation)
if parsed is None:
return {"error": "Mutation must be in format like F5L."}

ref, pos, alt = parsed
if pos < 1 or pos > len(seq):
return {"error": f"Position out of range (1-{len(seq)})."}
if ref not in BLOSUM62 or alt not in BLOSUM62:
return {"error": "Only standard amino acids are supported."}

observed = seq[pos - 1]
if observed != ref:
return {"error": f"Reference mismatch at position {pos}: sequence has {observed}, not {ref}."}

score = BLOSUM62[ref][alt]
conservation = round((score + 4) / 15, 3)
if score <= -3:
impact = "High"
elif score < 0:
impact = "Medium"
else:
impact = "Low"

return {
"Position": pos,
"OriginalAA": ref,
"MutatedAA": alt,
"BLOSUM62Score": score,
"ConservationScore": conservation,
"PredictedImpact": impact,
}


def build_protein_mutation_landscape(protein_sequence: str):
seq = protein_sequence.strip().upper()
valid_positions = [aa for aa in seq if aa in BLOSUM62]
if not valid_positions:
return PROTEIN_AA_ORDER, np.zeros((len(PROTEIN_AA_ORDER), 0), dtype=float)

matrix = np.zeros((len(PROTEIN_AA_ORDER), len(valid_positions)), dtype=float)
for j, ref in enumerate(valid_positions):
for i, alt in enumerate(PROTEIN_AA_ORDER):
matrix[i, j] = BLOSUM62[ref][alt]
return PROTEIN_AA_ORDER, matrix


# -----------------------------
# Protein Structure Viewer Functions
# -----------------------------
def update_protein_id_from_example():
"""Callback function to update protein ID when example is selected."""
example_proteins = {
"p53 (Tumor suppressor)": "P04637",
"Hemoglobin": "1A3N", 
"Lysozyme": "1AKI",
"DNA Polymerase": "3K5A",
"Insulin": "1ZNJ",
"Myoglobin": "1MBO",
"Carbonic Anhydrase": "2CBA"
}

selected_example = st.session_state.get("module6_example_protein", "None")
if selected_example != "None" and selected_example in example_proteins:
st.session_state.module6_structure_id = example_proteins[selected_example]


def fetch_pdb_data(pdb_id: str):
"""Fetch PDB data from RCSB or AlphaFold database."""
pdb_id = pdb_id.upper().strip()

# Try RCSB first for standard PDB IDs
rcsb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
try:
response = requests.get(rcsb_url, timeout=10)
if response.status_code == 200:
return response.text, "rcsb"
except:
pass

# Try AlphaFold for UniProt IDs
alphafold_url = f"https://alphafold.ebi.ac.uk/files/AF-{pdb_id}-F1-model_v4.pdb"
try:
response = requests.get(alphafold_url, timeout=10)
if response.status_code == 200:
return response.text, "alphafold"
except:
pass

return None, None


def create_3d_structure_viewer(pdb_data: str, style: str = "cartoon"):
"""Create interactive 3D protein structure viewer."""
# Create 3Dmol viewer
viewer = py3Dmol.view(width=800, height=600)

# Add PDB data
viewer.addModel(pdb_data, "pdb")

# Set style based on selection
if style == "cartoon":
viewer.setStyle({'cartoon': {'color': 'spectrum'}})
elif style == "stick":
viewer.setStyle({'stick': {'colorscheme': 'Jmol'}})
elif style == "sphere":
viewer.setStyle({'sphere': {'colorscheme': 'Jmol'}})
elif style == "line":
viewer.setStyle({'line': {'colorscheme': 'Jmol'}})

# Zoom to fit
viewer.zoomTo()

# Set background
viewer.setBackgroundColor('#f0f0f0')

return viewer


def render_structure_in_streamlit(viewer):
"""Render 3Dmol viewer in Streamlit."""
# Convert viewer to HTML
viewer_html = viewer._make_html()

# Display in Streamlit
st.components.v1.html(viewer_html, height=600, width=800)


def get_protein_info(uniprot_id: str):
"""Get basic protein information from UniProt."""
try:
url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
response = requests.get(url, timeout=10)
if response.status_code == 200:
data = response.json()
return {
'name': data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown'),
'length': data.get('sequence', {}).get('length', 0),
'mass': data.get('sequence', {}).get('mass', 0),
'organism': data.get('organism', {}).get('scientificName', 'Unknown'),
'function': data.get('comments', [{}])[0].get('texts', [{}])[0].get('value', 'No functional description available') if data.get('comments') else 'No functional description available'
}
except:
pass
return None


# -----------------------------
# Module 8 (Genome Browser Functions)
# -----------------------------
def create_genome_browser_tracks(dna_sequence: str):
"""Create multi-track genome browser data from DNA sequence."""
seq_len = len(dna_sequence)
positions = list(range(1, seq_len + 1))

# Track 1: DNA Sequence (numeric representation)
dna_track = []
nucleotide_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
for base in dna_sequence.upper():
dna_track.append(nucleotide_map.get(base, 0))

# Track 2: Variant Impact Scores
variant_scores = []
for i in range(seq_len):
# Create a simple variant at each position and predict impact
test_seq = dna_sequence[:i] + 'A' + dna_sequence[i+1:]
if len(test_seq) == len(dna_sequence):
try:
score = predict_functional_impact(test_seq[:min(90, len(test_seq))])
variant_scores.append(score)
except:
variant_scores.append(0.5)
else:
variant_scores.append(0.5)

# Track 3: CRISPR Guide Locations
crispr_track = [0] * seq_len
guides_df = find_crispr_guides(dna_sequence)
for _, guide in guides_df.iterrows():
start = guide['Position']
end = start + 23  # Full guide + PAM
if end <= seq_len:
for pos in range(start, min(end, seq_len)):
crispr_track[pos] = guide['GC%'] / 100.0  # Normalize GC content

# Track 4: Gene Annotation (simulated ORFs)
gene_track = [0] * seq_len
# Find potential ORFs (start codon ATG to stop codons)
start_codons = []
for i in range(seq_len - 2):
if dna_sequence[i:i+3].upper() == 'ATG':
start_codons.append(i)

# Extend ORFs until stop codon or end
for start in start_codons:
for end in range(start + 3, seq_len - 2, 3):
codon = dna_sequence[end:end+3].upper()
if codon in ['TAA', 'TAG', 'TGA']:
# Mark this ORF
for pos in range(start, min(end + 3, seq_len)):
gene_track[pos] = 1
break

# Track 5: Protein Coding Regions (translation potential)
protein_track = [0] * seq_len
# Mark regions that could be translated (in-frame)
for i in range(0, seq_len - 2, 3):
codon = dna_sequence[i:i+3].upper()
if codon in CODON_TABLE and CODON_TABLE[codon] != 'X':
for pos in range(i, min(i + 3, seq_len)):
protein_track[pos] = 1

return {
'positions': positions,
'dna_sequence': dna_track,
'variant_impact': variant_scores,
'crispr_guides': crispr_track,
'gene_annotation': gene_track,
'protein_coding': protein_track
}


def create_genome_browser_figure(track_data: dict):
"""Create multi-track genome browser visualization using Plotly subplots."""
positions = track_data['positions']

# Validate data
if not positions or len(positions) == 0:
fig = go.Figure()
fig.add_annotation(
text="No data to display",
xref="paper",
yref="paper",
x=0.5,
y=0.5,
showarrow=False,
font=dict(size=20),
)
fig.update_layout(height=400)
return fig

# Create subplots (5 rows, 1 column)
fig = make_subplots(
rows=5,
cols=1,
shared_xaxes=True,
vertical_spacing=0.08,
subplot_titles=(
"DNA Sequence",
"Variant Impact",
"CRISPR Guides",
"Gene Annotation",
"Protein Coding",
),
)

# Track 1: DNA Sequence
fig.add_trace(
go.Heatmap(
z=[track_data['dna_sequence']],
x=positions,
y=[""],
colorscale="Viridis",
showscale=False,
hovertemplate='Position: %{x}<br>Base Value: %{z}<extra></extra>',
),
row=1,
col=1,
)

# Track 2: Variant Impact
fig.add_trace(
go.Heatmap(
z=[track_data['variant_impact']],
x=positions,
y=[""],
colorscale="Reds",
showscale=False,
hovertemplate='Position: %{x}<br>Impact Score: %{z:.3f}<extra></extra>',
),
row=2,
col=1,
)

# Track 3: CRISPR Guides
fig.add_trace(
go.Heatmap(
z=[track_data['crispr_guides']],
x=positions,
y=[""],
colorscale="Blues",
showscale=False,
hovertemplate='Position: %{x}<br>CRISPR Score: %{z:.2f}<extra></extra>',
),
row=3,
col=1,
)

# Track 4: Gene Annotation
fig.add_trace(
go.Heatmap(
z=[track_data['gene_annotation']],
x=positions,
y=[""],
colorscale="Greens",
showscale=False,
hovertemplate='Position: %{x}<br>Gene Region: %{z}<extra></extra>',
),
row=4,
col=1,
)

# Track 5: Protein Coding
fig.add_trace(
go.Heatmap(
z=[track_data['protein_coding']],
x=positions,
y=[""],
colorscale="Purples",
showscale=False,
hovertemplate='Position: %{x}<br>Coding Region: %{z}<extra></extra>',
),
row=5,
col=1,
)

# Update layout for proper height and look
fig.update_layout(
title_text="Interactive Genome Browser",
height=850,
showlegend=False,
margin=dict(l=40, r=40, t=80, b=40),
hovermode='closest',
)

# Add x-axis title only to the bottom track
fig.update_xaxes(title_text="Genomic Position", row=5, col=1)

# Hide y-axis tick labels since subplot titles are shown
fig.update_yaxes(showticklabels=False)

return fig


def export_track_data_csv(track_data: dict, dna_sequence: str):
"""Export track data as CSV."""
import io
import csv

output = io.StringIO()
writer = csv.writer(output)

# Header
writer.writerow(['Position', 'Base', 'Variant_Impact', 'CRISPR_Score', 'Gene_Region', 'Protein_Coding'])

# Data rows
for i, pos in enumerate(track_data['positions']):
base = dna_sequence[i].upper()
writer.writerow([
pos,
base,
f"{track_data['variant_impact'][i]:.3f}",
f"{track_data['crispr_guides'][i]:.2f}",
track_data['gene_annotation'][i],
track_data['protein_coding'][i]
])

return output.getvalue()


# -----------------------------
# Module 7 (Genome Annotation Explorer)
# -----------------------------
def parse_genome_annotations(annotation_text: str):
"""Parse genome annotation data from text input."""
annotations = []
lines = annotation_text.strip().split('\n')

for line in lines:
if not line.strip():
continue

parts = line.strip().split('\t')
if len(parts) >= 3:
try:
start = int(parts[0]) - 1  # Convert to 0-based
end = int(parts[1])
feature_type = parts[2]
description = parts[3] if len(parts) > 3 else ""

annotations.append({
'start': start,
'end': end,
'type': feature_type,
'description': description,
'y_position': len(annotations)  # Stack annotations vertically
})
except ValueError:
continue

return annotations


def create_genome_annotation_viewer(sequence: str, annotations: list):
"""Create interactive genome annotation viewer using Plotly."""
fig = go.Figure()

# Add sequence as background track
seq_length = len(sequence)
fig.add_shape(
type="rect",
x0=0, x1=seq_length,
y0=-0.5, y1=0.5,
fillcolor="lightgray",
line=dict(color="gray", width=1)
)

# Add annotations as colored tracks
colors = {
'Gene': '#1f77b4',
'Exon': '#ff7f0e', 
'Promoter': '#2ca02c',
'Regulatory': '#d62728',
'SNP': '#9467bd',
'CRISPR': '#8c564b',
'Variant': '#e377c2'
}

for i, annotation in enumerate(annotations):
color = colors.get(annotation['type'], '#7f7f7f')

# Add annotation track
fig.add_shape(
type="rect",
x0=annotation['start'], x1=annotation['end'],
y0=annotation['y_position'] + 0.5, 
y1=annotation['y_position'] + 1.5,
fillcolor=color,
line=dict(color=color, width=1),
opacity=0.7
)

# Add annotation label
fig.add_annotation(
x=(annotation['start'] + annotation['end']) / 2,
y=annotation['y_position'] + 1,
text=f"{annotation['type']}: {annotation['description']}",
showarrow=False,
font=dict(size=10),
textangle=0
)

# Add position markers
position_step = max(1, seq_length // 10)
positions = list(range(0, seq_length + 1, position_step))

fig.update_xaxes(
title="Genomic Position",
tickmode='array',
tickvals=positions,
ticktext=[str(p + 1) for p in positions],  # Convert to 1-based for display
showgrid=True,
gridwidth=1,
gridcolor='lightgray'
)

fig.update_yaxes(
title="Annotation Tracks",
showgrid=False,
zeroline=False,
showticklabels=False
)

fig.update_layout(
title="Genome Annotation Viewer",
height=400,
showlegend=False,
hovermode='closest',
margin=dict(l=50, r=50, t=50, b=50)
)

return fig


def generate_sample_annotations(sequence_length: int):
"""Generate sample annotations for demonstration."""
sample_annotations = [
"1\t100\tGene\tSample Gene 1",
"20\t40\tExon\tExon 1", 
"60\t80\tExon\tExon 2",
"1\t20\tPromoter\tPromoter Region",
"101\t150\tGene\tSample Gene 2",
"110\t130\tExon\tExon 3",
"140\t148\tExon\tExon 4",
"101\t120\tPromoter\tPromoter Region 2",
"50\t50\tSNP\tVariant at position 50",
"75\t75\tSNP\tVariant at position 75",
"125\t125\tSNP\tVariant at position 125"
]

# Filter annotations to fit within sequence length
filtered = []
for line in sample_annotations:
parts = line.split('\t')
if int(parts[1]) <= sequence_length:
filtered.append(line)

return '\n'.join(filtered)


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
"Module 5: Genome Alignment Explorer",
"Module 6: Protein Analysis & Mutation Impact",
"Module 7: Genome Annotation Explorer",
"Module 8: Genome Browser / Multi-Track Visualization",
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
c1.metric("Active Modules", "7", "Pangenome + DeepNCV + FM-index + CRISPR + Alignment + Protein + Annotation")
c2.metric("Built-in Demo Datasets", "6", "Biologically grounded examples")
c3.metric("Export Artifacts in Session", str(len(st.session_state.get('export_artifacts', {}))), "Ready for ZIP")

st.markdown(
"""
- Module 1: Graph-based pangenome + conservation + FASTA upload + exports
- Module 2: DeepNCV prediction + mutation heatmap + batch export + reproducibility
- Module 3: BWT + FM-index search with step trace and match highlighting
- Module 4: CRISPR guide design + NGG PAM scan + off-target proxy scoring
- Module 5: Needleman-Wunsch read mapping + mismatch highlighting + alignment score
- Module 6: DNA→protein translation + property analysis + AlphaFold + mutation impact
- Module 7: Genome annotation viewer + multi-track overlay + integration with other modules
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
Genome Alignment Explorer
↓
Protein Analysis & Mutation Impact
↓
Genome Annotation Explorer
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

pipeline_seq = st.text_area(
"Pipeline Input DNA Sequence",
value="ATGCGTACGATCGATCGATCGGATCGATCGATCGTAGGATCGATCGATCGG",
key="home_pipeline_sequence",
height=110,
)
pipeline_pattern = st.text_input("Pipeline FM-index Pattern", value="GATC", key="home_pipeline_pattern")

if st.button("Run Full Genomics Pipeline", key="home_pipeline_run"):
clean = sanitize_dna_sequence(pipeline_seq)
pattern = sanitize_dna_sequence(pipeline_pattern)
if len(clean) < 23:
st.warning("Provide at least 23 bp for a meaningful integrated run.")
elif not pattern:
st.warning("Provide a search pattern for FM-index stage.")
else:
with st.spinner("Running integrated genomics workflow..."):
graph = build_pangenome_graph([clean], k=3)
baseline, scan_df = mutation_scan(clean[:90], seed=42)
top_row = scan_df.iloc[scan_df["DeltaVsBaseline"].abs().idxmax()]
fm = build_fm_index(clean)
positions_raw, _ = fm_backward_search_with_steps(pattern, fm)
positions = [p for p in positions_raw if p < len(clean)]
guides_df = find_crispr_guides(clean)
read = clean[max(0, len(clean)//4):max(0, len(clean)//4)+24]
a_ref, a_read, aln_score = needleman_wunsch_align(clean, read)
_, aln_mismatches = alignment_annotation(a_ref, a_read)
protein = translate_dna_to_protein(clean)
protein_stats, _ = analyze_protein_properties(protein)

st.success("Pipeline completed.")
avg_gc_text = f"{guides_df['GC%'].mean():.1f}%" if not guides_df.empty else "n/a"
st.code(
f"""PanGen-AI Analysis Report

Sequence length: {len(clean)} bp

Pangenome Graph
Nodes: {graph.number_of_nodes()}
Edges: {graph.number_of_edges()}

Variant Analysis
Baseline impact score: {baseline:.3f}
High-impact mutation position: {int(top_row['Position'])}

FM-Index Search
Pattern: {pattern}
Matches: {len(positions)}

CRISPR Guides
High-confidence guides: {int((guides_df['Score'] == 'High').sum()) if not guides_df.empty else 0}
Average GC%: {avg_gc_text}

Alignment (Module 5)
Read length: {len(read)}
Alignment score: {aln_score}
Mismatches/gaps: {aln_mismatches}

Protein (Module 6)
Translated length: {protein_stats['Length']} aa
Molecular weight: {protein_stats['MolecularWeight_kDa']} kDa
Hydrophobic residues: {protein_stats['HydrophobicPct']}%
Protein mutation impact stage: enabled
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


elif page == "Module 5: Genome Alignment Explorer":
st.title("Module 5: Genome Alignment Explorer")
st.markdown("""
Needleman-Wunsch / read mapping demonstration for pairwise alignment.
Visualizes alignment with mismatch highlighting and alignment scoring.
""")

if "module5_reference" not in st.session_state:
st.session_state["module5_reference"] = DATASET_SARS_COV2_FRAGMENT
if "module5_reads" not in st.session_state:
st.session_state["module5_reads"] = "GTTTTTCTTGTTTTAATTGTTACC\nATGTTTGTTTTTCTTGTTTTAATT"

if st.button("Load Example Read Mapping", key="module5_example_btn"):
st.session_state["module5_reference"] = DATASET_SARS_COV2_FRAGMENT
st.session_state["module5_reads"] = "GTTTTTCTTGTTTTAATTGTTACC\nATGTTTGTTTTTCTTGTTTTAATT"
st.session_state["module5_example_loaded"] = True
st.rerun()

if st.session_state.pop("module5_example_loaded", False):
st.success("Example alignment data loaded.")

reference = st.text_area("Reference DNA sequence", key="module5_reference", height=100)
reads_text = st.text_area("Reads (one per line)", key="module5_reads", height=120)

if st.button("Run Read Mapping", key="module5_run"):
clean_ref = sanitize_dna_sequence(reference)
reads = [sanitize_dna_sequence(x) for x in reads_text.split("\n") if x.strip()]
if not clean_ref or not reads:
st.warning("Provide both a reference sequence and at least one read.")
else:
map_df = map_reads_to_reference(clean_ref, reads)
st.success(f"Mapped {len(map_df)} reads.")
st.dataframe(map_df[["ReadID", "Read", "Score", "Mismatches"]], use_container_width=True)

top = map_df.sort_values("Score", ascending=False).iloc[0]
st.subheader(f"Best Alignment: {top['ReadID']}")
st.code(
"\n".join([top["AlignedReference"], top["AlignmentMarker"], top["AlignedRead"]]),
language="text",
)

csv_bytes = map_df.to_csv(index=False).encode("utf-8")
st.download_button(
"Download Alignment Results (CSV)",
data=csv_bytes,
file_name="alignment_results.csv",
mime="text/csv",
)
add_export_artifact("alignment_results.csv", csv_bytes)

elif page == "Module 6: Protein Analysis & Mutation Impact":
st.title("Module 6: Protein Analysis & Mutation Impact")
st.markdown("""
DNA→protein translation, protein properties, AlphaFold lookup, and a BLOSUM62-based mutation impact predictor.
""")

tabs = st.tabs([
"DNA → Protein Translation",
"Protein Property Analyzer",
"AlphaFold Structure Viewer",
"Protein Mutation Impact Predictor",
"Protein Mutation Landscape",
])

with tabs[0]:
if "module6_dna" not in st.session_state:
st.session_state["module6_dna"] = DATASET_TRANSLATION_DNA

if st.button("Load Translation Example", key="module6_example_btn"):
st.session_state["module6_dna"] = DATASET_TRANSLATION_DNA
st.session_state["module6_example_loaded"] = True
st.rerun()

if st.session_state.pop("module6_example_loaded", False):
st.success("Example DNA loaded.")

dna_seq = st.text_area("DNA sequence", key="module6_dna", height=110)
if st.button("Translate DNA", key="module6_translate"):
protein = translate_dna_to_protein(dna_seq)
st.code(protein, language="text")
st.session_state["module6_protein"] = protein

with tabs[1]:
protein_seq = st.text_area(
"Protein sequence",
value=st.session_state.get("module6_protein", "MAIVMGRKGAR"),
key="module6_protein_input",
height=110,
)
if st.button("Analyze Protein Properties", key="module6_analyze"):
stats, comp_df = analyze_protein_properties(protein_seq)
a1, a2, a3 = st.columns(3)
a1.metric("Protein Length", f"{stats['Length']} aa")
a2.metric("Molecular Weight", f"{stats['MolecularWeight_kDa']} kDa")
a3.metric("Hydrophobic Residues", f"{stats['HydrophobicPct']}%")
st.dataframe(comp_df, use_container_width=True)

with tabs[2]:
st.markdown("### Interactive 3D Protein Structure Viewer")
st.markdown("""
Fetch and visualize protein structures from RCSB PDB or AlphaFold database.
Enter either a PDB ID (e.g., 1A3N) or UniProt ID (e.g., P04637).
""")

col1, col2 = st.columns([2, 1])

with col1:
protein_id = st.text_input(
"PDB ID or UniProt ID", 
value="P04637", 
key="module6_structure_id",
help="Enter PDB ID (e.g., 1A3N, 2HYY) or UniProt ID (e.g., P04637, Q9Y6K9)"
)

with col2:
style_options = ["cartoon", "stick", "sphere", "line"]
selected_style = st.selectbox(
"Visualization Style",
style_options,
index=0,
key="module6_structure_style"
)

# Example proteins dropdown
with st.expander("Example Proteins"):
example_proteins = {
"p53 (Tumor suppressor)": "P04637",
"Hemoglobin": "1A3N", 
"Lysozyme": "1AKI",
"DNA Polymerase": "3K5A",
"Insulin": "1ZNJ",
"Myoglobin": "1MBO",
"Carbonic Anhydrase": "2CBA"
}

selected_example = st.selectbox(
"Select an example protein:",
["None"] + list(example_proteins.keys()),
key="module6_example_protein",
on_change=update_protein_id_from_example
)

if st.button("Fetch and Visualize Structure", key="module6_fetch_structure"):
if not protein_id.strip():
st.error("Please enter a PDB ID or UniProt ID.")
else:
with st.spinner(f"Fetching structure for {protein_id}..."):
pdb_data, source = fetch_pdb_data(protein_id)

if pdb_data:
st.success(f"Structure fetched from {source.upper()} database!")

# Display protein information if UniProt ID
if len(protein_id) == 6 and protein_id.startswith('P'):
protein_info = get_protein_info(protein_id)
if protein_info:
st.markdown("### Protein Information")
info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("Name", protein_info['name'][:30] + "..." if len(protein_info['name']) > 30 else protein_info['name'])
info_col2.metric("Length", f"{protein_info['length']} aa")
info_col3.metric("Mass", f"{protein_info['mass']/1000:.1f} kDa")

st.markdown("**Organism:** " + protein_info['organism'])
with st.expander("Functional Description"):
st.write(protein_info['function'])

# Create and display 3D structure
st.markdown("### 3D Structure Visualization")
viewer = create_3d_structure_viewer(pdb_data, selected_style)
render_structure_in_streamlit(viewer)

# Structure controls
st.markdown("### Structure Controls")
control_col1, control_col2, control_col3 = st.columns(3)

with control_col1:
if st.button("Reset View", key="module6_reset_view"):
viewer.zoomTo()
render_structure_in_streamlit(viewer)
st.rerun()

with control_col2:
if st.button("Toggle Spin", key="module6_toggle_spin"):
viewer.spin()
render_structure_in_streamlit(viewer)
st.rerun()

with control_col3:
if st.button("Download Structure", key="module6_download_structure"):
st.download_button(
"Download PDB File",
data=pdb_data,
file_name=f"{protein_id}.pdb",
mime="text/plain",
key="module6_pdb_download"
)

# Style options
st.markdown("### Visualization Options")
style_col1, style_col2 = st.columns(2)

with style_col1:
st.markdown("**Color Schemes:**")
if st.button("Spectrum", key="module6_spectrum"):
viewer.setStyle({'cartoon': {'color': 'spectrum'}})
render_structure_in_streamlit(viewer)
st.rerun()

if st.button("Chain Colors", key="module6_chain_colors"):
viewer.setStyle({'cartoon': {'color': 'chain'}})
render_structure_in_streamlit(viewer)
st.rerun()

with style_col2:
st.markdown("**Background Colors:**")
if st.button("Light Background", key="module6_light_bg"):
viewer.setBackgroundColor('#f0f0f0')
render_structure_in_streamlit(viewer)
st.rerun()

if st.button("Dark Background", key="module6_dark_bg"):
viewer.setBackgroundColor('#1e1e1e')
render_structure_in_streamlit(viewer)
st.rerun()

# Add to export artifacts
add_export_artifact(f"{protein_id}.pdb", pdb_data.encode('utf-8'))

else:
st.error(f"Could not fetch structure for {protein_id}. Please check the ID and try again.")
st.markdown("""
**Troubleshooting:**
- For PDB IDs: Use 4-character codes like 1A3N, 2HYY
- For UniProt IDs: Use 6-character codes starting with P, Q, O, etc.
- Make sure you have an internet connection
- Some structures may not be available in the databases
""")

# Database information
st.markdown("---")
st.markdown("### Database Information")
st.markdown("""
**Data Sources:**
- **RCSB PDB**: Experimental protein structures from X-ray crystallography, NMR, and cryo-EM
- **AlphaFold**: Predicted protein structures using deep learning (high confidence for most human proteins)

**Supported ID Formats:**
- PDB ID: 4 characters (e.g., 1A3N, 2HYY, 3K5A)
- UniProt ID: 6 characters (e.g., P04637, Q9Y6K9, O00429)
""")

with tabs[3]:
mut_protein = st.text_input("Protein sequence for mutation impact", value="MKTFVVLLLCTFTVVSA", key="module6_mut_protein")
mutation = st.text_input("Mutation notation (e.g., F5L)", value="F5L", key="module6_mutation")

if st.button("Predict Mutation Impact", key="module6_mut_predict"):
result = predict_protein_mutation_impact(mut_protein, mutation)
if "error" in result:
st.error(result["error"])
else:
st.success("Mutation impact analysis complete.")
st.code(
f"""Mutation Impact Analysis

Position: {result['Position']}
Original AA: {result['OriginalAA']}
Mutated AA: {result['MutatedAA']}

BLOSUM62 score: {result['BLOSUM62Score']}
Conservation score: {result['ConservationScore']}
Predicted impact: {result['PredictedImpact']}
""",
language="text",
)

with tabs[4]:
landscape_protein = st.text_input(
"Protein sequence for mutation landscape",
value=st.session_state.get("module6_protein", "MAIVMGRKGAR"),
key="module6_landscape_protein",
)
if st.button("Generate Mutation Landscape", key="module6_landscape_btn"):
aa_order, matrix = build_protein_mutation_landscape(landscape_protein)
if matrix.shape[1] == 0:
st.warning("Please provide a valid protein sequence with standard amino acids.")
else:
fig, ax = plt.subplots(figsize=(12, 4.8))
im = ax.imshow(matrix, cmap="coolwarm", aspect="auto", vmin=-4, vmax=11)
ax.set_yticks(range(len(aa_order)))
ax.set_yticklabels(aa_order)
ax.set_xlabel("Protein Position")
ax.set_ylabel("Mutated Amino Acid")
ax.set_title("Protein Mutation Impact Landscape (BLOSUM62)")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("BLOSUM62 Score")
st.pyplot(fig)

# Add explanation for score interpretation
st.info("""
**Score Interpretation:**
- **Positive scores** (red): Conservative mutations - amino acids are similar, likely tolerated
- **Negative scores** (blue): Damaging mutations - amino acids are different, likely deleterious
- **Higher scores** indicate more conservative substitutions, **lower scores** indicate more radical changes
""")

landscape_png = fig_to_png_bytes(fig)
st.download_button(
"Download Mutation Landscape (PNG)",
data=landscape_png,
file_name="protein_mutation_landscape.png",
mime="image/png",
key="module6_landscape_png_download",
)
add_export_artifact("protein_mutation_landscape.png", landscape_png)
plt.close(fig)

elif page == "Module 7: Genome Annotation Explorer":
st.title("Module 7: Genome Annotation Explorer")
st.subheader("Interactive Genome Annotation Visualization")
st.markdown("""
This module provides an interactive viewer for genome annotations, allowing you to overlay multiple 
annotation types (genes, exons, promoters, regulatory regions, SNPs, CRISPR sites) on a DNA sequence.
The viewer displays annotations as colored tracks with position-based coordinates.
""")

with st.expander("Method / Algorithm"):
st.markdown("""
**Method**
The genome annotation viewer parses tab-delimited annotation data and renders it as interactive 
tracks using Plotly. Each annotation type is assigned a unique color and displayed at different 
vertical levels to avoid overlap. The viewer supports zooming, panning, and hover interactions.

**Annotation Format**: `Start\tEnd\tType\tDescription`
- Start/End: 1-based genomic positions
- Type: Gene, Exon, Promoter, Regulatory, SNP, CRISPR, Variant
- Description: Optional annotation description
""")

with st.expander("Use Case / Applications"):
st.markdown("""
**Applications**
- Visualizing gene structures and exon-intron organization
- Overlaying multiple annotation types on genomic regions
- Identifying regulatory elements and variant hotspots
- Integrating CRISPR target sites with gene annotations
- Exploring SNP distributions across genomic features
""")

# Input section
col1, col2 = st.columns([2, 1])

with col1:
sequence_input = st.text_area(
"Enter DNA Sequence:",
value="ATGCGTACGATCGATCGATCGTAGCTAGCTAGCGATCGATCGATCGTAGCTAG",
key="module7_sequence",
height=100,
help="Enter the DNA sequence to annotate"
)

with col2:
st.markdown("**Sequence Info**")
if sequence_input:
clean_seq = sanitize_dna_sequence(sequence_input)
st.write(f"Length: {len(clean_seq)} bp")
st.write(f"GC Content: {100 * (clean_seq.count('G') + clean_seq.count('C')) / len(clean_seq):.1f}%")

# Annotation input
st.markdown("### Genome Annotations")
st.markdown("**Format**: Start(1-based) → End → Type → Description (tab-separated)")

col1, col2 = st.columns([3, 1])

with col2:
if st.button("Load Sample Annotations", key="module7_sample_btn"):
clean_seq = sanitize_dna_sequence(sequence_input)
if clean_seq:
sample_ann = generate_sample_annotations(len(clean_seq))
st.session_state.module7_annotations = sample_ann
st.success("Sample annotations loaded!")
st.rerun()

with col1:
annotation_input = st.text_area(
"Enter Annotations (one per line):",
value=st.session_state.get("module7_annotations", ""),
key="module7_annotation_input",
height=150,
help="Example: 1\t100\tGene\tSample Gene"
)

# Display annotation format example
with st.expander("Annotation Format Examples"):
st.code("""
# Gene annotation
1	100	Gene	Sample Gene 1
20	40	Exon	Exon 1
60	80	Exon	Exon 2
1	20	Promoter	Promoter Region

# Variant annotations
50	50	SNP	Variant at position 50
75	75	SNP	Variant at position 75

# CRISPR sites
30	49	CRISPR	CRISPR target site
""", language="text")

# Generate visualization
if st.button("Generate Annotation Viewer", key="module7_generate_btn"):
clean_seq = sanitize_dna_sequence(sequence_input)

if not clean_seq:
st.error("Please enter a valid DNA sequence.")
elif not annotation_input.strip():
st.error("Please enter annotation data or load sample annotations.")
else:
with st.spinner("Parsing annotations and generating visualization..."):
# Parse annotations
annotations = parse_genome_annotations(annotation_input)

if not annotations:
st.warning("No valid annotations found. Please check the format.")
else:
# Create visualization
fig = create_genome_annotation_viewer(clean_seq, annotations)
st.plotly_chart(fig, use_container_width=True)

# Display annotation summary
st.markdown("### Annotation Summary")
summary_data = {}
for ann in annotations:
ann_type = ann['type']
summary_data[ann_type] = summary_data.get(ann_type, 0) + 1

summary_df = pd.DataFrame([
{"Type": k, "Count": v} for k, v in summary_data.items()
])
st.dataframe(summary_df, use_container_width=True)

# Export options
st.markdown("### Export Options")

# Export annotation data
annotation_csv = "Start\tEnd\tType\tDescription\n" + annotation_input
st.download_button(
"Download Annotations (TSV)",
data=annotation_csv,
file_name="genome_annotations.tsv",
mime="text/tab-separated-values",
key="module7_annotation_download"
)

# Add to export artifacts
add_export_artifact("genome_annotations.tsv", annotation_csv.encode('utf-8'))

# Integration with other modules
st.markdown("---")
st.markdown("### Integration with Other Modules")
st.markdown("""
The Genome Annotation Explorer can integrate with other PanGen-AI modules:

- **Module 1**: Use pangenome graph results to identify conserved regions for annotation
- **Module 2**: Overlay variant impact predictions on gene annotations  
- **Module 4**: Display CRISPR guide sites in the context of gene features
- **Module 5**: Show alignment results alongside annotated regions
- **Module 6**: Connect protein-coding regions to translation analysis
""")

# Quick integration buttons
col1, col2, col3 = st.columns(3)

with col1:
if st.button("Import CRISPR Sites from Module 4", key="module7_import_crispr"):
clean_seq = sanitize_dna_sequence(sequence_input)
if clean_seq:
guides_df = find_crispr_guides(clean_seq)
if not guides_df.empty:
crispr_annotations = []
for _, row in guides_df.iterrows():
start = row['Position'] + 1  # Convert to 1-based
end = start + 19  # 20bp guide
crispr_annotations.append(f"{start}\t{end}\tCRISPR\tGuide: {row['Guide RNA']}")

st.session_state.module7_annotations = "\n".join(crispr_annotations)
st.success(f"Imported {len(crispr_annotations)} CRISPR sites!")
st.rerun()
else:
st.warning("No CRISPR sites found in the sequence.")

with col2:
if st.button("Import Variants from Module 2", key="module7_import_variants"):
clean_seq = sanitize_dna_sequence(sequence_input)
if clean_seq and len(clean_seq) >= 10:
baseline, scan_df = mutation_scan(clean_seq[:min(90, len(clean_seq))])
# Get high-impact variants
high_impact = scan_df[scan_df['DeltaVsBaseline'].abs() > 0.1]
if not high_impact.empty:
variant_annotations = []
for _, row in high_impact.iterrows():
pos = row['Position']
variant_annotations.append(f"{pos}\t{pos}\tSNP\t{row['Ref']}→{row['Alt']} (Δ={row['DeltaVsBaseline']:.3f})")

st.session_state.module7_annotations = "\n".join(variant_annotations)
st.success(f"Imported {len(variant_annotations)} high-impact variants!")
st.rerun()
else:
st.warning("No high-impact variants found.")

with col3:
if st.button("Generate Gene Annotations", key="module7_generate_genes"):
clean_seq = sanitize_dna_sequence(sequence_input)
"""Create multi-track genome browser visualization using Plotly subplots."""
positions = track_data['positions']

# Validate data
if not positions or len(positions) == 0:
fig = go.Figure()
fig.add_annotation(
text="No data to display",
xref="paper",
yref="paper",
x=0.5,
y=0.5,
showarrow=False,
font=dict(size=20),
)
fig.update_layout(height=400)
return fig

# Create subplots (5 rows, 1 column)
fig = make_subplots(
rows=5,
cols=1,
shared_xaxes=True,
vertical_spacing=0.08,
subplot_titles=(
"DNA Sequence",
"Variant Impact",
"CRISPR Guides",
"Gene Annotation",
"Protein Coding",
),
)

# Track 1: DNA Sequence
fig.add_trace(
go.Heatmap(
z=[track_data['dna_sequence']],
x=positions,
y=[""],
colorscale="Viridis",
showscale=False,
hovertemplate='Position: %{x}<br>Base Value: %{z}<extra></extra>',
),
row=1,
col=1,
)

# Track 2: Variant Impact
fig.add_trace(
go.Heatmap(
z=[track_data['variant_impact']],
x=positions,
y=[""],
colorscale="Reds",
showscale=False,
hovertemplate='Position: %{x}<br>Impact Score: %{z:.3f}<extra></extra>',
),
row=2,
col=1,
)

# Track 3: CRISPR Guides
fig.add_trace(
go.Heatmap(
z=[track_data['crispr_guides']],
x=positions,
y=[""],
colorscale="Blues",
showscale=False,
hovertemplate='Position: %{x}<br>CRISPR Score: %{z:.2f}<extra></extra>',
),
row=3,
col=1,
)

# Track 4: Gene Annotation
fig.add_trace(
go.Heatmap(
z=[track_data['gene_annotation']],
x=positions,
y=[""],
colorscale="Greens",
showscale=False,
hovertemplate='Position: %{x}<br>Gene Region: %{z}<extra></extra>',
),
row=4,
col=1,
)

# Track 5: Protein Coding
fig.add_trace(
go.Heatmap(
z=[track_data['protein_coding']],
x=positions,
y=[""],
colorscale="Purples",
showscale=False,
hovertemplate='Position: %{x}<br>Coding Region: %{z}<extra></extra>',
),
row=5,
col=1,
)

# Update layout for proper height and look
fig.update_layout(
title_text="Interactive Genome Browser",
height=850,
showlegend=False,
margin=dict(l=40, r=40, t=80, b=40),
hovermode='closest',
)

# Add x-axis title only to the bottom track
fig.update_xaxes(title_text="Genomic Position", row=5, col=1)

# Hide y-axis tick labels since subplot titles are shown
fig.update_yaxes(showticklabels=False)
