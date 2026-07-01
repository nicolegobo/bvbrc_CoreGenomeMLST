#!/usr/bin/env python3
import argparse
import click
import csv
import glob
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# Ensure files adhere to the rules defined in the the chewbbaca allele call function
def chewbbaca_filename_format(filename):
    # Rule 1 Replace spaces and illegal characters with underscores
    name, ext = os.path.splitext(filename)
    if ext != ".fasta":
        # add_extension = os.path.join(filename, ".fasta")
        add_extension = filename + ".fasta"
        name, ext = os.path.splitext(add_extension)

    # Rule 1: Remove extra dots in the name (keep only one before the extension)
    name = name.replace(".", "_")

    # Rule 2: Replace spaces and illegal characters with underscores
    name = re.sub(r'[^A-Za-z0-9_\-]', '_', name)

    # Ensure we return a properly formatted filename
    return "{}{}".format(name, ext)

def clean_file(allelic_profile):
    mapping = {}
    input_path = Path(allelic_profile)
    stem = input_path.stem
    orig_col1_zero_file = input_path.with_name(stem + "_clean.tsv")

    with open(allelic_profile, newline="", encoding="utf-8") as fin, \
         open(orig_col1_zero_file, "w", newline="", encoding="utf-8") as forig:

        reader = csv.reader(fin, delimiter="\t")
        orig_writer = csv.writer(forig, delimiter="\t")

        # ---- Header handling ----
        header = next(reader)  # first row (unaltered header)

        # File A header: unchanged
        orig_writer.writerow(header)

        # ---- Data rows ----
        seen_ids = set()  # track genome IDs already written; first occurrence (newest assignment) wins
        for idx, row in enumerate(reader):
            if not row:
                continue  # skip completely empty lines

            # Normalize genome ID (chewBBACA uses '_' in place of '.') before dedup
            col1_value = row[0].replace('_', '.')

            # Keep only the first occurrence of each genome ID — JoinProfiles puts
            # newly computed assignments before the older reference rows, so the
            # first occurrence is always the new assignment.
            if col1_value in seen_ids:
                print("Skipping duplicate genome ID: {}".format(col1_value))
                continue
            seen_ids.add(col1_value)

            # Record mapping from index -> original first-column value
            mapping[idx] = col1_value

            # Start with a copy of the row
            new_row_orig = row[:]
            new_row_orig[0] = col1_value

            # For all columns after the first (index 1..end),
            # replace any non-numeric (non-digits-only) value with "0"
            for j in range(1, len(row)):
                if not is_numeric_digits_only(row[j]):
                    new_row_orig[j] = "0"

            # Write out the row
            orig_writer.writerow(new_row_orig)

def copy_new_file(clean_fasta_dir, new_name, filename, original_path):
    # deal with moving the files
    clean_path = os.path.join(clean_fasta_dir, new_name)
    # If the filename was changed, copy the renamed file to the output directory
    if filename != new_name:
        print("Renaming and copying: {} -> {}".format(filename, new_name))
        shutil.copy2(original_path, clean_path)
    else:
        print("Copying: {}".format(filename))
        shutil.copy2(original_path, clean_path)

def is_numeric_digits_only(s: str) -> bool:
    """
    Returns True if s consists ONLY of digits 0-9 (no dots, minus signs, etc),
    and is not empty. Otherwise False.
    """
    s = s.strip()
    return s != "" and s.isdigit()

@click.group()
def cli():
    """ This script supports the Core Genome MLST service with multiple commands."""
    pass

@cli.command()
@click.argument("service_config")
def clean_fasta_filenames(service_config):
    """Ensure files adhere to the rules defined by chewbbacca"""

    with open(service_config) as file:
        data = json.load(file)
        raw_fasta_dir = data["raw_fasta_dir"]
        clean_fasta_dir = data["clean_data_dir"]
        for filename in sorted(os.listdir(raw_fasta_dir)):
            original_path = os.path.join(raw_fasta_dir, filename)
            new_name = chewbbaca_filename_format(filename)
            copy_new_file(clean_fasta_dir, new_name, filename, original_path)

@cli.command()
@click.argument("allelic_profile")
def clean_allelic_profile(allelic_profile):
    """Clean a ChewBBACA allelic profile TSV file: replace non-numeric values with 0,
    and reformat the genome ID column by replacing any '_' with '.'"""
    clean_file(allelic_profile)

def generate_table_html_2(kchooser_df, table_width='75%'):
    headers = ''.join(f'<th>{header}</th>' for header in kchooser_df.columns)
    rows = ''
    for _, row in kchooser_df.iterrows():
        row_html = ''
        for column in kchooser_df.columns:
            cell_value = row[column]
            if pd.api.types.is_numeric_dtype(type(cell_value)):
                if isinstance(cell_value, (int, np.integer)):
                    formatted_value = f"{cell_value:,}"
                elif isinstance(cell_value, (float, np.floating)):
                    formatted_value = f"{cell_value:,.2f}"
                else:
                    formatted_value = str(cell_value)
                row_html += f'<td style="text-align: center;">{formatted_value}</td>'
            else:
                row_html += f'<td>{cell_value}</td>'
        rows += f'<tr>{row_html}</tr>'
    table_html = f'''
    <div style="text-align: center;">
    <table style="margin: auto; width: {table_width}; border-collapse: collapse; border: 1px solid black;">
        <thead>
            <tr>{headers}</tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    </div>
    '''
    return table_html


def read_plotly_html(plot_path):
    with open(plot_path, 'r') as file:
        plotly_html_content = file.read()
    extracted_content = re.findall(r'<body>(.*?)</body>', plotly_html_content, re.DOTALL)
    plotly_graph_content = extracted_content[0] if extracted_content else ''
    return plotly_graph_content


# ---------------------------------------------------------------------------
# Classification codes chewBBACA uses for missing / problematic loci
# ---------------------------------------------------------------------------
MISSING_CODES = {"LNF", "ASM", "ALM", "NIPH", "NIPHEM", "PLOT3", "PLOT5"}


# ---------------------------------------------------------------------------
# cgMLST report helper functions
# ---------------------------------------------------------------------------

def create_cgmlst_metadata_table(metadata_json, tsv_out):
    """Load genome metadata, keeping genome_id dots intact (not converted to underscores).

    Parameters
    ----------
    metadata_json : str
        Path to genome_metadata.json.
    tsv_out : str
        Path to write a TSV copy of the metadata.

    Returns
    -------
    metadata : list of dict
    metadata_df : pd.DataFrame
    """
    with open(metadata_json) as f:
        metadata = json.load(f)

    all_headers = sorted({key for row in metadata for key in row.keys()})

    for row in metadata:
        for header in all_headers:
            row.setdefault(header, "N/A")

    metadata_df = pd.json_normalize(metadata)
    if "genome_id" in metadata_df.columns:
        cols = ["genome_id"] + [c for c in metadata_df.columns if c != "genome_id"]
        metadata_df = metadata_df[cols]

    def to_pascal_case(name):
        return "".join(p.capitalize() for p in name.split("_"))

    metadata_df.columns = [to_pascal_case(c) for c in metadata_df.columns]
    metadata_df.to_csv(tsv_out, index=False, sep="\t")

    return metadata, metadata_df


def write_cgmlst_distance_report(ids, dist_matrix, report_path):
    """Write a pairwise distance report matching kSNPdist.report format.

    Produces a tab-separated file with a header row and one line per unique
    genome pair (upper triangle only).  Columns: distance, genome_id_1, genome_id_2.

    Parameters
    ----------
    ids : list of str
        Genome IDs in the same order as dist_matrix rows/columns.
    dist_matrix : np.ndarray or list of lists, shape (n, n)
        Symmetric integer distance matrix.
    report_path : str
        Output file path.
    """
    with open(report_path, "w") as f:
        f.write("distance\tgenome_id_1\tgenome_id_2\n")
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                f.write("{}\t{}\t{}\n".format(int(dist_matrix[i][j]), ids[i], ids[j]))


def parse_result_alleles(result_alleles_tsv):
    """Read result_alleles.tsv and return summary information.

    Parameters
    ----------
    result_alleles_tsv : str
        Path to chewBBACA AlleleCall result_alleles.tsv.

    Returns
    -------
    genome_ids : list of str
        Sample identifiers (first column, underscores replaced with periods).
    loci_ids : list of str
        Locus identifiers (header row, columns 1 onwards).
    raw_df : pd.DataFrame
        Raw DataFrame with original values (genome IDs as index).
    coverage_df : pd.DataFrame
        Per-sample counts of: exact, INF, and each missing code.
    """
    # pandas 2.x ignores dtype for the index column when index_col is used,
    # causing numeric-looking IDs like "211759.20" to lose trailing zeros.
    # Read all columns as str first, then set the index manually.
    df = pd.read_csv(result_alleles_tsv, sep="\t", dtype=str)
    _id_col = df.columns[0]
    df[_id_col] = df[_id_col].str.replace("_", ".", regex=False)
    df = df.set_index(_id_col)

    genome_ids = df.index.tolist()
    loci_ids = df.columns.tolist()

    records = []
    for genome_id, row in df.iterrows():
        counts = {"genome_id": genome_id, "Exact": 0, "INF": 0}
        for code in MISSING_CODES:
            counts[code] = 0
        for val in row:
            val = str(val).strip()
            if val.startswith("INF-"):
                counts["INF"] += 1
            elif val in MISSING_CODES:
                counts[val] += 1
            elif val.isdigit():
                counts["Exact"] += 1
        records.append(counts)

    coverage_df = pd.DataFrame(records).set_index("genome_id")

    return genome_ids, loci_ids, df, coverage_df


# ---------------------------------------------------------------------------
# Distance calculation — exact replication of chewBBACA's pipeline
# ---------------------------------------------------------------------------
# These functions replicate the logic from chewBBACA's:
#   - CHEWBBACA/utils/iterables_manipulation.py  (replace_chars)
#   - CHEWBBACA/ExtractCgMLST/determine_cgmlst.py (binarize_matrix, presAbs, compute_cgMLST)
#   - CHEWBBACA/utils/distance_matrix.py          (compute_distances)
#   - CHEWBBACA/AlleleCallEvaluator/evaluate_calls.py (pipeline orchestration)
# ---------------------------------------------------------------------------

def _chewbbaca_replace_chars(column, missing_char='0'):
    """Mask a single column of allelic calls — exact replica of chewBBACA's
    iterables_manipulation.replace_chars.

    Applied column-by-column via DataFrame.apply(), matching the way
    chewBBACA invokes it: ``profiles_matrix.apply(im.replace_chars)``.

    Steps:
      1. Strip 'INF-' prefix from inferred alleles  (INF-5  -> 5)
      2. Strip '*'    prefix from Chewie-NS alleles  (*5     -> 5)
      3. Replace all remaining non-numeric values    (LNF, ASM, ... -> 0)
    """
    replaced_inf = column.replace(to_replace='INF-', value='', regex=True)
    replaced_ns = replaced_inf.replace(to_replace=r'\*', value='', regex=True)
    replaced_special = replaced_ns.replace(to_replace=r'\D+.*',
                                           value=missing_char, regex=True)
    return replaced_special


def _chewbbaca_binarize(column):
    """Convert a column to presence (1) / absence (0) — exact replica of
    chewBBACA's determine_cgmlst.binarize_matrix."""
    return np.int64(pd.to_numeric(column) > 0)


def _chewbbaca_above_threshold(column, column_length, threshold):
    """Test whether a locus is present at or above *threshold* — exact replica
    of chewBBACA's determine_cgmlst.above_threshold."""
    return (np.sum(column) / column_length) >= threshold


def _compute_distance_matrix(np_matrix):
    """Compute a symmetric pairwise allelic distance matrix — exact replica
    of chewBBACA's distance_matrix.compute_distances.

    For each pair of samples, the distance is the number of loci where both
    have a non-zero (exact) allele AND those alleles differ.

    Parameters
    ----------
    np_matrix : np.ndarray, shape (n_samples, n_loci), dtype int32

    Returns
    -------
    dist_matrix : np.ndarray, shape (n_samples, n_samples), dtype int32
    """
    n = np_matrix.shape[0]
    dist_matrix = np.zeros((n, n), dtype='int32')

    for i in range(n):
        current_row = np_matrix[i:i + 1, :]
        permutation_rows = np_matrix[i:, :]
        # non-zero only where both samples have an exact call
        multiplied = current_row * permutation_rows
        # count positions where both are exact AND alleles differ
        pairwise_diffs = np.count_nonzero(
            multiplied * (current_row - permutation_rows), axis=-1
        ).astype('int32')
        dist_matrix[i, i:] = pairwise_diffs
        dist_matrix[i:, i] = pairwise_diffs

    return dist_matrix


def chewbbaca_distance_pipeline(result_alleles_tsv):
    """Full chewBBACA-compatible distance computation pipeline.

    Replicates the exact sequence of operations from chewBBACA's
    AlleleCallEvaluator/evaluate_calls.py:

      1. Read allelic profiles (letting pandas infer dtypes, matching chewBBACA)
      2. Mask with column-by-column apply of replace_chars
      3. Compute presence-absence matrix (binarize_matrix)
      4. Determine cgMLST at 100% threshold (loci present in ALL samples)
      5. Compute symmetric distance matrix on cgMLST loci only

    Parameters
    ----------
    result_alleles_tsv : str
        Path to chewBBACA result_alleles.tsv.

    Returns
    -------
    genome_ids : list of str
        Sample identifiers (original, with underscores).
    genome_ids_display : list of str
        Sample identifiers (underscores replaced with periods, for display).
    cgmlst_loci : list of str
        Locus identifiers in the cgMLST-100%.
    dist_matrix : np.ndarray, shape (n, n), dtype int32
        Symmetric pairwise distance matrix.
    """
    # Step 1 — Read profiles matching chewBBACA
    # chewBBACA: pd.read_csv(file, header=0, index_col=0, sep='\t', low_memory=False)
    # pandas 2.x ignores dtype for the index column when index_col is used,
    # causing numeric-looking IDs like "211759.20" to lose trailing zeros.
    # Read all columns as str first, then set the index manually.
    profiles = pd.read_csv(result_alleles_tsv, header=0,
                           sep='\t', low_memory=False, dtype=str)
    _id_col = profiles.columns[0]
    profiles = profiles.set_index(_id_col)
    genome_ids = profiles.index.tolist()
    genome_ids_display = [gid.replace('_', '.') for gid in genome_ids]

    # Step 2 — Mask (column-by-column, matching chewBBACA)
    # chewBBACA: masked_profiles = profiles_matrix.apply(im.replace_chars)
    masked_profiles = profiles.apply(_chewbbaca_replace_chars)

    # Step 3 — Presence-absence matrix
    # chewBBACA: pa_matrix = masked_profiles.apply(binarize_matrix)
    pa_matrix = masked_profiles.apply(_chewbbaca_binarize)

    # Step 3b — Drop genomes with zero valid allele calls (all loci absent).
    # A single all-zero row prevents any locus from reaching the 100% threshold.
    loci_per_genome = pa_matrix.sum(axis=1)
    zero_coverage = loci_per_genome[loci_per_genome == 0].index.tolist()
    if zero_coverage:
        print("WARNING: {} genome(s) have 0 valid allele calls and will be "
              "excluded from the distance matrix: {}".format(
                  len(zero_coverage), zero_coverage))
        pa_matrix = pa_matrix.drop(index=zero_coverage)
        masked_profiles = masked_profiles.drop(index=zero_coverage)
        genome_ids = [g for g in genome_ids if g not in zero_coverage]
        genome_ids_display = [gid.replace('_', '.') for gid in genome_ids]

    # Step 4 — cgMLST at 100% (loci present in every sample)
    # chewBBACA: compute_cgMLST(pa_matrix, sample_ids, 1, len(sample_ids))
    n_samples, _ = pa_matrix.shape
    is_above = pa_matrix.apply(_chewbbaca_above_threshold,
                               args=(n_samples, 1))
    cgmlst_loci = pa_matrix.columns[is_above].tolist()

    if len(cgmlst_loci) == 0:
        raise ValueError(
            "cgMLST is composed of 0 loci — cannot compute distance matrix. "
            "Check sample quality or loci coverage.")

    # Step 5 — Subset to cgMLST loci and convert to numpy
    # chewBBACA: cgMLST_matrix = masked_profiles[cgMLST_genes]
    #            then tsv_to_nparray reads it back with dtype=int32
    cgmlst_matrix = masked_profiles[cgmlst_loci]
    np_matrix = cgmlst_matrix.apply(pd.to_numeric).values.astype('int32')

    # Step 6 — Compute distances
    # chewBBACA: distance_matrix.compute_distances (same formula)
    dist_matrix = _compute_distance_matrix(np_matrix)

    return genome_ids, genome_ids_display, cgmlst_loci, dist_matrix


# ---------------------------------------------------------------------------
# Heatmap clustering
# ---------------------------------------------------------------------------

def cluster_heatmap_data(genome_ids, dist_matrix):
    """Cluster genome IDs by hierarchical (single-linkage) clustering.

    Parameters
    ----------
    genome_ids : list of str
    dist_matrix : list of lists or np.ndarray, shape (n, n)

    Returns
    -------
    clustered_labels : list of str
    clustered_matrix : list of lists (int)
    """
    matrix = np.array(dist_matrix)
    dist_array = squareform(matrix)
    linkage_result = linkage(dist_array, method="single")
    idx = leaves_list(linkage_result)
    clustered_matrix = [[int(matrix[i][j]) for j in idx] for i in idx]
    clustered_labels = [genome_ids[i] for i in idx]
    return clustered_labels, clustered_matrix


def create_loci_coverage_plot(coverage_df):
    """Create a stacked bar chart showing loci call status per sample.

    Parameters
    ----------
    coverage_df : pd.DataFrame
        Index = genome_id; columns = Exact, INF, LNF, ASM, ALM, ...

    Returns
    -------
    barplot_html : str
        HTML fragment (body content of the Plotly figure).
    """
    colour_map = {
        "Exact":  "#2ca02c",
        "INF":     "#17becf",
        "LNF":     "#d62728",
        "ASM":     "#ff7f0e",
        "ALM":     "#9467bd",
        "NIPH":    "#8c564b",
        "NIPHEM":  "#e377c2",
        "PLOT3":   "#bcbd22",
        "PLOT5":   "#7f7f7f",
    }

    cols = [c for c in colour_map if c in coverage_df.columns and coverage_df[c].sum() > 0]

    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Bar(
            name=col,
            x=coverage_df.index.tolist(),
            y=coverage_df[col].tolist(),
            marker_color=colour_map[col],
        ))

    fig.update_layout(
        barmode="stack",
        title="Loci Coverage per Sample",
        xaxis_title="Sample",
        yaxis_title="Number of Loci",
        xaxis_tickangle=-45,
        legend_title="Classification",
        template="plotly_white",
    )

    out_path = "loci_coverage_barplot.html"
    fig.write_html(out_path, include_plotlyjs=False)
    barplot_html = read_plotly_html(out_path)
    os.remove(out_path)
    return barplot_html


def run_quast(quast_output_dir, fasta_file, threads=12, min_contig=200):
    """Run QUAST on a FASTA file and write results to quast_output_dir.

    Args:
        quast_output_dir (str): Directory to save QUAST output.
        fasta_file (str): Path to the input FASTA file.
        threads (int): Number of threads. Defaults to 12.
        min_contig (int): Minimum contig size. Defaults to 200.
    """
    os.makedirs(quast_output_dir, exist_ok=True)

    quast_cmd = [
        "quast.py",
        "-o", quast_output_dir,
        "-t", str(threads),
        "--min-contig", str(min_contig),
        fasta_file
    ]

    try:
        print(f"Running QUAST with command: {' '.join(quast_cmd)}")
        subprocess.run(quast_cmd, check=True)
        print("QUAST analysis completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running QUAST: {e}")


def parse_quast_results(quast_output_dir):
    """Parse key metrics from QUAST report.tsv.

    Parameters
    ----------
    quast_output_dir : str
        Directory containing QUAST output (must contain report.tsv).

    Returns
    -------
    dict or None
        Dict of selected metrics, or None if report.tsv is not found.
    """
    report_path = os.path.join(quast_output_dir, "report.tsv")
    if not os.path.exists(report_path):
        return None

    metrics = {}
    keys_of_interest = {
        "# contigs",
        "Largest contig",
        "Total length",
        "N50",
        "GC (%)",
        "L50",
    }

    with open(report_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] in keys_of_interest:
                metrics[parts[0]] = parts[1]

    return metrics if metrics else None


def create_low_coverage_warning(coverage_df, n_loci, config_json_path="config.json"):
    """Return a warning HTML block if any genome has < 75% exact/INF allele calls.

    For each flagged genome, QUAST is run using the FASTA found at
    <input_data_dir>/raw_fastas/<genome_id> (from config.json). Results are written to
    <output_data_dir>/qc_low_quality_genomes/<genome_id>/ and summarised in the HTML.

    Parameters
    ----------
    coverage_df : pd.DataFrame
        Index = genome_id; columns include Exact, INF, and missing-code counts.
    n_loci : int
        Total number of loci in the schema.
    config_json_path : str
        Path to config.json.

    Returns
    -------
    str
        Warning HTML block, or empty string if all genomes pass the threshold.
    """
    THRESHOLD = 0.75

    exact_plus_inf = coverage_df["Exact"] + coverage_df.get("INF", pd.Series(0, index=coverage_df.index))
    pct_exact = exact_plus_inf / n_loci if n_loci > 0 else exact_plus_inf * 0
    flagged = pct_exact[pct_exact < THRESHOLD].index.tolist()

    if not flagged:
        return ""

    genome_list_html = "".join(f"<li><code>{g}</code> ({pct_exact[g]*100:.1f}% exact)</li>" for g in flagged)

    # Read config.json once for both genome group link and QUAST paths
    genome_group_html = ""
    quast_results = {}
    try:
        if os.path.exists(config_json_path):
            with open(config_json_path) as f:
                config = json.load(f)

            genome_group = config.get("params", {}).get("input_genome_group", "")
            if genome_group:
                group_name = os.path.basename(genome_group.rstrip("/"))
                genome_group_html = (
                    f'<p><a id="genomeGroupLink" href="#" target="_blank">Click here</a> '
                    f'to review or edit your input genome group ({group_name}).</p>'
                    f'<script>'
                    f'document.getElementById("genomeGroupLink").href = '
                    f'window.location.origin + "/workspace{genome_group}";'
                    f'</script>'
                )

            input_data_dir = config.get("input_data_dir", "")
            output_dir = config.get("output_data_dir", "")
            params = config.get("params", {})
            output_path = params.get("output_path", "") + "/." + params.get("output_file", "")
            if input_data_dir and output_dir:
                raw_fasta_dir = os.path.join(input_data_dir, "raw_fastas")
                for genome_id in flagged:
                    fasta_file = os.path.join(raw_fasta_dir, genome_id)
                    if not os.path.exists(fasta_file):
                        print(f"QUAST skipped for {genome_id}: FASTA not found at {fasta_file}")
                        continue
                    quast_out = os.path.join(output_dir, "qc_low_quality_genomes", genome_id)
                    run_quast(quast_out, fasta_file)
                    icarus = os.path.join(quast_out, "icarus.html")
                    if os.path.exists(icarus):
                        os.remove(icarus)
                    metrics = parse_quast_results(quast_out)
                    if metrics:
                        quast_results[genome_id] = {"metrics": metrics, "output_path": output_path}
    except Exception as e:
        print(f"Warning: could not complete QUAST analysis: {e}")

    # Build QUAST summary table with links
    quast_html = ""
    if quast_results:
        quast_cols = ["# contigs", "Largest contig", "Total length", "N50", "GC (%)", "L50"]
        header_cells = "".join(f'<th style="padding:4px 8px;">{c}</th>' for c in quast_cols)
        rows_html = ""
        for genome_id in flagged:
            result = quast_results.get(genome_id)
            if result is None:
                continue
            metrics = result["metrics"]
            ws_base = result["output_path"]
            ws_report = f"{ws_base}/qc_low_quality_genomes/{genome_id}/report.pdf"
            ws_dir = f"{ws_base}/qc_low_quality_genomes/{genome_id}"
            cells = "".join(f'<td style="padding:4px 8px;">{metrics.get(c, "N/A")}</td>' for c in quast_cols)
            links_cell = (
                f'<td style="padding:4px 8px; white-space:nowrap;">'
                f'<a href="#" data-ws-path="{ws_report}" class="qc-ws-link" target="_blank">'
                f'View genome quality report</a>'
                f'<br><small><a href="#" data-ws-path="{ws_dir}" class="qc-ws-link" target="_blank">'
                f'Additional assembly details available</a></small>'
                f'</td>'
            )
            rows_html += f'<tr><td style="padding:4px 8px;"><code>{genome_id}</code></td>{cells}{links_cell}</tr>'
        if rows_html:
            quast_html = f"""
    <h5 style="margin-top:12px; color:#856404;">Assembly Quality (QUAST)</h5>
    <div style="overflow-x:auto;">
      <table style="border-collapse:collapse; font-size:0.9em; width:100%;">
        <thead>
          <tr style="background:#ffeeba; text-align:left;">
            <th style="padding:4px 8px;">Genome ID</th>
            {header_cells}
            <th style="padding:4px 8px;">Links</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    <script>
      document.querySelectorAll('.qc-ws-link').forEach(function(el) {{
        el.href = window.location.origin + '/workspace' + el.getAttribute('data-ws-path');
      }});
    </script>"""

    return f"""
  <div style="background:#fff3cd; border-left:5px solid #ffc107; border-radius:4px;
              padding:16px 20px; margin-top:16px;">
    <h4 style="margin-top:0; color:#856404;">Low Allele Call Coverage Detected</h4>
    <p>
      The following genome(s) have fewer than 75% exact or inferred allele matches.
      A high proportion of non-exact allele calls can reduce the reliability of pairwise
      distance calculations and may lead to inaccurate clustering results. We recommend
      reviewing assembly quality and consider rerunning the analysis without the following
      genomes:
    </p>
    <ul>{genome_list_html}</ul>
    {quast_html}
    {genome_group_html}
  </div>
"""


def create_summary_table(genome_ids, loci_ids, coverage_df, n_cgmlst_loci):
    """Build a small summary DataFrame for the report header table.

    Parameters
    ----------
    genome_ids : list of str
    loci_ids : list of str
    coverage_df : pd.DataFrame
    n_cgmlst_loci : int
        Number of loci in the cgMLST-100%.

    Returns
    -------
    table_html : str
    """
    total_exact = int(coverage_df["Exact"].sum() + coverage_df.get("INF", 0).sum())
    total_cells = len(genome_ids) * len(loci_ids)
    pct_exact = int(round(100 * total_exact / total_cells)) if total_cells > 0 else 0

    summary = pd.DataFrame([{
        "Number of Genomes": len(genome_ids),
        "Number of Loci in Schema": len(loci_ids),
        "cgMLST Loci (100%)": n_cgmlst_loci,
        "Total Loci Exact": total_exact,
        "Total Possible Calls": total_cells,
        "Percent Exact (%)": pct_exact,
    }])

    return generate_table_html_2(summary, table_width="75%")


def build_heatmap_html(genome_ids, dist_matrix, metadata_json_string):
    """Build the interactive heatmap HTML block.

    Parameters
    ----------
    genome_ids : list of str
        Hierarchically clustered genome IDs.
    dist_matrix : list of lists (int)
        Clustered distance matrix.
    metadata_json_string : str
        JSON-serialised metadata (list of dicts).

    Returns
    -------
    heatmap_html : str
    """
    genome_ids_json = json.dumps(genome_ids)
    dist_matrix_json = json.dumps(dist_matrix)

    heatmap_html = """
    <h2>Allelic Distance Analysis</h2>
    <p>
      The tabs below offer different views that visualize differences in allele assignments
      between genomes. The values represent the number of loci that differ between two genomes.
      Greater values indicate the genomes share fewer alleles. A distance of 0 means the two
      genomes are identical at all compared loci. The Distance threshold set above applies to
      all three tabs.
    </p>

    <!-- Shared controls (visible in both views) -->
    <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin-bottom:12px;">
      <div class="linkage-controls" style="display:flex; gap:10px; align-items:center;">
        <label style="font-weight:bold;">Distance threshold:
          <input type="number" id="linkageThreshold" min="0" style="width:60px; margin-left:6px;">
        </label>
      </div>
    </div>

    <!-- View toggle tabs -->
    <div style="display:flex; gap:0; margin-bottom:16px; border-bottom:2px solid #ccc;">
      <button id="tabHeatmap"
        onclick="switchView('heatmap')"
        style="padding:8px 20px; cursor:pointer; border:1px solid #ccc; border-bottom:none;
               background:#fff; font-size:14px; font-weight:bold; border-radius:4px 4px 0 0;
               margin-bottom:-2px; border-bottom:2px solid #fff;">
        Heatmap
      </button>
      <button id="tabClosePairs"
        onclick="switchView('closePairs')"
        style="padding:8px 20px; cursor:pointer; border:1px solid #ccc; border-bottom:none;
               background:#f5f5f5; font-size:14px; color:#555; border-radius:4px 4px 0 0;
               margin-bottom:-2px;">
        Close Genome Pairs
      </button>
      <button id="tabDistTable"
        onclick="switchView('distTable')"
        style="padding:8px 20px; cursor:pointer; border:1px solid #ccc; border-bottom:none;
               background:#f5f5f5; font-size:14px; color:#555; border-radius:4px 4px 0 0;
               margin-bottom:-2px;">
        Distance Matrix Table
      </button>
    </div>

    <!-- Heatmap view -->
    <div id="heatmapViewSection">
      <p>
        The Allelic Distance Heatmap view is a square matrix comparing all genomes in the input
        genome group. A line of zeros goes across the heatmap where the same genome is compared
        to itself. Interact with the heatmap by clicking any cell to compare the metadata for
        that pair of genomes in the panel below.
      </p>
      <div class="heatmap-controls">
        <h4>Filter and Sort the Data:</h4>
        <label>Reorder Heatmap by Metadata:
          <select id="metadataFieldSelect" onchange="recolorHeatmap()">
            <!-- options populated dynamically -->
          </select>
        </label>
        <label style="margin-left:16px; font-weight:bold;">
          <input type="checkbox" id="hoverMetaToggle" onchange="recolorHeatmap()">
          Show Metadata on Hover
        </label>
      </div>

      <!-- Heatmap (full width) -->
      <div id="heatmap" style="width:100%;"></div>

      <!-- Genome comparison panel (appears on cell click) -->
      <div id="comparisonPanel" style="display:none; margin-top:18px; border:1px solid #ccc;
           border-radius:6px; padding:16px; background:#fafafa;">
        <h3 id="comparisonTitle" style="margin-top:0;"></h3>
        <div style="display:flex; gap:24px; flex-wrap:wrap;">
          <div style="flex:1; min-width:220px;">
            <h4 id="genome1Label" style="margin-bottom:6px;"></h4>
            <table id="meta1Table" style="width:100%; border-collapse:collapse;">
            </table>
          </div>
          <div style="flex:1; min-width:220px;">
            <h4 id="genome2Label" style="margin-bottom:6px;"></h4>
            <table id="meta2Table" style="width:100%; border-collapse:collapse;">
            </table>
          </div>
        </div>
      </div>

    </div>

    <script>
      // ===== Embedded data =====
      const genomeLabels = {genome_ids_json};
      const distMatrix   = {dist_matrix_json};
      const metadata     = {metadata_json_string};

      // Build a lookup once: genome_id -> metadata object
      const idToMeta = {{}};
      metadata.forEach(obj => {{ idToMeta[obj.genome_id] = obj; }});

      // ===== Populate metadata field dropdown =====
      (function populateMetadataFields() {{
        const select  = document.getElementById('metadataFieldSelect');
        const allKeys = Object.keys(metadata[0]).filter(k => k !== 'id');

        const defaultOpt       = document.createElement('option');
        defaultOpt.value       = '';
        defaultOpt.textContent = 'Hierarchical Clustering';
        select.appendChild(defaultOpt);

        allKeys.forEach(field => {{
          const opt       = document.createElement('option');
          opt.value       = field;
          opt.textContent = field;
          select.appendChild(opt);
        }});
      }})();

      // ===== Reorder matrix by metadata field =====
      function reorderByField(fieldName, labelsArr, matrixArr) {{
        const arr = metadata.map(obj => ({{ id: obj.genome_id, value: obj[fieldName] }}));
        arr.sort((a, b) => {{
          if (a.value < b.value) return -1;
          if (a.value > b.value) return  1;
          if (a.id   < b.id)    return -1;
          if (a.id   > b.id)    return  1;
          return 0;
        }});
        const newLabels = arr.map(o => o.id);
        const indexMap  = {{}};
        labelsArr.forEach((lbl, idx) => {{ indexMap[lbl] = idx; }});
        const n = labelsArr.length;
        const newMatrix = [];
        for (let i = 0; i < n; i++) {{
          const origRow = indexMap[newLabels[i]];
          const row = [];
          for (let j = 0; j < n; j++) {{
            row.push(matrixArr[origRow][indexMap[newLabels[j]]]);
          }}
          newMatrix.push(row);
        }}
        return {{ newLabels, newMatrix }};
      }}

      // ===== Render a metadata table into a <table> element =====
      function renderMetaTable(tableEl, metaObj) {{
        tableEl.innerHTML = '';
        for (const [field, val] of Object.entries(metaObj)) {{
          const tr  = document.createElement('tr');
          const tdK = document.createElement('td');
          const tdV = document.createElement('td');
          tdK.style.cssText = 'padding:4px 8px 4px 0; font-weight:bold; white-space:nowrap; vertical-align:top;';
          tdV.style.cssText = 'padding:4px 0; vertical-align:top;';
          tdK.textContent = field;
          tdV.textContent = val;
          tr.appendChild(tdK);
          tr.appendChild(tdV);
          tableEl.appendChild(tr);
        }}
      }}

      // ===== Handle heatmap cell click -> show comparison panel =====
      function onHeatmapClick(eventData) {{
        if (!eventData || !eventData.points || eventData.points.length === 0) return;
        const pt   = eventData.points[0];
        const id1  = pt.y;
        const id2  = pt.x;
        const dist = distMatrix[genomeLabels.indexOf(id1)][genomeLabels.indexOf(id2)];

        const meta1 = idToMeta[id1] || {{ genome_id: id1 }};
        const meta2 = idToMeta[id2] || {{ genome_id: id2 }};

        document.getElementById('comparisonTitle').textContent =
          `Allelic Distance: ${{dist}} loci`;
        document.getElementById('genome1Label').innerHTML = `<a href="${{window.location.origin}}/view/Genome/${{id1}}" target="_blank">${{id1}}</a>`;
        document.getElementById('genome2Label').innerHTML = `<a href="${{window.location.origin}}/view/Genome/${{id2}}" target="_blank">${{id2}}</a>`;

        renderMetaTable(document.getElementById('meta1Table'), meta1);
        renderMetaTable(document.getElementById('meta2Table'), meta2);

        const panel = document.getElementById('comparisonPanel');
        panel.style.display = 'block';
        panel.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
      }}

      // ===== Linkage threshold: live-update all three sections =====
      document.addEventListener('DOMContentLoaded', function () {{
        const ltEl = document.getElementById('linkageThreshold');
        if (ltEl) ltEl.addEventListener('input', function () {{
          recolorHeatmap();
          if (typeof buildClosePairs === 'function') buildClosePairs();
          if (typeof buildDistTable  === 'function') buildDistTable();
        }});
      }});

      // ===== Threshold binning helpers =====
      function assignBin(val, t) {{
        if (val === 0) return 0;
        if (t !== null && val <= t) return 1;
        return 2;
      }}

      // Discrete 3-bin colorscale matching getDmColor: identical / within / above
      function getThresholdColorscale() {{
        return [
          [0,       '#2ca02c'],
          [1/3,     '#2ca02c'],
          [1/3,     '#a8d5a2'],
          [2/3,     '#a8d5a2'],
          [2/3,     '#f4a460'],
          [1.0,     '#f4a460']
        ];
      }}

      // ===== Draw / redraw heatmap =====
      function recolorHeatmap() {{
        const tRaw      = document.getElementById('linkageThreshold').value.trim();
        const t         = tRaw !== '' ? parseInt(tRaw) : null;
        const metaField = document.getElementById('metadataFieldSelect').value;

        let labels = genomeLabels.slice();
        let matrix = distMatrix.map(row => row.slice());

        if (metaField) {{
          const reordered = reorderByField(metaField, labels, matrix);
          labels = reordered.newLabels;
          matrix = reordered.newMatrix;
        }}

        const showHoverMeta = document.getElementById('hoverMetaToggle').checked;
        const hoverText = matrix.map((row, i) =>
          row.map((val, j) => {{
            let text = `${{labels[i]}} vs ${{labels[j]}}<br>Allelic Distance: ${{val}}`;
            if (showHoverMeta) {{
              const meta1 = idToMeta[labels[i]] || {{ genome_id: labels[i] }};
              const meta2 = idToMeta[labels[j]] || {{ genome_id: labels[j] }};
              text += `<br><br><b>Genome 1:</b> ${{labels[i]}}<br>`;
              for (const [field, fval] of Object.entries(meta1)) {{
                text += `${{field}}: ${{fval}}<br>`;
              }}
              text += `<br><b>Genome 2:</b> ${{labels[j]}}<br>`;
              for (const [field, fval] of Object.entries(meta2)) {{
                text += `${{field}}: ${{fval}}<br>`;
              }}
            }} else {{
              text += `<br><i>Click for full metadata</i>`;
            }}
            return text;
          }})
        );

        let zData, colorscale, colorbarConfig, extraRange;
        if (t !== null) {{
          zData        = matrix.map(row => row.map(val => assignBin(val, t)));
          colorscale   = getThresholdColorscale();
          extraRange   = {{ zmin: 0, zmax: 2 }};
          colorbarConfig = {{
            tickvals: [0, 1, 2],
            ticktext: ['Identical (0)', `Matching (\u2264${{t}})`, `Above (>${{t}})`],
            title:    'Distance'
          }};
        }} else {{
          zData        = matrix;
          colorscale   = 'Viridis';
          extraRange   = {{}};
          colorbarConfig = {{ title: 'Allelic Distance' }};
        }}

        const traceData = [Object.assign({{
          z:          zData,
          x:          labels,
          y:          labels,
          type:       'heatmap',
          colorscale: colorscale,
          text:       hoverText,
          hoverinfo:  'text',
          colorbar:   colorbarConfig
        }}, extraRange)];

        // Scale height to number of samples: 40px/sample, min 500, max 1800.
        const n = labels.length;
        const heatmapHeight = Math.max(500, Math.min(1800, n * 40 + 150));

        const titleStr = t !== null
          ? `Allelic Distance Heatmap (threshold: ${{t}})` +
            (metaField ? ` \u2013 Reordered by "${{metaField}}"` : '')
          : 'Allelic Distance Heatmap' +
            (metaField ? ` \u2013 Reordered by "${{metaField}}"` : '');

        const layout = {{
          title: titleStr,
          height: heatmapHeight,
          width: heatmapHeight,
          xaxis: {{ type: 'category', tickangle: 45 }},
          yaxis: {{ type: 'category', tickangle: 45 }}
        }};

        const heatmapDiv = document.getElementById('heatmap');
        Plotly.newPlot(heatmapDiv, traceData, layout);
        heatmapDiv.on('plotly_click', onHeatmapClick);
      }}

      // ===== Tab switching =====
      function switchView(view) {{
        const sections = {{
          heatmap:    document.getElementById('heatmapViewSection'),
          closePairs: document.getElementById('closePairsViewSection'),
          distTable:  document.getElementById('distanceTableViewSection'),
        }};
        const tabs = {{
          heatmap:    document.getElementById('tabHeatmap'),
          closePairs: document.getElementById('tabClosePairs'),
          distTable:  document.getElementById('tabDistTable'),
        }};

        // Hide all sections, reset all tabs
        Object.keys(sections).forEach(k => {{
          if (sections[k]) sections[k].style.display = 'none';
        }});
        Object.keys(tabs).forEach(k => {{
          if (tabs[k]) {{
            tabs[k].style.background   = '#f5f5f5';
            tabs[k].style.color        = '#555';
            tabs[k].style.borderBottom = '';
            tabs[k].style.fontWeight   = '';
          }}
        }});

        // Show active section, highlight active tab
        if (sections[view]) sections[view].style.display = '';
        if (tabs[view]) {{
          tabs[view].style.background   = '#fff';
          tabs[view].style.color        = '';
          tabs[view].style.borderBottom = '2px solid #fff';
          tabs[view].style.fontWeight   = 'bold';
        }}

        // Trigger data build on first switch to each section
        if (view === 'closePairs' && typeof buildClosePairs === 'function') buildClosePairs();
        if (view === 'distTable'  && typeof buildDistTable  === 'function') buildDistTable();
      }}

      // Initial render
      recolorHeatmap();
    </script>
    """.format(
        genome_ids_json=genome_ids_json,
        dist_matrix_json=dist_matrix_json,
        metadata_json_string=metadata_json_string,
    )

    return heatmap_html


def build_distance_analysis_html():
    """Build interactive Close Pairs and Distance Matrix Table HTML blocks.

    Reuses the ``genomeLabels`` and ``distMatrix`` JavaScript globals
    already embedded by :func:`build_heatmap_html`.  Must appear in the
    HTML *after* the heatmap section.

    Returns
    -------
    html : str
    """
    return """
    <!-- ================================================================== -->
    <!-- CLOSE PAIRS                                                         -->
    <!-- ================================================================== -->
    <div id="closePairsViewSection" style="display:none;">
    <h2>Close Genome Pairs</h2>
    <p>
      This view lists each unique genome pair sorted by allelic distance, making it easy to
      identify the most closely related samples. Only the upper triangle of the distance matrix
      is shown (each pair appears once). Refine your results by changing the distance threshold
      above. Click any column header to sort.
    </p>
    <div style="margin-bottom:12px;">
      <span id="cpCount" style="color:#555; font-size:13px;"></span>
    </div>
    <div style="overflow:auto; max-height:400px; border:1px solid #ccc; border-radius:4px;">
      <table id="cpTable" style="border-collapse:collapse; width:100%;">
        <thead>
          <tr style="position:sticky; top:0; background:#f0f0f0; z-index:2;">
            <th onclick="sortClosePairs('g1')"
                style="padding:6px 12px; border:1px solid #ddd; cursor:pointer; user-select:none;">
              Genome A <span id="cpSort_g1"></span></th>
            <th onclick="sortClosePairs('g2')"
                style="padding:6px 12px; border:1px solid #ddd; cursor:pointer; user-select:none;">
              Genome B <span id="cpSort_g2"></span></th>
            <th onclick="sortClosePairs('dist')"
                style="padding:6px 12px; border:1px solid #ddd; cursor:pointer; user-select:none;">
              Distance <span id="cpSort_dist">&#9650;</span></th>
          </tr>
        </thead>
        <tbody id="cpBody"></tbody>
      </table>
    </div>

    </div><!-- end closePairsViewSection -->

    <!-- ================================================================== -->
    <!-- FULL DISTANCE MATRIX TABLE                                          -->
    <!-- ================================================================== -->
    <div id="distanceTableViewSection" style="display:none;">
    <h2>Distance Matrix Table</h2>
    <p>
      The table below presents the full pairwise allelic distance matrix in a searchable,
      color-coded format. Each cell shows the number of alleles that differ between two genomes.
      Use the <strong>search box</strong> to filter rows by genome ID, or set a distance
      threshold above to highlight cells at or below that value.
    </p>
    <div style="display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-bottom:10px;">
      <label style="display:flex; align-items:center; gap:6px;">
        Search genome ID:
        <input type="text" id="dmSearch" placeholder="Filter rows by ID..."
               style="padding:4px 8px; width:200px; font-size:13px;"
               oninput="buildDistTable()">
      </label>
      <span id="dmRowCount" style="color:#555; font-size:13px;"></span>
      <button onclick="exportDistTableToExcel()"
              style="padding:5px 14px; font-size:13px; cursor:pointer;
                     background:#217346; color:white; border:none;
                     border-radius:4px; margin-left:auto;">
        &#8681; Export to Excel
      </button>
    </div>
    <!-- Color legend (labels updated dynamically by buildDistTable) -->
    <div style="display:flex; gap:16px; align-items:center; margin-bottom:10px; font-size:12px; flex-wrap:wrap;">
      <strong>Color key:</strong>
      <span id="dmKeyIdentical" style="background:#2ca02c; color:white; padding:2px 8px; border-radius:3px;">Identical (0)</span>
      <span id="dmKeyMatching" style="background:#a8d5a2; color:#333; padding:2px 8px; border-radius:3px;">Matching threshold</span>
      <span id="dmKeyAbove"    style="background:#f4a460; color:#333; padding:2px 8px; border-radius:3px;">Above threshold</span>
    </div>
    <div id="dmTableWrapper"
         style="overflow:auto; max-height:600px; border:1px solid #ccc; border-radius:4px;">
      <table id="dmTable" style="border-collapse:collapse; white-space:nowrap;"></table>
    </div>
    </div><!-- end distanceTableViewSection -->

    <script>
      // ===== Shared color helper =====
      function getDmColor(val, t) {
        if (val === 0) return { bg: '#2ca02c', fg: 'white' };
        if (t !== null && val <= t) return { bg: '#a8d5a2', fg: '#333' };
        return { bg: t !== null ? '#f4a460' : '#ffffff', fg: '#333' };
      }

      function getDmThresholds() {
        const el  = document.getElementById('linkageThreshold');
        const raw = el ? el.value.trim() : '';
        return { t: raw !== '' ? parseInt(raw) : null };
      }

      // ===== Close Pairs =====
      let _cpPairs     = [];
      let _cpSortField = 'dist';
      let _cpSortAsc   = true;

      function buildClosePairs() {
        const { t } = getDmThresholds();
        const labels  = genomeLabels;
        const matrix  = distMatrix;
        const n       = labels.length;

        _cpPairs = [];
        for (let i = 0; i < n; i++) {
          for (let j = i + 1; j < n; j++) {
            const d = matrix[i][j];
            if (t === null || d <= t) {
              _cpPairs.push({ g1: labels[i], g2: labels[j], dist: d });
            }
          }
        }
        _renderClosePairs(t);
      }

      function sortClosePairs(field) {
        if (_cpSortField === field) {
          _cpSortAsc = !_cpSortAsc;
        } else {
          _cpSortField = field;
          _cpSortAsc   = (field === 'dist');
        }
        const { t } = getDmThresholds();
        _renderClosePairs(t);
      }

      function _renderClosePairs(t) {
        ['g1', 'g2', 'dist'].forEach(f => {
          const el = document.getElementById('cpSort_' + f);
          if (el) el.textContent = '';
        });
        const sortEl = document.getElementById('cpSort_' + _cpSortField);
        if (sortEl) sortEl.textContent = _cpSortAsc ? ' \u25b2' : ' \u25bc';

        const sorted = _cpPairs.slice().sort((a, b) => {
          const av = a[_cpSortField], bv = b[_cpSortField];
          const cmp = av < bv ? -1 : av > bv ? 1 : 0;
          return _cpSortAsc ? cmp : -cmp;
        });

        const tbody = document.getElementById('cpBody');
        tbody.innerHTML = '';

        if (sorted.length === 0) {
          const tr = document.createElement('tr');
          const td = document.createElement('td');
          td.colSpan = 3;
          td.style.cssText = 'padding:14px; text-align:center; color:#666; font-style:italic;';
          td.textContent = 'No pairs found within the specified distance threshold. Try increasing the threshold.';
          tr.appendChild(td);
          tbody.appendChild(tr);
        } else {
          sorted.forEach(pair => {
            const { bg, fg } = getDmColor(pair.dist, t);
            const tr  = document.createElement('tr');
            const tdg1 = document.createElement('td');
            const tdg2 = document.createElement('td');
            const tdd  = document.createElement('td');
            tdg1.style.cssText = 'padding:4px 12px; border:1px solid #ddd; font-size:12px;';
            tdg2.style.cssText = 'padding:4px 12px; border:1px solid #ddd; font-size:12px;';
            tdd.style.cssText  = 'padding:4px 12px; border:1px solid #ddd; font-size:12px; text-align:center; font-weight:bold; background:' + bg + '; color:' + fg + ';';
            tdg1.textContent = pair.g1;
            tdg2.textContent = pair.g2;
            tdd.textContent  = pair.dist;
            tr.appendChild(tdg1); tr.appendChild(tdg2); tr.appendChild(tdd);
            tbody.appendChild(tr);
          });
        }

        document.getElementById('cpCount').textContent =
          sorted.length === 1 ? '1 pair found' : sorted.length + ' pairs found';
      }

      // ===== Full Distance Matrix Table =====
      function buildDistTable() {
        const searchVal = document.getElementById('dmSearch').value.trim().toLowerCase();
        const { t } = getDmThresholds();

        const labels = genomeLabels;
        const matrix = distMatrix;
        const n      = labels.length;

        // Update color key labels to reflect the current threshold
        const kmEl = document.getElementById('dmKeyMatching');
        const kaEl = document.getElementById('dmKeyAbove');
        if (kmEl) kmEl.textContent = t !== null ? `Matching threshold (\u2264${t})` : 'Matching threshold';
        if (kaEl) kaEl.textContent = t !== null ? `Above threshold (>${t})`         : 'Above threshold';

        // Filter rows by search only; all rows shown regardless of threshold
        const visibleRows = [];
        labels.forEach((lbl, i) => {
          if (!searchVal || lbl.toLowerCase().includes(searchVal)) visibleRows.push(i);
        });

        const table    = document.getElementById('dmTable');
        const fragment = document.createDocumentFragment();

        const thead  = document.createElement('thead');
        const hRow   = document.createElement('tr');
        const corner = document.createElement('th');
        corner.style.cssText = 'position:sticky; top:0; left:0; z-index:4; background:#fff; padding:4px 8px; border:1px solid #ddd; min-width:120px;';
        corner.textContent = '';
        hRow.appendChild(corner);
        labels.forEach(lbl => {
          const th = document.createElement('th');
          th.style.cssText = 'position:sticky; top:0; z-index:2; background:#f8f8f8; border:1px solid #ddd; padding:2px; font-size:10px; font-weight:normal; writing-mode:vertical-rl; transform:rotate(180deg); height:110px; vertical-align:bottom; text-align:left;';
          th.textContent = lbl;
          hRow.appendChild(th);
        });
        thead.appendChild(hRow);
        fragment.appendChild(thead);

        const tbody = document.createElement('tbody');
        visibleRows.forEach(i => {
          const tr    = document.createElement('tr');
          const rowTh = document.createElement('th');
          rowTh.style.cssText = 'position:sticky; left:0; z-index:1; background:#f8f8f8; border:1px solid #ddd; padding:3px 8px; font-size:11px; font-weight:normal; white-space:nowrap; text-align:left;';
          rowTh.textContent = labels[i];
          tr.appendChild(rowTh);

          labels.forEach((lbl, j) => {
            const val        = matrix[i][j];
            const { bg, fg } = getDmColor(val, t);
            const td         = document.createElement('td');
            td.style.cssText = 'background:' + bg + '; color:' + fg + '; padding:3px 5px; border:1px solid #ddd; text-align:center; font-size:11px; min-width:28px;';
            td.textContent   = val;
            td.title         = labels[i] + ' vs ' + lbl + ': ' + val + ' allele differences';
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });

        if (visibleRows.length === 0) {
          const emptyTr = document.createElement('tr');
          const emptyTd = document.createElement('td');
          emptyTd.colSpan = n + 1;
          emptyTd.style.cssText = 'padding:16px; text-align:center; color:#666; font-style:italic;';
          emptyTd.textContent   = 'No genomes match the current filters.';
          emptyTr.appendChild(emptyTd);
          tbody.appendChild(emptyTr);
        }
        fragment.appendChild(tbody);

        table.innerHTML = '';
        table.appendChild(fragment);

        document.getElementById('dmRowCount').textContent =
          visibleRows.length === n
            ? 'Showing all ' + n + ' genomes'
            : 'Showing ' + visibleRows.length + ' of ' + n + ' genomes';
      }

      // ===== Excel export for Distance Matrix Table =====
      function colorToRgb(color) {
        if (color === 'white') return 'FFFFFF';
        const hex = color.replace('#', '');
        return hex.length === 3
          ? hex.split('').map(c => c + c).join('').toUpperCase()
          : hex.toUpperCase();
      }

      function exportDistTableToExcel() {
        if (typeof XLSX === 'undefined') {
          alert('Excel export library not loaded. Please check your internet connection.');
          return;
        }
        const searchVal = document.getElementById('dmSearch')
          ? document.getElementById('dmSearch').value.trim().toLowerCase() : '';
        const { t } = getDmThresholds();
        const labels = genomeLabels;
        const matrix = distMatrix;

        const visibleRows = [];
        labels.forEach((lbl, i) => {
          if (!searchVal || lbl.toLowerCase().includes(searchVal)) visibleRows.push(i);
        });

        const wsData = [];
        const headerRow = [{ v: '', s: { font: { bold: true }, fill: { patternType: 'solid', fgColor: { rgb: 'E0E0E0' } } } }];
        labels.forEach(lbl => {
          headerRow.push({
            v: lbl,
            s: {
              font: { bold: true },
              fill: { patternType: 'solid', fgColor: { rgb: 'E0E0E0' } },
              alignment: { textRotation: 90, horizontal: 'center', vertical: 'bottom' }
            }
          });
        });
        wsData.push(headerRow);

        visibleRows.forEach(i => {
          const row = [{
            v: labels[i],
            s: { font: { bold: true }, fill: { patternType: 'solid', fgColor: { rgb: 'E0E0E0' } } }
          }];
          labels.forEach((lbl, j) => {
            const val = matrix[i][j];
            const { bg, fg } = getDmColor(val, t);
            row.push({
              v: val,
              t: 'n',
              s: {
                fill: { patternType: 'solid', fgColor: { rgb: colorToRgb(bg) } },
                font: { color: { rgb: colorToRgb(fg) } },
                alignment: { horizontal: 'center' }
              }
            });
          });
          wsData.push(row);
        });

        const ws = XLSX.utils.aoa_to_sheet(wsData);
        ws['!cols'] = [{ wch: 22 }].concat(labels.map(() => ({ wch: 10 })));
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, 'Distance Matrix');
        XLSX.writeFile(wb, 'cgmlst_distance_matrix.xlsx');
      }

      // ===== Initialise on load + sync with heatmap threshold =====
      document.addEventListener('DOMContentLoaded', function () {
        buildClosePairs();
        buildDistTable();

        const ltIn = document.getElementById('linkageThreshold');
        if (ltIn) ltIn.addEventListener('input', function () { buildClosePairs(); buildDistTable(); });
      });
    </script>
    """


def define_html_template(summary_table_html, barplot_html,
                          heatmap_html, distance_analysis_html, tree_html,
                          metadata_json_string,
                          n_genomes, n_loci, low_coverage_warning_html=""):
    """Assemble the complete cgMLST report HTML.

    Parameters
    ----------
    summary_table_html : str
    barplot_html : str
    heatmap_html : str
    metadata_json_string : str
    n_genomes : int
    n_loci : int

    Returns
    -------
    html : str
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Core Genome MLST Report</title>
  <style>
    body {{
      font-family: Roboto, sans-serif;
      color: black;
      margin: 0 auto;
      max-width: 1400px;
      padding: 0 20px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 20px;
    }}
    .title {{
      font-size: 36px;
      font-family: 'Roboto', sans-serif;
      font-weight: bold;
      color: black;
    }}
    .plot-container {{
      display: block;
      text-align: center;
    }}
    .plot {{
      width: 90%;
      margin: 0 auto;
    }}
    .heatmap-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 1em;
    }}
    .heatmap-controls h4 {{
      flex-basis: 100%;
      margin: 0;
    }}
    table, th, td {{
      border: 1px solid black;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 5px;
      text-align: left;
    }}
    table {{
      width: 100%;
    }}
    th input {{
      width: 100%;
      box-sizing: border-box;
    }}
    #heatmap {{
      width: 100%;
    }}
  </style>

  <!-- Plotly.js v3.0.1 -->
  <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
  <!-- xlsx-js-style for Excel export with cell colours -->
  <script src="https://cdn.jsdelivr.net/npm/xlsx-js-style@1.2.0/dist/xlsx.bundle.js"></script>
  <!-- DataTables CSS -->
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">
</head>

<body>
  <header>
    <div class="title">Core Genome MLST Report</div>
  </header>

  <!-- ================================================================== -->
  <!-- ABOUT THE TOOL                                                       -->
  <!-- ================================================================== -->
  <h2>About the Analysis</h2>
  <p>
    Explore the results of your Core Genome Multi-Locus Sequence Typing (cgMLST) analysis in
    this interactive report. An allele is a specific sequence variant that occurs at a given
    locus. The aim of this service is to characterize bacteria and viruses based on the presence
    and absence of specific loci. These loci are defined in a schema.
  </p>
  <p>
    Typical Multi-Locus Sequence Typing (MLST) schemas use 5&ndash;7 genes that are conserved
    and essential for basic cellular functions and are expected to be present in all strains of
    a species even as they evolve. This service uses schemas with predefined sets of
    species-specific loci including both core and accessory genes. This allows for finer
    resolution of bacterial strain differences and consistent tracking across labs. There are 32
    schemas for priority pathogen species curated by species experts at
    <a href="https://www.ridom.de/seqsphere/cgmlst/" target="_blank">Ridom</a>.
  </p>
  <p>
    A core genome is composed of the genes shared amongst all genomes in the group.
  </p>

  <h3>About the Analysis Workflow</h3>
  <p>
    The analysis begins with the <strong>chewBBACA 3.3.10 AlleleCall</strong> command. This
    reviews the input assembled genome sequences for Coding DNA sequences (CDSs) and open
    reading frames (ORFs). To be considered, a CDS must be classified as such by
    <a href="https://github.com/althonos/pyrodigal" target="_blank">Pyrodigal</a>. These
    coding regions are then compared to the known alleles. If the Blast Score Ratio (BSR) of a
    known allele is equal to or greater than 0.6 it is considered an Exact Match. Note: matches
    are made at the gene level. If a sequence does not match any of the known alleles it is
    considered a new allele candidate and stored as INF-###. These alleles are not added to our
    schemas (chewBBACA does allow for this functionality). The allele calling results are used
    throughout the rest of the pipeline.
  </p>
  <p>The AlleleCall module assigns one of the following classification codes to each locus:</p>
  <div style="background:#e8f4f8; border-left:5px solid #2196f3; border-radius:4px;
              padding:16px 20px; margin-top:8px;">
    <dl style="margin:0; display:grid; grid-template-columns:max-content 1fr; gap:6px 20px;
               align-items:baseline;">
      <dt style="font-weight:bold; white-space:nowrap;">EXC</dt>
      <dd style="margin:0;">Exact Locus Match — an exact allele match was found.</dd>

      <dt style="font-weight:bold; white-space:nowrap;">INF</dt>
      <dd style="margin:0;">Inferred New Allele — a valid but previously unobserved allele sequence.</dd>

      <dt style="font-weight:bold; white-space:nowrap;">LNF</dt>
      <dd style="margin:0;">Locus Not Found — no match detected in the genome.</dd>

      <dt style="font-weight:bold; white-space:nowrap;">ASM</dt>
      <dd style="margin:0;">Allele Smaller than Minimum — the matched region is too short.</dd>

      <dt style="font-weight:bold; white-space:nowrap;">ALM</dt>
      <dd style="margin:0;">Allele Larger than Maximum — the matched region is too long.</dd>

      <dt style="font-weight:bold; white-space:nowrap;">NIPH</dt>
      <dd style="margin:0;">Non-Informative Paralogous Hit — multiple matches of equal quality.</dd>

      <dt style="font-weight:bold; white-space:nowrap;">NIPHEM</dt>
      <dd style="margin:0;">Non-Informative Paralogous Hit with Exact Match — multiple exact matches.</dd>

      <dt style="font-weight:bold; white-space:nowrap;">PLOT3 / PLOT5</dt>
      <dd style="margin:0;">Possible Locus on Tip of Contig — the locus may lie at the 3′ or 5′ end of a contig.</dd>
    </dl>
  </div>

  <p>
    A step called <strong>Extract Core Loci</strong> reviews which alleles are shared across
    all genomes. The detailed results are not reflected in this report but are available in the
    extract cgMLST directory, which contains summary statistics evaluating results per sample
    and per locus.
  </p>
  <p>
    <a href="https://github.com/zheminzhou/pHierCC/tree/master" target="_blank">pHierCC</a>
    applies an unsupervised machine learning algorithm that organizes genomes into a
    hierarchical tree of nested clusters. Cluster assignments are informed by a representative
    set of precomputed clustering data for the same species.
  </p>
  <p>
    <a href="https://github.com/achtman-lab/GrapeTree" target="_blank">GrapeTree</a>
    generates trees using the MSTree V2 algorithm to construct a minimum spanning tree. A
    snapshot of the tree is included in this report. The job results include the same tree as
    both a newick file and a phyloxml file.
  </p>


  <!-- ================================================================== -->
  <!-- INPUT DATA SUMMARY                                                   -->
  <!-- ================================================================== -->
  <h2>Getting to Know the Input Data</h2>
  <h3>Input Genomes and Schema</h3>
  <p>
    This analysis includes <strong>{n_genomes}</strong> genomes typed against
    a schema of <strong>{n_loci}</strong> loci.
  </p>
  {summary_table_html}

  <h3>Loci Coverage per Sample</h3>
  <p>
    The chart below shows the breakdown of allele call results for each sample. Hover over each
    genome to view loci assignments. Majority classifications can overwhelm the barplot &mdash;
    hide a given subset by clicking on the square in the Classification key.
    <b>Exact</b> (green) loci received a numeric allele assignment.
    <b>INF</b> (teal) loci were assigned a newly inferred allele. All other colors represent
    missing or problematic calls that do not contribute to the distance calculation. Samples
    with a high proportion of missing loci may warrant quality review.
  </p>
  <div class="plot-container">
    <div class="plot" id="lociCoveragePlot">
      {barplot_html}
    </div>
  </div>
  {low_coverage_warning_html}

  <!-- ================================================================== -->
  <!-- METADATA                                                             -->
  <!-- ================================================================== -->
  <h3>Input Genome Metadata</h3>
  <table id="dataTable" class="display" style="width:100%">
    <thead id="tableHead"></thead>
    <tbody id="tableBody"></tbody>
  </table>

  <!-- Embedded JSON metadata -->
  <script>
    const tableData = {metadata_json_string};
  </script>

  <!-- jQuery + DataTables -->
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
  <script>
    function populateTable(data) {{
      const tableHead = document.getElementById('tableHead');
      const tableBody = document.getElementById('tableBody');
      tableHead.innerHTML = '';
      tableBody.innerHTML = '';

      const headers   = Object.keys(data[0]);
      const headRow   = document.createElement('tr');
      const filterRow = document.createElement('tr');

      headers.forEach((header, index) => {{
        const th = document.createElement('th');
        th.textContent = header;
        headRow.appendChild(th);

        const filterCell  = document.createElement('th');
        const filterInput = document.createElement('input');
        filterInput.type        = 'text';
        filterInput.placeholder = `Filter ${{header}}`;
        filterInput.dataset.column = index;
        filterInput.addEventListener('keyup', function () {{
          const colIdx      = parseInt(this.dataset.column);
          const filterValue = this.value.trim();
          const isNumeric   = data.every(row => !isNaN(parseFloat(row[header])));
          if (filterValue) {{
            if (isNumeric) {{
              const parsed = parseFloat(filterValue);
              if (!isNaN(parsed)) {{
                $.fn.dataTable.ext.search = [];
                $.fn.dataTable.ext.search.push((settings, row) =>
                  parseFloat(row[colIdx]) >= parsed
                );
              }}
            }} else {{
              $.fn.dataTable.ext.search = [];
              $.fn.dataTable.ext.search.push((settings, row) =>
                row[colIdx].toLowerCase().includes(filterValue.toLowerCase())
              );
            }}
          }} else {{
            $.fn.dataTable.ext.search = [];
          }}
          $('#dataTable').DataTable().draw();
        }});
        filterCell.appendChild(filterInput);
        filterRow.appendChild(filterCell);
      }});

      tableHead.appendChild(headRow);
      tableHead.appendChild(filterRow);

      data.forEach(row => {{
        const tr = document.createElement('tr');
        headers.forEach(header => {{
          const td = document.createElement('td');
          if (header === 'genome_id') {{
            td.innerHTML = `<a href="${{window.location.origin}}/view/Genome/${{row[header]}}" target="_blank">${{row[header]}}</a>`;
          }} else {{
            td.innerHTML = row[header];
          }}
          tr.appendChild(td);
        }});
        tableBody.appendChild(tr);
      }});

      $('#dataTable').DataTable({{
        pageLength: 10,
        lengthMenu: [10, 25, 50, 100],
        orderCellsTop: true,
      }});
    }}

    document.addEventListener('DOMContentLoaded', function () {{
      populateTable(tableData);
      document.querySelectorAll('#dataTable a').forEach(link => {{
        link.target = '_blank';
      }});
    }});
  </script>

  <br>

  <!-- ================================================================== -->
  <!-- HEATMAP + TREE                                                       -->
  <!-- ================================================================== -->
  {heatmap_html}

  <!-- ================================================================== -->
  <!-- CLOSE PAIRS + DISTANCE MATRIX TABLE                                 -->
  <!-- ================================================================== -->
  {distance_analysis_html}

  <!-- ================================================================== -->
  <!-- PHYLOGENETIC TREE                                                    -->
  <!-- ================================================================== -->
  {tree_html}

  <!-- ================================================================== -->
  <!-- REFERENCES                                                           -->
  <!-- ================================================================== -->
  <h3>References</h3>
  <ol type="1">
    <li>
    Introducing the Bacterial and Viral Bioinformatics Resource Center (BV-BRC): a resource combining PATRIC, IRD and ViPR.
    Olson RD, Assaf R, Brettin T, Conrad N, Cucinell C, Davis JJ, Dempsey DM, Dickerman A, Dietrich EM, Kenyon RW, Kuscuoglu M, Lefkowitz EJ, Lu J, Machi D, Macken C, Mao C, Niewiadomska A, Nguyen M, Olsen GJ, Overbeek JC, Parrello B, Parrello V, Porter JS, Pusch GD, Shukla M, Singh I, Stewart L, Tan G, Thomas C, VanOeffelen M, Vonstein V, Wallace ZS, Warren AS, Wattam AR, Xia F, Yoo H, Zhang Y, Zmasek CM, Scheuermann RH, Stevens RL.
    Nucleic Acids Res. 2022 Nov 9:gkac1003. doi: 10.1093/nar/gkac1003. Epub ahead of print.
    PMID: <a href="https://pubmed.ncbi.nlm.nih.gov/36350631/" target="_blank">36350631</a>
    DOI: <a href="https://doi.org/10.1093/nar/gkac1003" target="_blank">10.1093/nar/gkac1003</a><br>
    </li>
    <li>
      Silva M, Machado MP, Silva DN, Rossi M, Moran-Gilad J, Santos S,
      Ramirez M, Carrico JA. (2018). chewBBACA: A complete suite for gene-by-gene
      schema creation and strain identification. <em>Microbial Genomics</em>,
      4(3). DOI: <a href="https://doi.org/10.1099/mgen.0.000166" target="_blank">10.1099/mgen.0.000166</a>
    </li>
  </ol>

</body>
</html>""".format(
        n_genomes=n_genomes,
        n_loci=n_loci,
        summary_table_html=summary_table_html,
        barplot_html=barplot_html,
        heatmap_html=heatmap_html,
        distance_analysis_html=distance_analysis_html,
        tree_html=tree_html,
        metadata_json_string=metadata_json_string,
        low_coverage_warning_html=low_coverage_warning_html,
    )

    return html


@cli.command()
@click.argument("result_alleles")
@click.argument("metadata_json")
@click.argument("html_report_path")
@click.option("--svg-dir", default="work", show_default=True,
              help="Directory to search for the tree SVG file (*.svg).")
@click.option("--config", "config_json", default=None,
              help="Path to service config.json (used for tree viewer link and low-coverage QC).")
def write_html_report(result_alleles, metadata_json, html_report_path, svg_dir, config_json):
    """Generate an interactive cgMLST HTML report.

    \b
    Arguments:
      RESULT_ALLELES   Path to chewBBACA result_alleles.tsv
      METADATA_JSON    Path to genome_metadata.json
      HTML_REPORT_PATH Path for the output HTML report
    """
    click.echo("Using result_alleles: {}".format(result_alleles))

    tree_ws_path = ""
    if config_json:
        with open(config_json) as f:
            config = json.load(f)
        params = config.get("params", {})
        ws_output_path = params.get("output_path", "")
        ws_output_file = params.get("output_file", "")
        tree_ws_path = (ws_output_path + "/." + ws_output_file) if ws_output_path and ws_output_file else ""

    # ---- Coverage stats (for bar chart and summary table) ----
    click.echo("Parsing result_alleles.tsv ...")
    genome_ids, loci_ids, raw_df, coverage_df = parse_result_alleles(result_alleles)
    n_genomes = len(genome_ids)
    n_loci = len(loci_ids)
    click.echo("  {} genomes x {} loci".format(n_genomes, n_loci))

    # ---- Distance calculation (chewBBACA-compatible pipeline) ----
    click.echo("Running chewBBACA-compatible distance pipeline ...")
    (dist_genome_ids,
     dist_genome_ids_display,
     cgmlst_loci,
     dist_matrix) = chewbbaca_distance_pipeline(result_alleles)
    n_cgmlst = len(cgmlst_loci)
    click.echo("  cgMLST: {} / {} loci present in all {} samples".format(
        n_cgmlst, n_loci, n_genomes))

    # Use the display IDs (periods) for the report
    ids = dist_genome_ids_display

    output_dir = os.path.dirname(os.path.abspath(html_report_path))
    distance_metrics_dir = os.path.join(output_dir, "distance_metrics")
    os.makedirs(distance_metrics_dir, exist_ok=True)

    click.echo("Writing distance matrix ...")
    dist_matrix_path = os.path.join(distance_metrics_dir, "cgMLST_Report_distance_matrix.tsv")
    dist_df = pd.DataFrame(dist_matrix, index=ids, columns=ids)
    dist_df.index.name = "FILE"
    dist_df.to_csv(dist_matrix_path, sep="\t")
    click.echo("  Distance matrix written to {}".format(dist_matrix_path))

    click.echo("Writing pairwise distance report ...")
    dist_report_path = os.path.join(distance_metrics_dir, "cgMLST_distance.report")
    write_cgmlst_distance_report(ids, dist_matrix, dist_report_path)
    click.echo("  Distance report written to {}".format(dist_report_path))

    click.echo("Clustering for heatmap ...")
    clustered_labels, clustered_matrix = cluster_heatmap_data(ids, dist_matrix)

    click.echo("Building metadata table ...")
    metadata, metadata_df = create_cgmlst_metadata_table(metadata_json, "metadata.tsv")
    metadata_json_string = json.dumps(metadata)

    click.echo("Creating loci coverage bar plot ...")
    barplot_html = create_loci_coverage_plot(coverage_df)

    click.echo("Building summary table ...")
    summary_table_html = create_summary_table(genome_ids, loci_ids, coverage_df,
                                               n_cgmlst_loci=n_cgmlst)

    click.echo("Checking loci coverage thresholds ...")
    low_coverage_warning_html = create_low_coverage_warning(coverage_df, n_loci,
                                                             config_json_path=config_json or "config.json")

    click.echo("Assembling heatmap ...")
    heatmap_html = build_heatmap_html(clustered_labels, clustered_matrix, metadata_json_string)

    click.echo("Building distance analysis tables ...")
    distance_analysis_html = build_distance_analysis_html()

    click.echo("Building tree section ...")
    svg_candidates = sorted(glob.glob(os.path.join(svg_dir, '*.svg')))
    if svg_candidates:
        with open(svg_candidates[0]) as f:
            svg_content = f.read()
    else:
        svg_content = '<p style="color:#555;">Tree SVG not found. Expected at {}/*.svg</p>'.format(svg_dir)
    tree_ws_path_json = json.dumps(tree_ws_path if tree_ws_path else None)
    tree_html = """
    <h2>Phylogenetic Tree</h2>
    <p>
      The tree below is generated by
      <a href="https://github.com/achtman-lab/GrapeTree" target="_blank">GrapeTree</a>
      using the MSTree V2 algorithm to construct a minimum spanning tree. The job results
      include the same tree as both a newick file and a phyloxml file.
      <a id="treeViewerLink" href="#" target="_blank">Click here</a> to view the phyloxml
      trees in our tree viewer to paint metadata onto the tree.
    </p>
    <script>
      (function() {
        var el = document.getElementById('treeViewerLink');
        if (el) {
          var wsPath = TREE_WS_PATH_JSON;
          if (wsPath) {
            el.href = window.location.origin + '/workspace' + wsPath;
          } else {
            el.parentNode.removeChild(el);
          }
        }
      })();
    </script>
    <div id="svgContainer" style="width:100%; overflow-x:auto; margin-top:8px;">
      TREE_SVG_PLACEHOLDER
    </div>
    """.replace('TREE_WS_PATH_JSON', tree_ws_path_json).replace('TREE_SVG_PLACEHOLDER', svg_content)

    click.echo("Writing HTML report ...")
    html = define_html_template(
        summary_table_html=summary_table_html,
        barplot_html=barplot_html,
        heatmap_html=heatmap_html,
        distance_analysis_html=distance_analysis_html,
        tree_html=tree_html,
        metadata_json_string=metadata_json_string,
        n_genomes=n_genomes,
        n_loci=n_loci,
        low_coverage_warning_html=low_coverage_warning_html,
    )

    with open(html_report_path, "w") as f:
        f.write(html)

    metadata_tsv_dst = os.path.join(output_dir, "metadata.tsv")
    if os.path.exists("metadata.tsv"):
        shutil.copy("metadata.tsv", metadata_tsv_dst)
        click.echo("  Metadata TSV written to {}".format(metadata_tsv_dst))

    click.echo("Report written to {}".format(html_report_path))


if __name__ == "__main__":
    cli()
