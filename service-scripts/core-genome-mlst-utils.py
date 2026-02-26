#!/usr/bin/env python3
import argparse
import click
import csv
import glob
import json
import os
import re
import shutil

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
        seen_rows = set()  # track duplicate rows from precomputed data
        for idx, row in enumerate(reader):
            if not row:
                continue  # skip completely empty lines

            # Skip duplicate rows 
            row_key = tuple(row)
            if row_key in seen_rows:
                print("Skipping duplicate row: {}".format(row[0]))
                continue
            seen_rows.add(row_key)

            # Record mapping from index -> original first-column value
            col1_value = row[0]
            mapping[idx] = col1_value

            # Start with a copy of the row
            new_row_orig = row[:]

            # Replace underscores with periods in the first column (genome ID)
            new_row_orig[0] = col1_value.replace('_', '.')

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

    This is a cgMLST-specific replacement for the SNP report's create_metadata_table,
    which converts '.' → '_' for kSNP4 compatibility.  Here we need genome_ids to match
    the period-delimited IDs used in result_alleles.tsv (e.g. '1008297.7').

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
    metadata_df.to_csv(tsv_out, index=False, sep="\t")

    return metadata, metadata_df


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
    df = pd.read_csv(result_alleles_tsv, sep="\t", index_col=0, dtype=str)

    df.index = df.index.astype(str).str.replace("_", ".", regex=False)

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


def mask_allele_df(raw_df):
    """Return a numeric DataFrame suitable for distance computation.

    INF-* values are treated as valid novel alleles (kept as unique integers
    would be, but for distance purposes the INF- prefix is stripped and the
    numeric suffix used).  All other non-numeric codes → 0.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw allele calls with string values.

    Returns
    -------
    masked : np.ndarray, shape (n_genomes, n_loci), dtype int32
    genome_ids : list of str
    """
    def _mask_val(v):
        v = str(v).strip()
        if v.startswith("INF-"):
            suffix = v[4:]
            return int(suffix) if suffix.isdigit() else 0
        if v.isdigit():
            return int(v)
        return 0

    masked = raw_df.apply(lambda col: col.apply(_mask_val)).values.astype("int32")
    return masked, raw_df.index.tolist()


def compute_distance_matrix(np_matrix):
    """Compute symmetric pairwise allelic distance matrix.

    Two loci contribute to the distance only when both samples have an exact
    (non-zero) value AND those values differ.  Loci that are missing (0) in
    either sample are ignored.

    Parameters
    ----------
    np_matrix : np.ndarray, shape (n, m), dtype int32

    Returns
    -------
    dist_matrix : np.ndarray, shape (n, n), dtype int32
    """
    n = np_matrix.shape[0]
    dist_matrix = np.zeros((n, n), dtype="int32")

    for i in range(n):
        row = np_matrix[i:i + 1, :]
        rest = np_matrix[i:, :]
        shared = row * rest                    # non-zero only where both exact
        diffs = np.count_nonzero(
            shared * (row - rest), axis=-1
        ).astype("int32")
        dist_matrix[i, i:] = diffs
        dist_matrix[i:, i] = diffs

    return dist_matrix


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
        Index = genome_id; columns = Exact, INF, LNF, ASM, ALM, …

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


def create_summary_table(genome_ids, loci_ids, coverage_df):
    """Build a small summary DataFrame for the report header table.

    Parameters
    ----------
    genome_ids : list of str
    loci_ids : list of str
    coverage_df : pd.DataFrame

    Returns
    -------
    table_html : str
    """
    total_exact = int(coverage_df["Exact"].sum() + coverage_df.get("INF", 0).sum())
    total_cells = len(genome_ids) * len(loci_ids)
    pct_exact = round(100 * total_exact / total_cells, 1) if total_cells > 0 else 0

    summary = pd.DataFrame([{
        "Number of Genomes": len(genome_ids),
        "Number of Loci in Schema": len(loci_ids),
        "Total Loci Exact": total_exact,
        "Total Possible Calls": total_cells,
        "Percent Exact (%)": pct_exact,
    }])

    return generate_table_html_2(summary, table_width="75%")


def build_heatmap_html(genome_ids, dist_matrix, metadata_json_string, svg_content=''):
    """Build the interactive heatmap HTML block.

    Parameters
    ----------
    genome_ids : list of str
        Hierarchically clustered genome IDs.
    dist_matrix : list of lists (int)
        Clustered distance matrix.
    metadata_json_string : str
        JSON-serialised metadata (list of dicts).
    svg_content : str
        SVG file content to embed directly in the report.

    Returns
    -------
    heatmap_html : str
    """
    genome_ids_json = json.dumps(genome_ids)
    dist_matrix_json = json.dumps(dist_matrix)

    heatmap_html = """
    <h2>Allelic Distance Heatmap</h2>
    <p>
      The Allelic Distance Heatmap visualises pairwise differences in allele
      assignments between genomes. Each cell represents the number of loci
      where two genomes carry different alleles, considering only loci
      successfully exact in both samples. Lower distances indicate closer
      genetic relationships. <strong>Click any cell</strong> to compare the
      metadata for that pair of genomes in the panel below.
    </p>

    <div class="heatmap-controls">
      <h4>Filter and Sort the Data:</h4>
      <label>Reorder Heatmap by Metadata:
        <select id="metadataFieldSelect" onchange="recolorHeatmap()">
          <!-- options populated dynamically -->
        </select>
      </label>
    </div>

    <div class="linkage-controls">
      <h4>Recolor Heatmap According to Linkage Thresholds:</h4>
      <label>Strong Linkage Thresholds:
        <input type="number" id="t0" value="0" disabled style="width:40px;">
        <input type="number" id="t1a" value="10" style="width:40px;">
      </label>
      <label>Mid Linkage Thresholds:
        <input type="number" id="t1b" value="10" style="width:40px;">
        <input type="number" id="t2a" value="40" style="width:40px;">
      </label>
      <label>Weak Linkage Thresholds:
        <input type="number" id="t2b" value="40" style="width:40px;">
        <input type="number" id="t3"  placeholder="Max" disabled style="width:40px;">
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

    <!-- Tree (full width, below heatmap) -->
    <h3 style="margin-top:32px;">Phylogenetic Tree</h3>
    <div id="svgContainer" style="width:100%; overflow-x:auto; margin-top:8px;">
      TREE_SVG_PLACEHOLDER
    </div>

    <script>
      // ===== Embedded data =====
      const genomeLabels = {genome_ids_json};
      const distMatrix   = {dist_matrix_json};
      const metadata     = {metadata_json_string};

      // Build a lookup once: genome_id → metadata object
      const idToMeta = {{}};
      metadata.forEach(obj => {{ idToMeta[obj.genome_id] = obj; }});

      // ===== Sync paired threshold inputs =====
      function syncThresholdInputs() {{
        const t1a = document.getElementById('t1a');
        const t1b = document.getElementById('t1b');
        const t2a = document.getElementById('t2a');
        const t2b = document.getElementById('t2b');

        t1a.addEventListener('input', () => {{ t1b.value = t1a.value; recolorHeatmap(); }});
        t1b.addEventListener('input', () => {{ t1a.value = t1b.value; recolorHeatmap(); }});
        t2a.addEventListener('input', () => {{ t2b.value = t2a.value; recolorHeatmap(); }});
        t2b.addEventListener('input', () => {{ t2a.value = t2b.value; recolorHeatmap(); }});

        function validateOnBlur() {{
          const t1 = parseFloat(t1a.value);
          const t2 = parseFloat(t2a.value);
          if (t1 > t2) {{
            alert("Strong threshold must be less than or equal to Mid threshold. Resetting.");
            t1a.value = t2;
            t1b.value = t2;
            recolorHeatmap();
          }}
        }}
        t1a.addEventListener('blur', validateOnBlur);
        t1b.addEventListener('blur', validateOnBlur);
      }}
      document.addEventListener('DOMContentLoaded', syncThresholdInputs);

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

      // ===== Bin values for colour scale =====
      function assignBin(val, t1, t2, maxVal) {{
        if (val === 0)    return 0;
        if (val < t1)     return 1;
        if (val < t2)     return 2;
        if (val < maxVal) return 3;
        return 4;
      }}

      function getColorScale() {{
        return [
          [0.0,  '#440154'],
          [0.25, '#3b528b'],
          [0.5,  '#21918c'],
          [0.75, '#5ec962'],
          [1.0,  '#fde725']
        ];
      }}

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

      // ===== Handle heatmap cell click → show comparison panel =====
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
        document.getElementById('genome1Label').innerHTML = `<a href="https://www.bv-brc.org/view/Genome/${{id1}}" target="_blank">${{id1}}</a>`;
        document.getElementById('genome2Label').innerHTML = `<a href="https://www.bv-brc.org/view/Genome/${{id2}}" target="_blank">${{id2}}</a>`;

        renderMetaTable(document.getElementById('meta1Table'), meta1);
        renderMetaTable(document.getElementById('meta2Table'), meta2);

        const panel = document.getElementById('comparisonPanel');
        panel.style.display = 'block';
        panel.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
      }}

      // ===== Draw / redraw heatmap =====
      function recolorHeatmap() {{
        const t1        = parseInt(document.getElementById('t1a').value);
        const t2        = parseInt(document.getElementById('t2a').value);
        const metaField = document.getElementById('metadataFieldSelect').value;

        let labels = genomeLabels.slice();
        let matrix = distMatrix.map(row => row.slice());

        if (metaField) {{
          const reordered = reorderByField(metaField, labels, matrix);
          labels = reordered.newLabels;
          matrix = reordered.newMatrix;
        }}

        const maxVal = Math.max(...matrix.flat());
        const bins   = matrix.map(row => row.map(val => assignBin(val, t1, t2, maxVal)));

        const hoverText = matrix.map((row, i) =>
          row.map((val, j) => `${{labels[i]}} vs ${{labels[j]}}<br>Allelic Distance: ${{val}}<br><i>Click for full metadata</i>`)
        );

        const traceData = [{{
          z:          bins,
          x:          labels,
          y:          labels,
          type:       'heatmap',
          colorscale: getColorScale(),
          zmin: 0,
          zmax: 4,
          text:       hoverText,
          hoverinfo:  'text',
          colorbar: {{
            tickvals: [0, 1, 2, 3, 4],
            ticktext: ['Zero', 'Strong', 'Mid', 'Weak', 'Max Value'],
            title:    'Linkage Strength'
          }}
        }}];

        const layout = {{
          title: `Allelic Distance Heatmap (Thresholds: ${{t1}}, ${{t2}})` +
                 (metaField ? ` – Reordered by "${{metaField}}"` : ''),
          xaxis: {{ type: 'category', tickangle: 45 }},
          yaxis: {{ type: 'category', tickangle: 45 }}
        }};

        const heatmapDiv = document.getElementById('heatmap');
        Plotly.newPlot(heatmapDiv, traceData, layout);
        heatmapDiv.on('plotly_click', onHeatmapClick);
      }}

      // Initial render
      recolorHeatmap();
    </script>
    """.format(
        genome_ids_json=genome_ids_json,
        dist_matrix_json=dist_matrix_json,
        metadata_json_string=metadata_json_string,
    )
    heatmap_html = heatmap_html.replace('TREE_SVG_PLACEHOLDER', svg_content)

    return heatmap_html


def define_html_template(summary_table_html, barplot_html,
                          heatmap_html, metadata_json_string,
                          n_genomes, n_loci):
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
    .heatmap-controls,
    .linkage-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 1em;
    }}
    .heatmap-controls h4,
    .linkage-controls h4 {{
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
    This report summarizes the results of a Core Genome Multi-Locus Sequence
    Typing (cgMLST) analysis performed with
    <a href="https://chewbbaca.readthedocs.io" target="_blank">chewBBACA</a>.
    cgMLST is a standardised, high-resolution approach to bacterial typing
    that compares allele assignments across hundreds to thousands of shared
    genes (loci), providing a reproducible and portable measure of genomic
    relatedness.

    Note, you can download the images of the plots in this
    report by clicking on the camera icon in the upper left-hand corner of
    each plot.  Poor quality assemblies can impact performance.
  </p>

  <h3>About the Analysis Workflow</h3>
  <p>
    The analysis begins with a set of bacterial genome assemblies and a
    curated cgMLST schema — a defined collection of loci representative of
    the core genome of the species. chewBBACA's AlleleCall module identifies
    and assigns allele numbers at each locus in each genome.
  </p>
  <p>
    Loci that could not be reliably exact receive one of the following
    classification codes:
  </p>
  <ul style="list-style-type: disc; padding-left: 25px;">
    <li><b>LNF</b> – Locus Not Found: no match detected in the genome.</li>
    <li><b>ASM</b> – Allele Smaller than Minimum: matched region is too short.</li>
    <li><b>ALM</b> – Allele Larger than Maximum: matched region is too long.</li>
    <li><b>NIPH</b> – Non-Informative Paralogous Hit: multiple matches of equal quality.</li>
    <li><b>NIPHEM</b> – Non-Informative Paralogous Hit with Exact Match: multiple exact matches.</li>
    <li><b>PLOT3</b> / <b>PLOT5</b> – Possible LOcus on the Tip of a contig (3′ or 5′ end).</li>
    <li><b>INF</b> – Inferred new allele: a valid but previously unobserved allele sequence.</li>
  </ul>
  <p>
    Pairwise allelic distances are computed from the exact loci.  Only loci
    successfully exact in <em>both</em> genomes being compared contribute to
    the distance.  A distance of zero means the two genomes share identical
    alleles at every mutually exact locus.
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
    The chart below shows the breakdown of locus call outcomes for each
    sample. The hover will display the genome id followed by the number
    of loci called.
    <b>Exact</b> (green) loci received a numeric allele assignment.
    <b>INF</b> (teal) loci were assigned a newly inferred allele.  All other
    colors represent missing or problematic calls that do not contribute to
    the distance calculation.  Samples with a high proportion of missing loci
    may warrant quality review.
  </p>
  <div class="plot-container">
    <div class="plot" id="lociCoveragePlot">
      {barplot_html}
    </div>
  </div>

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
          td.innerHTML = row[header];
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
  <!-- REFERENCES                                                           -->
  <!-- ================================================================== -->
  <h3>References</h3>
  <ol type="1">
    <li>
      Oliveira PH, Touchon M, Cury J, Rocha EPC. (2017). The chromosomal
      organisation of horizontal gene transfer in bacteria. <em>Nature
      Communications</em>, 8, 841.
      <a href="https://doi.org/10.1038/s41467-017-00808-w" target="_blank">
        https://doi.org/10.1038/s41467-017-00808-w</a>
    </li>
    <li>
      Silva M, Machado MP, Silva DN, Rossi M, Moran-Gilad J, Santos S,
      Ramirez M, Carriço JA. (2018). chewBBACA: A complete suite for gene-by-gene
      schema creation and strain identification. <em>Microbial Genomics</em>,
      4(3).
      <a href="https://doi.org/10.1099/mgen.0.000166" target="_blank">
        https://doi.org/10.1099/mgen.0.000166</a>
    </li>
  </ol>

</body>
</html>""".format(
        n_genomes=n_genomes,
        n_loci=n_loci,
        summary_table_html=summary_table_html,
        barplot_html=barplot_html,
        heatmap_html=heatmap_html,
        metadata_json_string=metadata_json_string,
    )

    return html


@cli.command()
@click.argument("result_alleles")
@click.argument("metadata_json")
@click.argument("html_report_path")
@click.option("--svg-dir", default="work", show_default=True,
              help="Directory to search for the tree SVG file (*.svg).")
def write_html_report(result_alleles, metadata_json, html_report_path, svg_dir):
    """Generate an interactive cgMLST HTML report.

    \b
    Arguments:
      RESULT_ALLELES   Path to chewBBACA result_alleles.tsv
      METADATA_JSON    Path to genome_metadata.json
      HTML_REPORT_PATH Path for the output HTML report
    """
    click.echo("Parsing result_alleles.tsv ...")
    genome_ids, loci_ids, raw_df, coverage_df = parse_result_alleles(result_alleles)
    n_genomes = len(genome_ids)
    n_loci = len(loci_ids)
    click.echo("  {} genomes x {} loci".format(n_genomes, n_loci))

    click.echo("Masking allele calls ...")
    masked_matrix, masked_ids = mask_allele_df(raw_df)

    click.echo("Computing pairwise allelic distances ...")
    dist_matrix = compute_distance_matrix(masked_matrix)

    click.echo("Clustering for heatmap ...")
    clustered_labels, clustered_matrix = cluster_heatmap_data(masked_ids, dist_matrix)

    click.echo("Building metadata table ...")
    metadata, metadata_df = create_cgmlst_metadata_table(metadata_json, "cgmlst_metadata.tsv")
    metadata_json_string = json.dumps(metadata)

    click.echo("Creating loci coverage bar plot ...")
    barplot_html = create_loci_coverage_plot(coverage_df)

    click.echo("Building summary table ...")
    summary_table_html = create_summary_table(genome_ids, loci_ids, coverage_df)

    click.echo("Assembling heatmap ...")
    svg_candidates = sorted(glob.glob(os.path.join(svg_dir, '*.svg')))
    if svg_candidates:
        with open(svg_candidates[0]) as f:
            svg_content = f.read()
    else:
        svg_content = '<p style="color:#555;">Tree SVG not found. Expected at {}/*.svg</p>'.format(svg_dir)
    heatmap_html = build_heatmap_html(clustered_labels, clustered_matrix, metadata_json_string, svg_content=svg_content)

    click.echo("Writing HTML report ...")
    html = define_html_template(
        summary_table_html=summary_table_html,
        barplot_html=barplot_html,
        heatmap_html=heatmap_html,
        metadata_json_string=metadata_json_string,
        n_genomes=n_genomes,
        n_loci=n_loci,
    )

    with open(html_report_path, "w") as f:
        f.write(html)

    click.echo("Report written to {}".format(html_report_path))


if __name__ == "__main__":
    cli()