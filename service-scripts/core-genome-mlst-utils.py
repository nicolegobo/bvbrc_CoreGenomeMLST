#!/usr/bin/env python3
import argparse
import click
import csv
import json
import os
import re
import shutil

from pathlib import Path

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
        for idx, row in enumerate(reader):
            if not row:
                continue  # skip completely empty lines

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

if __name__ == "__main__":
    cli()