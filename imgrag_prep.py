import numpy as np
import pandas as pd
import json, csv, os
from datasets import load_dataset

###############################################################################
######################### Part 1: Load DiffusionDB ############################

from urllib.request import urlretrieve

# Download the parquet table
table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
urlretrieve(table_url, 'metadata.parquet')

# Read the table using Pandas
raw_df = pd.read_parquet('metadata.parquet')
raw_df.head()

# Keep top 10K prompts
prompts_raw = raw_df['prompt'][0:10000]

del raw_df


###############################################################################
######################### Part 2: Data Preparation ############################

# Remove prompts with word count less than 10
def filter_strings_with_word_count(strings):
    filtered_strings = []
    for text in strings:
        words = text.split()
        if len(words) >= 10:
            filtered_strings.append(text)
    return filtered_strings

prompts_filtered = filter_strings_with_word_count(prompts_raw)

# remove prompts with very high similarities
import Levenshtein
import concurrent.futures

def remove_similar_strings(strings, threshold):
    unique_strings = []
    step_counter = 0

    def is_unique(s):
        nonlocal unique_strings
        for us in unique_strings:
            distance = Levenshtein.distance(s, us)
            if distance <= threshold:
                return False
        return True

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i, s in enumerate(strings):
            if executor.submit(is_unique, s).result():
                unique_strings.append(s)

            # Print number of strings processed for every 1000 steps
            #if (i + 1) % 1000 == 0:
            #    print(f"Processed {i + 1} strings")

    return unique_strings

# Set a similarity threshold (adjust as needed)
similarity_threshold = 10 # Adjust threshold as desired

# Remove similar prompts
prompts_unique = remove_similar_strings(prompts_filtered, similarity_threshold)

###############################################################################
########################## Part 3: Data Storage ###############################

# Specify the CSV file name
csv_file_name = "prompts_unique.csv"

# Open the CSV file for writing
with open(csv_file_name, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["prompt example"])
    # Write each string as a separate row in the CSV file
    for string in prompts_unique[0:1000]:
        csv_writer.writerow([string])