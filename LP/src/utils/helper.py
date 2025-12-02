# This file contains general helper functions.

import csv
from src.algorithms.utils.results import Results

def csv_to_results(filepath):
    """
    Read results CSV file into a Results object.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        Results: An object containing the aggregated results.
    """
    with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        results = Results()
        for row in csv_reader:
            # Convert string values to appropriate types if necessary
            iterations = int(row['iterations'])
            times = float(row['times'])
            fvaluegaps = float(row['fvaluegaps'])
            metric_lps = float(row['metricLPs'])
            
            results.log_result(iterations, times, fvaluegaps, metric_lps)
    return results