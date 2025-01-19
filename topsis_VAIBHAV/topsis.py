import pandas as pd
import numpy as np
import os
import sys

def compute_topsis(data_path, priority_weights, benefit_impacts, result_path):
    """
    Compute TOPSIS scores and ranks for a given dataset.

    Parameters:
    -----------
    data_path : str
        Path to the input CSV file containing the decision matrix.
    priority_weights : str
        Comma-separated string of weights (e.g., "1,1,1,2").
    benefit_impacts : str
        Comma-separated string of impacts (e.g., "+,+,-,+").
    result_path : str
        Path to save the resulting CSV file.

    Returns:
    --------
    None
        Saves the results to the specified output file.

    Raises:
    -------
    Exception
        If there are any validation or calculation errors.
    """
    try:
        # Validate file existence
        if not os.path.exists(data_path):
            raise Exception(f"Input file '{data_path}' does not exist")

        # Read input file
        dataset = pd.read_csv(data_path)

        if len(dataset.columns) < 3:
            raise Exception("Input file must contain three or more columns")

        # Convert weights and impacts to appropriate formats
        weight_values = [float(weight) for weight in priority_weights.split(',')]
        impact_values = benefit_impacts.split(',')

        # Validate weights and impacts
        if len(weight_values) != len(dataset.columns) - 1:
            raise Exception("Number of weights must match the number of criteria columns")

        if len(impact_values) != len(dataset.columns) - 1:
            raise Exception("Number of impacts must match the number of criteria columns")

        if not all(impact in ['+', '-'] for impact in impact_values):
            raise Exception("Impacts must be either '+' or '-'")

        # Extract numeric decision matrix
        criteria_matrix = dataset.iloc[:, 1:].values.astype(float)

        # Step 1: Normalize the decision matrix
        normalized_matrix = criteria_matrix / np.sqrt(np.sum(criteria_matrix**2, axis=0))

        # Step 2: Compute weighted normalized matrix
        weighted_matrix = normalized_matrix * weight_values

        # Step 3: Determine ideal best and worst values
        ideal_best = []
        ideal_worst = []

        for index, impact in enumerate(impact_values):
            if impact == '+':
                ideal_best.append(max(weighted_matrix[:, index]))
                ideal_worst.append(min(weighted_matrix[:, index]))
            else:
                ideal_best.append(min(weighted_matrix[:, index]))
                ideal_worst.append(max(weighted_matrix[:, index]))

        # Step 4: Calculate separation measures
        distance_positive = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
        distance_negative = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

        # Step 5: Compute performance scores
        scores = distance_negative / (distance_positive + distance_negative)

        # Add results to the dataset
        dataset['Performance Score'] = scores
        dataset['Rank'] = dataset['Performance Score'].rank(ascending=False)

        # Save results to file
        dataset.to_csv(result_path, index=False)

    except Exception as error:
        raise Exception(str(error))