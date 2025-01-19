# TOPSIS Analysis Tool

This project implements the Technique for Order Preference by Similarity to Ideal Solution (TOPSIS), a method for multi-criteria decision-making. The tool processes a decision matrix, applies specified weights and impacts for each criterion, and ranks the alternatives based on their performance. This Python package, developed under the acronym `topsis_Vaibhav_102203877`, was created as a University Student Project.

## Features
- Supports input of decision matrices in CSV or Excel formats (`.xlsx`, `.xls`).
- Allows customization of weights and impacts for each criterion.
- Generates ranked results as a CSV file.

---

## Installation
1. Clone the repository or download the script.
2. Ensure Python (version 3.6 or later) is installed.
3. Install the required dependencies using pip:

```bash
pip install numpy pandas openpyxl
```

---

## Usage

The script uses command-line arguments to specify the input file, weights, impacts, and output file.

### Command
```bash
topsis <input_file> <weights> <impacts> <output_file>
```

### Parameters
- **`<input_file>`**: Path to the decision matrix file. Accepts `.csv`, `.xlsx`, or `.xls` formats.
- **`<weights>`**: Comma-separated weights for each criterion (e.g., `0.3,0.4,0.3`).
- **`<impacts>`**: Comma-separated impacts for each criterion (`+` for beneficial, `-` for non-beneficial, e.g., `+,+,-`).
- **`<output_file>`**: Path to save the output CSV file with rankings.

### Example
#### Input File (`data.csv`):
| Alternatives | Criterion 1 | Criterion 2 | Criterion 3 |
|--------------|-------------|-------------|-------------|
| A1           | 250         | 16          | 12          |
| A2           | 200         | 20          | 10          |
| A3           | 300         | 18          | 15          |

#### Command:
```bash
topsis data.csv 0.3,0.4,0.3 +,+,- output.csv
```

#### Output File (`output.csv`):
| Alternatives | Criterion 1 | Criterion 2 | Criterion 3 | Rank |
|--------------|-------------|-------------|-------------|------|
| A1           | 250         | 16          | 12          | 2    |
| A2           | 200         | 20          | 10          | 3    |
| A3           | 300         | 18          | 15          | 1    |

---

## Functions

### `topsis(data, weights, impacts)`
Executes the TOPSIS algorithm and returns the rankings of the alternatives.

#### Parameters:
- **`data`**: A 2D list or numpy array representing the decision matrix.
- **`weights`**: A list of weights for each criterion.
- **`impacts`**: A list of impacts (`+` or `-`) for each criterion.

#### Returns:
- A list of ranks for the alternatives.

---

### `excel_to_csv(excel_file, csv_output)`
Converts an Excel file into a CSV file.

#### Parameters:
- **`excel_file`**: Path to the Excel file.
- **`csv_output`**: Path to save the converted CSV file.

#### Returns:
- The path to the converted CSV file.

---

## Notes
- The number of weights and impacts must match the number of criteria in the decision matrix.
- The decision matrix should not include alternative names in the criteria columns (e.g., the first column should list alternatives like A1, A2, etc.).

---

## License
This project is licensed under the MIT License.

---

Let me know if you need further adjustments!