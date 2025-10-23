import nbformat

# Load the Jupyter Notebook file
with open('skin-cancer-type-detection.ipynb', 'r') as f:
    notebook = nbformat.read(f, as_version=4)

# Extract the code from each cell
code_cells = [cell.source for cell in notebook.cells if cell.cell_type == 'code']

# Print the extracted code
for code in code_cells:
    print(code)
    print('---')