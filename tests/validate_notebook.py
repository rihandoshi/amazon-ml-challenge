"""Validate the generated notebook's code cells for syntax errors."""
import json, ast, sys

nb = json.load(open("sagemaker_pipeline.ipynb", encoding="utf-8"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Checking {len(code_cells)} code cells...")

errors = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = cell["source"][0]
    # Skip shell commands
    src_clean = "\n".join(
        line if not line.strip().startswith("!") else "pass  # shell: " + line.strip()
        for line in src.split("\n")
    )
    try:
        ast.parse(src_clean)
    except SyntaxError as e:
        print(f"  SYNTAX ERROR in cell {i}: {e}")
        # Show the problematic line
        lines = src_clean.split("\n")
        if e.lineno and e.lineno <= len(lines):
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            for j in range(start, end):
                marker = ">>>" if j == e.lineno - 1 else "   "
                print(f"    {marker} {j+1}: {lines[j]}")
        errors += 1

if errors == 0:
    print("All code cells have valid Python syntax!")
else:
    print(f"\n{errors} cells with syntax errors")
    sys.exit(1)
