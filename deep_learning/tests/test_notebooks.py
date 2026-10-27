from pathlib import Path

from nbclient import NotebookClient
import nbformat
import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOK_FILES = (
    "atividade_tensorflow_rede_neural_atraso_entrega.ipynb",
    "rede_neural_tensorflow_backpropagation.ipynb",
)


@pytest.mark.parametrize("notebook_name", NOTEBOOK_FILES)
def test_notebook_executes_without_errors(notebook_name):
    notebook_path = NOTEBOOKS_DIR / notebook_name
    notebook = nbformat.read(notebook_path, as_version=4)

    executed_notebook = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    ).execute()

    code_cells = [cell for cell in executed_notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(
        not any(output.output_type == "error" for output in cell.outputs) for cell in code_cells
    )
