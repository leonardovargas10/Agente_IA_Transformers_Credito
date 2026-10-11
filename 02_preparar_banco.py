"""Reconstrói corpus e banco executando os notebooks 01 e 02 em ordem.

Uso:
    python 02_preparar_banco.py

O script existe para operação reproduzível. Para aprender cada etapa, executar os
dois notebooks manualmente e inspecionar suas saídas.
"""
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
NOTEBOOKS = [
    ROOT / "01_documentos_extracao_e_chunks.ipynb",
    ROOT / "02_banco_vetorial_hnsw_cosseno_e_mmr.ipynb",
]

for caminho in NOTEBOOKS:
    notebook = nbformat.read(caminho, as_version=4)
    executor = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="dsadlp1",
        resources={"metadata": {"path": str(ROOT)}},
    )
    executor.execute()
    nbformat.write(notebook, caminho)
    print("Executado:", caminho.name)

print("Corpus processado e banco vetorial reconstruídos.")
