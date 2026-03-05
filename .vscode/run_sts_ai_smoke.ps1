$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'

& '.\.venv\Scripts\python.exe' -m sts_ai.cli smoke --train-epochs 1
