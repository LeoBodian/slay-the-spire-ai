$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'

& '.\.venv\Scripts\python.exe' -m pytest -q
