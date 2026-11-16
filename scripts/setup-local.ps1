param([ValidateSet("cuda", "cpu")][string]$Backend = "cuda")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
}
& uv sync --python 3.12 --extra $Backend
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& .\.venv\Scripts\python.exe -m local_rl.doctor
if ($LASTEXITCODE -ne 0) { throw "Environment check failed" }
