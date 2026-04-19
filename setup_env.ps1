# AMD Heterogeneous Programming Setup
# ====================================
# Run this script to set up your development environment.
#
# Usage: .\setup_env.ps1

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  AMD Heterogeneous Programming Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# --- 1. Check uv ---
Write-Host "`n[1/4] Checking uv..." -ForegroundColor Yellow
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "  uv not found. Installing..." -ForegroundColor Red
    irm https://astral.sh/uv/install.ps1 | iex
} else {
    $uvVer = (uv --version)
    Write-Host "  uv found: $uvVer" -ForegroundColor Green
}

# --- 2. Create virtual environment + install deps ---
Write-Host "`n[2/4] Setting up Python environment..." -ForegroundColor Yellow
uv sync
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Failed to sync dependencies!" -ForegroundColor Red
    exit 1
}
Write-Host "  Dependencies installed." -ForegroundColor Green

# --- 3. Check hardware ---
Write-Host "`n[3/4] Detecting hardware..." -ForegroundColor Yellow

$cpu = (Get-WmiObject Win32_Processor).Name.Trim()
Write-Host "  CPU: $cpu" -ForegroundColor Green

$gpu = (Get-WmiObject Win32_VideoController | Where-Object { $_.Name -notmatch 'DisplayLink' } | Select-Object -First 1).Name.Trim()
Write-Host "  GPU: $gpu" -ForegroundColor Green

$npu = (Get-PnpDevice | Where-Object { $_.FriendlyName -match 'NPU Compute' } | Select-Object -First 1)
if ($npu) {
    Write-Host "  NPU: $($npu.FriendlyName.Trim()) [$($npu.Status)]" -ForegroundColor Green
} else {
    Write-Host "  NPU: Not detected" -ForegroundColor Yellow
}

# --- 4. Verify Python packages ---
Write-Host "`n[4/4] Verifying packages..." -ForegroundColor Yellow
uv run python -c "import numpy; print(f'  numpy: {numpy.__version__}')"
uv run python -c "import onnxruntime as ort; print(f'  onnxruntime: {ort.__version__}'); print(f'  Providers: {ort.get_available_providers()}')"

try {
    uv run python -c "import pyopencl as cl; platforms = cl.get_platforms(); print(f'  pyopencl: OK ({len(platforms)} platform(s))')"
} catch {
    Write-Host "  pyopencl: Not available (OpenCL GPU compute won't work)" -ForegroundColor Yellow
    Write-Host "  You may need to install AMD OpenCL drivers or build pyopencl from source" -ForegroundColor Yellow
}

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Run the samples:" -ForegroundColor White
Write-Host "    uv run python src/01_system_overview.py     # System overview" -ForegroundColor Gray
Write-Host "    uv run python src/02_gpu_config.py          # GPU deep-dive" -ForegroundColor Gray
Write-Host "    uv run python src/03_npu_config.py          # NPU deep-dive" -ForegroundColor Gray
Write-Host "    uv run python src/04_heterogeneous_demo.py  # CPU vs GPU vs NPU" -ForegroundColor Gray
Write-Host ""
