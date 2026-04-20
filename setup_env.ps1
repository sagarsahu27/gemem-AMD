# AMD Heterogeneous Programming Setup
# ====================================
# Run this script to set up your development environment using Conda.
#
# Usage: .\setup_env.ps1

$ErrorActionPreference = "Stop"
$ENV_NAME = "amd-hetero"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  AMD Heterogeneous Programming Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# --- 1. Check conda ---
Write-Host "`n[1/4] Checking conda..." -ForegroundColor Yellow
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "  conda not found!" -ForegroundColor Red
    Write-Host "  Install Miniconda from: https://docs.anaconda.com/miniconda/" -ForegroundColor Red
    Write-Host "  Or Anaconda from: https://www.anaconda.com/download" -ForegroundColor Red
    exit 1
} else {
    $condaVer = (conda --version)
    Write-Host "  $condaVer" -ForegroundColor Green
}

# --- 2. Create conda environment + install deps ---
Write-Host "`n[2/4] Setting up conda environment '$ENV_NAME'..." -ForegroundColor Yellow
$envExists = conda env list | Select-String "^\s*$ENV_NAME\s"
if ($envExists) {
    Write-Host "  Environment '$ENV_NAME' exists. Updating..." -ForegroundColor Yellow
    conda env update -n $ENV_NAME -f environment.yml --prune
} else {
    Write-Host "  Creating environment '$ENV_NAME'..." -ForegroundColor Yellow
    conda env create -f environment.yml
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Failed to create/update conda environment!" -ForegroundColor Red
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
conda run -n $ENV_NAME python -c "import numpy; print(f'  numpy: {numpy.__version__}')"
conda run -n $ENV_NAME python -c "import onnxruntime as ort; print(f'  onnxruntime: {ort.__version__}'); print(f'  Providers: {ort.get_available_providers()}')"

try {
    conda run -n $ENV_NAME python -c "import pyopencl as cl; platforms = cl.get_platforms(); print(f'  pyopencl: OK ({len(platforms)} platform(s))')"
} catch {
    Write-Host "  pyopencl: Not available (OpenCL GPU compute won't work)" -ForegroundColor Yellow
    Write-Host "  You may need to install AMD OpenCL drivers or build pyopencl from source" -ForegroundColor Yellow
}

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Activate the environment:" -ForegroundColor White
Write-Host "    conda activate $ENV_NAME" -ForegroundColor Gray
Write-Host ""
Write-Host "  Run the samples:" -ForegroundColor White
Write-Host "    python src/01_system_overview.py     # System overview" -ForegroundColor Gray
Write-Host "    python src/02_gpu_config.py          # GPU deep-dive" -ForegroundColor Gray
Write-Host "    python src/03_npu_config.py          # NPU deep-dive" -ForegroundColor Gray
Write-Host "    python src/04_heterogeneous_demo.py  # CPU vs GPU vs NPU" -ForegroundColor Gray
Write-Host ""
