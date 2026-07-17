# run_movcounter.ps1
# Script untuk menjalankan STASRG Movement Counter
# Double-click untuk memulai aplikasi

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "=== STASRG Movement Counter ===" -ForegroundColor Cyan

if (Test-Path "venv\Scripts\Activate.ps1") {
    . "venv\Scripts\Activate.ps1"
    Write-Host "Virtual environment activated." -ForegroundColor Green
} else {
    Write-Host "WARNING: venv tidak ditemukan." -ForegroundColor Yellow
    Write-Host "  Buat venv: python -m venv venv" -ForegroundColor Yellow
    Write-Host "  Install dep: pip install -r requirements.txt" -ForegroundColor Yellow
}

python src/movement_counter/app.py
Write-Host "Movement Counter stopped." -ForegroundColor Gray
pause