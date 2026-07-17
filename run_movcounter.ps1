# run_movcounter.ps1
# Script untuk menjalankan STASRG Movement Counter

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== STASRG Movement Counter ===" -ForegroundColor Cyan
Write-Host "Working Directory: $ProjectRoot" -ForegroundColor Gray

Set-Location $ProjectRoot
python src/movement_counter/app.py