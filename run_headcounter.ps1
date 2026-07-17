# run_headcounter.ps1
# Script untuk menjalankan STASRG Head Counter

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== STASRG Head Counter ===" -ForegroundColor Cyan
Write-Host "Working Directory: $ProjectRoot" -ForegroundColor Gray

Set-Location $ProjectRoot
python src/headcounter/app.py