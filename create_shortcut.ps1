# create_shortcut.ps1
# Script untuk membuat 2 shortcut desktop otomatis untuk STASRG Crowded Detection
# Shortcut mengarah ke script .ps1 di root project

$ErrorActionPreference = "Stop"

# --- 1. Tentukan path ---
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunHeadcounter = Join-Path $ProjectRoot "run_headcounter.ps1"
$RunMovcounter = Join-Path $ProjectRoot "run_movcounter.ps1"
$IcoFile = Join-Path $ProjectRoot "tools\logo_stasrg.ico"
$DesktopDir = [Environment]::GetFolderPath("Desktop")

# Cari powershell.exe
$PowershellExe = (Get-Command powershell.exe -ErrorAction Stop).Source

# --- 2. Konversi PNG ke ICO jika belum ada ---
if (-not (Test-Path $IcoFile)) {
    Write-Host "Konversi icon PNG ke ICO..." -ForegroundColor Cyan
    $IcoScript = Join-Path $ProjectRoot "tools\create_ico.py"
    $ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
    $ProcessInfo.FileName = "python"
    $ProcessInfo.Arguments = "`"$IcoScript`""
    $ProcessInfo.RedirectStandardOutput = $true
    $ProcessInfo.RedirectStandardError = $true
    $ProcessInfo.UseShellExecute = $false
    $ProcessInfo.CreateNoWindow = $true
    $ProcessInfo.WorkingDirectory = Join-Path $ProjectRoot "tools"

    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $ProcessInfo
    $Process.Start() | Out-Null
    $StdOut = $Process.StandardOutput.ReadToEnd()
    $StdErr = $Process.StandardError.ReadToEnd()
    $Process.WaitForExit()

    if ($Process.ExitCode -ne 0) {
        Write-Host "ERROR: Gagal konversi ICO:" -ForegroundColor Red
        Write-Host $StdErr -ForegroundColor Red
        exit 1
    }
    Write-Host $StdOut -ForegroundColor Green
}

if (-not (Test-Path $IcoFile)) {
    Write-Host "ERROR: File ICO tidak ditemukan: $IcoFile" -ForegroundColor Red
    Write-Host "  Jalankan: python tools/create_ico.py" -ForegroundColor Yellow
    exit 1
}

# --- 3. Buat shortcut Head Counter ---
Write-Host "Membuat shortcut Head Counter..." -ForegroundColor Cyan

$ShortcutPathHC = Join-Path $DesktopDir "Head Counter.lnk"
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPathHC)
$Shortcut.TargetPath = $PowershellExe
$Shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$RunHeadcounter`""
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.IconLocation = $IcoFile
$Shortcut.Description = "STASRG Head Counter - Penghitung Jumlah Kepala"
$Shortcut.Save()

Write-Host "  -> $ShortcutPathHC" -ForegroundColor Green

# --- 4. Buat shortcut Movement Counter ---
Write-Host "Membuat shortcut Movement Counter..." -ForegroundColor Cyan

$ShortcutPathMC = Join-Path $DesktopDir "Movement Counter.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPathMC)
$Shortcut.TargetPath = $PowershellExe
$Shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$RunMovcounter`""
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.IconLocation = $IcoFile
$Shortcut.Description = "STASRG Movement Counter - Penghitung Pergerakan Masuk/Keluar"
$Shortcut.Save()

Write-Host "  -> $ShortcutPathMC" -ForegroundColor Green

Write-Host ""
Write-Host "Target   : $PowershellExe" -ForegroundColor Gray
Write-Host "Icon     : $IcoFile" -ForegroundColor Gray
Write-Host "Selesai. Dua shortcut telah dibuat di Desktop." -ForegroundColor Green