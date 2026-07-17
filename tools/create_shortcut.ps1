# create_shortcut.ps1
# Script untuk membuat 2 shortcut desktop otomatis untuk STASRG Crowded Detection
# Shortcut langsung mengarah ke app.py (tidak perlu build executable)

$ErrorActionPreference = "Stop"

# --- 1. Tentukan path ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$HeadcounterApp = Join-Path $ProjectRoot "src\headcounter\app.py"
$MovcounterApp = Join-Path $ProjectRoot "src\movement_counter\app.py"
$IcoScript = Join-Path $ScriptDir "create_ico.py"
$IcoOutput = Join-Path $ScriptDir "logo_stasrg.ico"
$DesktopDir = [Environment]::GetFolderPath("Desktop")

# Cari pythonw.exe (tanpa console) di lokasi yang sama dengan python.exe
$PythonDir = Split-Path -Parent (Get-Command python -ErrorAction Stop).Source
$PythonwPath = Join-Path $PythonDir "pythonw.exe"

if (-not (Test-Path $PythonwPath)) {
    Write-Host "WARNING: pythonw.exe tidak ditemukan, fallback ke python.exe" -ForegroundColor Yellow
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
} else {
    $PythonExe = $PythonwPath
}

# --- 2. Konversi PNG ke ICO ---
Write-Host "Konversi icon PNG ke ICO..." -ForegroundColor Cyan
$ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
$ProcessInfo.FileName = "python"
$ProcessInfo.Arguments = "`"$IcoScript`""
$ProcessInfo.RedirectStandardOutput = $true
$ProcessInfo.RedirectStandardError = $true
$ProcessInfo.UseShellExecute = $false
$ProcessInfo.CreateNoWindow = $true
$ProcessInfo.WorkingDirectory = $ScriptDir

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

if (-not (Test-Path $IcoOutput)) {
    Write-Host "ERROR: File ICO tidak ditemukan setelah konversi: $IcoOutput" -ForegroundColor Red
    exit 1
}

# --- 3. Buat shortcut Head Counter ---
Write-Host "Membuat shortcut Head Counter..." -ForegroundColor Cyan

$ShortcutPathHC = Join-Path $DesktopDir "Head Counter.lnk"
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPathHC)
$Shortcut.TargetPath = $PythonExe
$Shortcut.Arguments = "`"$HeadcounterApp`""
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.IconLocation = $IcoOutput
$Shortcut.Description = "STASRG Head Counter - Penghitung Jumlah Kepala"
$Shortcut.Save()

Write-Host "  -> $ShortcutPathHC" -ForegroundColor Green

# --- 4. Buat shortcut Movement Counter ---
Write-Host "Membuat shortcut Movement Counter..." -ForegroundColor Cyan

$ShortcutPathMC = Join-Path $DesktopDir "Movement Counter.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPathMC)
$Shortcut.TargetPath = $PythonExe
$Shortcut.Arguments = "`"$MovcounterApp`""
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.IconLocation = $IcoOutput
$Shortcut.Description = "STASRG Movement Counter - Penghitung Pergerakan Masuk/Keluar"
$Shortcut.Save()

Write-Host "  -> $ShortcutPathMC" -ForegroundColor Green

Write-Host ""
Write-Host "Target   : $PythonExe" -ForegroundColor Gray
Write-Host "Icon     : $IcoOutput" -ForegroundColor Gray
Write-Host "Selesai. Dua shortcut telah dibuat di Desktop." -ForegroundColor Green