# Changelog

## [Unreleased] - 2026-07-18

### Added
- `docs/` folder: dokumentasi lengkap proyek
  - `SETUP_ALAT.md`: persiapan alat, diagram blok hardware (Mermaid), troubleshooting
  - `CARA_KERJA_PROGRAM.md`: penjelasan algoritma, diagram blok sistem, diagram alir per-frame
  - `REFERENSI_PAPER.md`: daftar referensi paper
  - `PERUBAHAN.md`: perbandingan lengkap vs versi original mahasiswa magang, kelebihan program baru
- `docs/img/`: gambar pendukung dokumentasi (USB extender, housing CCTV, lensa wide)
- File `temp/`: arsip versi original mahasiswa magang sebagai baseline perbandingan

## [0.1.0] - 2026-07-17

### Added
- `run_headcounter.ps1` dan `run_movcounter.ps1` untuk auto-aktivasi venv dan launch aplikasi
- `create_shortcut.ps1` di root untuk membuat shortcut desktop Head Counter dan Movement Counter
- `tools/create_ico.py` untuk konversi PNG ke ICO
- `requirements.txt` gabungan di root untuk kedua aplikasi
- Bagian baru di `README.md`: Dua Aplikasi, cara menjalankan via `.ps1`, shortcut desktop, struktur proyek terbaru

### Changed
- Restruktur proyek: `headcounter` dan `movement_counter` dipindahkan ke `src/`
- Shortcut desktop sekarang mengarah ke `run_headcounter.ps1` / `run_movcounter.ps1` via `powershell.exe`
- `cv_processor_vertical.py` kini menggunakan `get_resource_path()` untuk load model, konsisten dengan `HorizontalProcessor`

### Fixed
- `src/headcounter/templates/index.html`: `telkom.png` -> `telkom_logo.png`

### Removed
- `create_shortcut.ps1` usang di root (lama)
- `tools/pngico.py`, `tools/output.ico`, `tools/build_app.py` (tidak relevan)
- `src/headcounter/requirements.txt` dan `src/movement_counter/requirements.txt` (digantikan root `requirements.txt`)
