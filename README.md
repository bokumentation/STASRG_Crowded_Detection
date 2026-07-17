# Initial STAS-RG Crowded Detection

Modified from: Original Source

## Prasyarat

- Python 3.x terinstal
- Pip terinstal

## Cara Menjalankan (Development)

1. Clone repository ini
2. Buat virtual environment:

```
python -m venv venv
```

3. Aktifkan virtual environment (Windows):

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser venv\Scripts\activate
```

4. Install dependensi:

```
pip install -r requirements.txt
```

5. Jalankan aplikasi:

```
python app.py
```

6. Browser akan otomatis terbuka di `http://127.0.0.1:5000/`.
   Jika tidak, buka secara manual di browser.

## Membuat Shortcut Desktop

1. Buka PowerShell di root folder proyek ini
2. Jalankan perintah berikut:

```
powershell -ExecutionPolicy Bypass -File tools/create_shortcut.ps1
```

3. Shortcut dengan nama **STASRG Crowded Detection** akan muncul di Desktop
4. Double-click shortcut tersebut untuk menjalankan aplikasi

Catatan: Script akan otomatis mengkonversi `tools/logo_stasrg.png` menjadi `tools/logo_stasrg.ico` untuk dijadikan icon shortcut.
Tidak perlu build executable - shortcut langsung menjalankan `app.py` menggunakan `pythonw.exe` (tanpa console window).

## Membangun Executable (.exe) - Opsional

1. Pastikan PyInstaller terinstal:

```
pip install pyinstaller
```

2. Jalankan build script:

```
python tools/build_app.py
```

3. Hasil build akan berada di folder `output/` dengan nama folder bernomor (contoh: `STASRG_Crowded_Detection_1`)

## Struktur Proyek

| File / Folder      | Keterangan                                                    |
|--------------------|---------------------------------------------------------------|
| `app.py`           | Aplikasi Flask utama (routing, API, streaming video)          |
| `cv_processor_vertical.py` | Prosesor deteksi dengan garis virtual vertikal (kiri/kanan) |
| `cv_processor_horizontal.py` | Prosesor deteksi dengan garis virtual horizontal (atas/bawah) |
| `config.json`      | File konfigurasi (versi, posisi garis, kapasitas, dll)        |
| `config_writer.py` | Helper untuk menyimpan konfigurasi ke disk secara aman        |
| `path_resolver.py` | Resolver path untuk development dan build PyInstaller         |
| `head.pt`          | Model YOLO untuk deteksi kepala                               |
| `requirements.txt` | Daftar dependensi Python                                      |
| `templates/`       | Template HTML (Flask)                                         |
| `static/`          | Aset statis (CSS, JS, gambar)                                 |
| `tools/`           | Script bantu (build, shortcut, konversi icon)                 |

## Default Configuration

Keterangan:
- Hijau ke merah = masuk
- Merah ke hijau = keluar

```json
{
    "version": "vertical",
    "model_path": "head.pt",
    "video_source": 0,
    "resize": {
        "width": 640,
        "height": 480
    },
    "tracking": {
        "max_disappeared": 100
    },
    "vertical_lines": {
        "entry_line_position": 330,
        "exit_line_position": 310
    },
    "horizontal_lines": {
        "entry_line_position": 250,
        "exit_line_position": 230
    },
    "capacity": {
        "max_crowd_count": 100
    },
    "initial_counts": {
        "entry": 0,
        "exit": 0
    },
    "recording": {
        "output_directory": "Crowded Detection Video"
    }
}