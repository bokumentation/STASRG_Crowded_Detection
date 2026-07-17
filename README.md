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

3. Buka PowerShell, lalu ubah execution policy (sekali saja) dan aktifkan virtual environment:

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
```

4. Install dependensi:

```
pip install -r requirements.txt
```

5. Jalankan aplikasi dengan double-click file `.ps1` di root project, atau dari terminal:

```
.\run_headcounter.ps1
```

atau

```
.\run_movcounter.ps1
```

6. Browser akan otomatis terbuka di `http://127.0.0.1:5000/`.
   Jika tidak, buka secara manual di browser.

7. Tutup terminal untuk menghentikan aplikasi.

## Dua Aplikasi

Proyek ini berisi dua aplikasi terpisah:

| Aplikasi | Folder | Fungsi |
|---|---|---|
| **Head Counter** | `src/headcounter/` | Menghitung jumlah kepala di frame saat ini |
| **Movement Counter** | `src/movement_counter/` | Menghitung pergerakan masuk/keluar melewati garis virtual |

## Membuat Shortcut Desktop

1. Buka PowerShell di root folder proyek ini
2. Jalankan perintah berikut:

```
powershell -ExecutionPolicy Bypass -File create_shortcut.ps1
```

3. Dua shortcut akan muncul di Desktop: **Head Counter** dan **Movement Counter**
4. Double-click shortcut untuk menjalankan aplikasi. Terminal akan muncul - tutup terminal untuk menghentikan aplikasi.

Catatan: Script akan otomatis mengkonversi `tools/logo_stasrg.png` menjadi `tools/logo_stasrg.ico` jika file `.ico` belum ada.
Shortcut menjalankan `run_headcounter.ps1` / `run_movcounter.ps1` via `powershell.exe`.

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

```
root/
├── run_headcounter.ps1          # Script launch Head Counter
├── run_movcounter.ps1           # Script launch Movement Counter
├── create_shortcut.ps1          # Script pembuat shortcut desktop
├── requirements.txt             # Daftar dependensi (untuk kedua aplikasi)
├── README.md
├── src/
│   ├── headcounter/             # Aplikasi Head Counter
│   │   ├── app.py
│   │   ├── survei2.pt
│   │   ├── path_resolver.py
│   │   ├── static/
│   │   └── templates/
│   └── movement_counter/        # Aplikasi Movement Counter
│       ├── app.py
│       ├── head.pt
│       ├── config.json
│       ├── config_writer.py
│       ├── cv_processor_vertical.py
│       ├── cv_processor_horizontal.py
│       ├── path_resolver.py
│       ├── static/
│       └── templates/
└── tools/                       # Script bantu
    ├── build_app.py
    ├── create_ico.py
    └── pngico.py
```

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