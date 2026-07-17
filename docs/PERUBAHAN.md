# Perubahan dari Versi Original Mahasiswa Magang

Proyek ini merupakan improvisasi dari dua proyek mahasiswa magang lab sebelumnya:

- **Head Counter** oleh Kurnia (`temp/HeadCounter-repo-kurnia-ori/`)
- **Movement Counter** oleh Adib (`temp/MovCounter-repo-adib-ori/`)

Berikut adalah daftar perubahan yang telah dilakukan.

---

## A. Head Counter

### Backend (Python)

| Aspek | Original (Kurnia) | Perbaikan |
|---|---|---|
| Arsitektur | 1 file `app.py` (245 baris, semua di root) | Modular: `app.py` + `path_resolver.py` di `src/headcounter/` |
| Load model | Hardcoded: `YOLO('survei2.pt')` | `get_resource_path()` - mendukung development dan PyInstaller build |
| Thread safety | Tidak ada lock | `data_lock` (threading.Lock) untuk semua shared state |
| Konfigurasi | Semua hardcoded (`confidence_threshold=0.3`, `log_interval` 60 detik, `NILAI_TRIGGER_KERAMAIAN=50`) | Endpoint `/api/konfigurasi` (GET/POST) untuk mengubah trigger, log interval, confidence threshold via web UI |
| Stabilization | Tidak ada | 3 detik delay + countdown overlay di frame |
| Browser launch | Tidak ada | Auto-launch Chrome/Edge App Mode fullscreen |
| Shutdown | Tidak ada | `/shutdown` POST endpoint + `cap.release()` |
| Error handling | Silent fail jika kamera error | Frame error dengan teks "KAMERA TERPUTUS" |
| Penamaan | `currentvisitor.html`, `hasilcount.html`, camelCase | `index.html`, `konfigurasi.html`, `tentang.html` (konsisten) |

### Frontend (HTML/JS)

| Aspek | Original (Kurnia) | Perbaikan |
|---|---|---|
| CSS | Tailwind via CDN | Tailwind lokal (`static/css/tailwind.min.css`) |
| Layout | Video di kanan, count di kiri dengan total count di bawah | Video di kiri, count di kanan (lebih compact) |
| Total count | Ditampilkan sebagai angka besar terpisah | Dihapus - hanya current count (lebih relevan untuk monitoring real-time) |
| Logo | `logo G.png` (nama tidak standar) | `logo_stasrg.png` |
| Crowd status | `Crowd` / `Overcrowded` (threshold 200, hardcoded) | `Normal` / `Padat` (configurable via API) |
| Bug: `fecth` typo | Ada (tidak berfungsi) | Fix: `fetch` |
| Bug: `telkom.png` | Referensi file tidak ada | Fix: `telkom_logo.png` |
| Reset count | Tidak ada lock, `visitor_data.clear()` | Lock-based, reset terstruktur |
| Download protection | Tidak ada | `is_downloading` flag untuk mencegah shutdown saat download |
| Halaman | 2 halaman (hasilcount, currentvisitor) | 3 halaman (index, konfigurasi, tentang) |

---

## B. Movement Counter

### Backend (Python)

| Aspek | Original (Adib) | Perbaikan |
|---|---|---|
| Arsitektur | 1 file `app.py` (251 baris, semua di root) | Modular: 6 file terpisah (`app.py`, `cv_processor_vertical.py`, `cv_processor_horizontal.py`, `config_writer.py`, `path_resolver.py`, `config.json`) |
| Mode garis | Hanya horizontal (`detect_direction` pakai `cy`) | Vertical DAN horizontal, dipilih via `config.json` |
| Garis virtual | Hardcoded (`entry=320`, `exit=160`) | Configurable via `config.json` + web UI `/konfigurasi` |
| Swap direction | Tidak ada | `swap_direction` untuk membalik logika arah tanpa ubah posisi garis |
| Recording | Tidak ada | Recording ke `~/Videos/Crowded Detection Video/` (MP4, 15fps) |
| Timestamp | Tidak ada | Timestamp overlay di kanan atas frame |
| Tampilan garis | Solid (100% opacity) | Alpha blending (50% transparan) + warna entry/exit berbeda |
| Load model | Hardcoded: `YOLO('head.pt')` | `get_resource_path()` support PyInstaller |
| Error handling | Silent fail | Frame error "KAMERA TERPUTUS" / "CAMERA ERROR" |
| Shutdown | Tidak ada | `/shutdown` POST endpoint + `processor.release_resources()` |
| Browser launch | Tidak ada | Auto-launch Chrome/Edge App Mode fullscreen |
| Konfigurasi | Semua hardcoded di `app.py` | `config.json` + web UI + atomic save via `config_writer.py` |
| Reset count | Reset ke 0 semua (entry, exit, current) | Reset + initial values dari config |
| Logging | Tidak ada status messages | Detail log di console (model load, kamera init, recording) |

### Frontend (HTML/JS)

| Aspek | Original (Adib) | Perbaikan |
|---|---|---|
| CSS/JS | Tailwind CDN + Chart.js CDN | Semua lokal (`tailwind.min.css`, `chart.js`, `script.js`) |
| Grafik | Tidak ada yang berfungsi | Grafik real-time dengan Chart.js |
| Recording controls | Tidak ada | Tombol Start/Stop recording + status indikator |
| Reset | Hanya reset angka | Reset + clear grafik |
| Layout | Entry/Exit di footer bawah | Tab layout dengan grafik dan count terintegrasi |
| Halaman | Hanya `index.html` | 4 halaman (index, konfigurasi, petunjuk, tentang) |
| Konfigurasi | Tidak ada | Form `/konfigurasi` untuk edit versi, posisi garis, swap direction, dll |
| Navbar | Simple link | Navbar lengkap dengan hover effects |

---

## C. Infrastruktur & Deployment (Tidak Ada di Versi Original)

| Fitur | Keterangan |
|---|---|
| **Restruktur proyek** | `src/headcounter/` dan `src/movement_counter/` sebagai dua aplikasi terpisah |
| **Satu `requirements.txt`** | Gabungan dependensi kedua aplikasi di root |
| **`run_headcounter.ps1` / `run_movcounter.ps1`** | Script PowerShell dengan auto-aktivasi venv + pause setelah stop |
| **`create_shortcut.ps1`** | Membuat 2 shortcut desktop (Head Counter + Movement Counter) dengan icon |
| **`tools/create_ico.py`** | Konversi PNG ke ICO untuk shortcut |
| **Deskripsi shortcut** | Nama jelas + deskripsi tooltip di Windows |
| **Auto-venv** | Script `.ps1` otomatis deteksi dan aktivasi virtual environment |
| **`docs/`** | Dokumentasi lengkap: SETUP_ALAT.md, CARA_KERJA_PROGRAM.md, REFERENSI_PAPER.md, PERUBAHAN.md |
| **`README.md`** | Panduan lengkap dari setup hingga deployment |
| **`CHANGELOGS.md`** | Riwayat perubahan |

---

## D. Kelebihan Program Versi Baru

### 1. Modularitas
Kode dipisah berdasarkan tanggung jawab. Satu bug tidak merusak seluruh aplikasi. Mahasiswa baru bisa fokus mengubah satu file tanpa memahami seluruh codebase.

### 2. Konfigurasi Dinamis
Semua parameter penting (posisi garis, swap direction, threshold keramaian, log interval) bisa diubah via web UI tanpa sentuh kode. Konfigurasi disimpan atomically (temp file -> rename) untuk mencegah korupsi data.

### 3. Siap Deployment
- `path_resolver.py` mendukung PyInstaller build untuk distribusi `.exe`
- Shortcut desktop siap pakai via `create_shortcut.ps1`
- Script `.ps1` dengan auto-venv: mahasiswa baru tinggal double-click

### 4. Error Handling
- **Movement Counter**: Kamera terputus tampil frame error visual "KAMERA TERPUTUS", bukan crash
- **Head Counter**: Kamera terputus return HTTP 503 dengan pesan error
- **Movement Counter**: Model gagal load -> exit dengan pesan jelas, bukan `NoneType` error
- **Head Counter**: Shared state dilindungi `threading.Lock`

### 5. Dua Mode Garis (Movement Counter)
Vertical dan horizontal line counting dalam satu aplikasi. Ganti mode via dropdown di halaman konfigurasi + restart.

### 6. Recording + Export
- Recording video beranotasi untuk analisis offline
- Export Excel dengan timestamp lengkap (tanggal + waktu + entry + exit + current)

### 7. Dokumentasi Lengkap
Setup alat (hardware), cara kerja program (algoritma), referensi paper, riwayat perubahan, dan panduan deployment - semua dalam folder `docs/`.

### 8. UX Improvements
- Stabilization delay mencegah false positive saat kamera baru nyala
- Timestamp overlay untuk verifikasi waktu
- Terminal tetap terbuka setelah close untuk lihat log
- Browser auto-launch dalam App Mode fullscreen