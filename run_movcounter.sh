#!/bin/bash
# run_movcounter.sh
# Script untuk menjalankan STASRG Movement Counter di Linux
# Jalankan dengan: chmod +x run_movcounter.sh && ./run_movcounter.sh

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== STASRG Movement Counter ==="

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated."
else
    echo "WARNING: venv tidak ditemukan."
    echo "  Buat venv: python3 -m venv venv"
    echo "  Install dep: pip install -r requirements.txt"
fi

python3 src/movement_counter/app.py
echo "Movement Counter stopped."