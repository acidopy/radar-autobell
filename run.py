#!/usr/bin/env python3
import os
import sys
import uvicorn

if __name__ == "__main__":
    # Ensure current directory is on python path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, curr_dir)

    print("=" * 60)
    print("  🚀 INICIANDO RADAR AUTOBELL - SISTEMA DE MONITOREO")
    print("  Modelos: Kia Sportage, Sorento | Hyundai Tucson, Santa Fe")
    print("  Servidor en: http://localhost:8000")
    print("=" * 60)

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
