import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = os.getenv("RADAR_DB_PATH", str(DATA_DIR / "radar_autobell.db"))

AUTOBELL_BASE_URL = "https://www.autobellglobal.com"
AUTOBELL_API_URL = "https://www.autobellglobal.com/api/glovis"
AUTOBELL_IMAGE_BASE = "https://cdn.autobellglobal.com/images"

# Default models to monitor
DEFAULT_MODELS = [
    {
        "make": "KIA",
        "model": "Sportage",
        "brand_code": 2,
        "car_model": 54,
        "search_keyword": "Sportage",
        "is_active": True
    },
    {
        "make": "KIA",
        "model": "Sorento",
        "brand_code": 2,
        "car_model": 56,
        "search_keyword": "Sorento",
        "is_active": True
    },
    {
        "make": "HYUNDAI",
        "model": "Tucson",
        "brand_code": 5,
        "car_model": 124,
        "search_keyword": "Tucson",
        "is_active": True
    },
    {
        "make": "HYUNDAI",
        "model": "Santa Fe",
        "brand_code": 5,
        "car_model": 110,
        "search_keyword": "Santa Fe",
        "is_active": True
    }
]

# Request headers for Autobell Global
AUTOBELL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.autobellglobal.com/usedcar/list",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8"
}

# Commercial Partner Credentials (for Samuel & Admin on client portal)
PARTNER_USERS = {
    "samuel": os.getenv("PARTNER_PASSWORD_SAMUEL", "socio2026"),
    "admin": os.getenv("PARTNER_PASSWORD_ADMIN", "admin2026")
}
PARTNER_SECRET_KEY = os.getenv("PARTNER_SECRET_KEY", "korea_paraguay_secret_auth_2026")

