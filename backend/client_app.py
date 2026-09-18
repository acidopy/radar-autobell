import os
import json
import base64
import time
import hmac
import hashlib
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

from fastapi import FastAPI, Query, HTTPException, Response, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.database import query_vehicles, get_vehicle_by_id
from backend.config import PARTNER_USERS, PARTNER_SECRET_KEY
from backend.image_sanitizer import sanitize_image

app = FastAPI(
    title="BUSCADOR DE SUV KOREA PARAGUAY",
    description="Catálogo de SUVs de Alta Gama Importadas Directamente desde Corea del Sur",
    version="1.0.0"
)

def create_partner_token(username: str) -> str:
    ts = int(time.time())
    msg = f"{username}:{ts}"
    sig = hmac.new(PARTNER_SECRET_KEY.encode('utf-8'), msg.encode('utf-8'), hashlib.sha256).hexdigest()
    return f"{username}:{ts}:{sig}"

def verify_partner_token(token: str) -> Optional[str]:
    if not token:
        return None
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return None
        username, ts_str, sig = parts
        msg = f"{username}:{ts_str}"
        expected_sig = hmac.new(PARTNER_SECRET_KEY.encode('utf-8'), msg.encode('utf-8'), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        ts = int(ts_str)
        if time.time() - ts > 30 * 86400:
            return None
        return username
    except Exception:
        return None

def is_partner_request(authorization: Optional[str]) -> bool:
    if not authorization or not authorization.startswith("Bearer "):
        return False
    token = authorization.split(" ", 1)[1].strip()
    return bool(verify_partner_token(token))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PORTAL_DIR = Path(__file__).resolve().parent.parent / "client_portal"

# In-memory image cache for high speed
IMAGE_CACHE: Dict[str, bytes] = {}

def proxy_url(raw_url: str) -> str:
    """Encodes raw image URL to prevent supplier exposure."""
    if not raw_url:
        return ""
    b64 = base64.urlsafe_b64encode(raw_url.encode("utf-8")).decode("utf-8")
    # Version the proxy URL so browsers do not reuse pre-sanitization images.
    return f"/api/client/image?img={b64}&sanitize=v4"

def clean_vehicle_for_client(v: Dict[str, Any], is_partner: bool = False) -> Dict[str, Any]:
    """Removes all internal supplier mentions and prepares customer-ready data with pricing logic.
    If is_partner is True, unlocks source cost and margin for commercial partners (Samuel).
    """
    ref_code = f"KP-{v['id']}"

    # Proxied photos
    clean_photos = []
    raw_photos = v.get("photos") or []
    if isinstance(raw_photos, str):
        try: raw_photos = json.loads(raw_photos)
        except: raw_photos = []

    for p in raw_photos:
        if isinstance(p, str) and p.startswith("http"):
            clean_photos.append(proxy_url(p))

    thumb = clean_photos[0] if clean_photos else ""

    # Pricing logic:
    # precio_vehiculo_cliente = precio_fuente + 800
    # RoRo: USD 2.800
    # Cigüeña en Paraguay: ≈ USD 700
    # total_estimado = precio_vehiculo_cliente + 2800 + 700
    raw_price = v.get("price_usd") or 0
    if raw_price > 0:
        precio_vehiculo_cliente = raw_price + 800
        roro_usd = 2800
        ciguena_usd = 700
        total_estimado = precio_vehiculo_cliente + roro_usd + ciguena_usd
    else:
        precio_vehiculo_cliente = 0
        roro_usd = 2800
        ciguena_usd = 700
        total_estimado = 0

    ciguena_nota = "Valor aproximado. Se abona en Paraguay y puede variar según la cotización vigente al momento de llegada del vehículo."

    # Message for WhatsApp consultation
    car_title = f"{v['make']} {v['model']} {v['year']}"
    precio_str = f"${precio_vehiculo_cliente:,} USD" if precio_vehiculo_cliente else "A consultar"
    total_str = f"≈ ${total_estimado:,} USD" if total_estimado else "A consultar"

    wa_msg = (
        f"Hola, me interesa la unidad {car_title} ({v.get('trim', '')}), Ref: #{ref_code}.\n"
        f"• Precio FOB: {precio_str}\n"
        f"• RoRo: $2,800 USD\n"
        f"• Cigüeña en Paraguay: ≈ $700 USD\n"
        f"• TOTAL ESTIMADO: {total_str}\n"
        f"Quisiera coordinar los detalles de la importación a Paraguay."
    )
    wa_link = f"https://wa.me/?text={urllib.parse.quote(wa_msg)}"

    vehicle_dict = {
        "id": v["id"],
        "ref_code": ref_code,
        "make": v["make"],
        "model": v["model"],
        "trim": v.get("trim", "Sin información"),
        "detail_model": v.get("detail_model", "Sin información"),
        "year": v.get("year", 0),
        "month_year": v.get("month_year", "Sin información"),
        "mileage_km": v.get("mileage_km", 0),
        "fuel": v.get("fuel", "Sin información"),
        "displacement_cc": v.get("displacement_cc", 0),
        "transmission": v.get("transmission", "Automática"),
        "drivetrain": v.get("drivetrain", "Sin información"),
        "color": v.get("color", "Sin información"),
        "precio_vehiculo_cliente": precio_vehiculo_cliente,
        "price_usd": precio_vehiculo_cliente,
        "roro_usd": roro_usd,
        "ciguena_usd": ciguena_usd,
        "ciguena_nota": ciguena_nota,
        "total_estimado": total_estimado,
        "currency": "USD",
        "photos": clean_photos,
        "thumbnail_url": thumb,
        "sunroof": v.get("sunroof", 0),
        "inspected": v.get("inspected", 1),
        "inspected_desc": "Certificado y verificado en Corea del Sur" if v.get("inspected") else "Inspección estándar",
        "seller": "Stock Verificado Corea (Exportación Directa)",
        "whatsapp_link": wa_link,
        "status": "DISPONIBLE EN COREA",
        "is_partner": is_partner
    }

    if is_partner:
        vehicle_dict["precio_fuente"] = raw_price
        vehicle_dict["cargo_margen"] = 800
        # The original supplier listing is available only in the partner portal.
        vehicle_dict["autobell_url"] = v.get("url", "")

    return vehicle_dict

class PartnerLoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/client/auth/login")
def partner_login(req: PartnerLoginRequest):
    """Authenticate commercial partner (Samuel) or admin."""
    user = req.username.strip().lower()
    pwd = req.password.strip()
    if user in PARTNER_USERS and PARTNER_USERS[user] == pwd:
        token = create_partner_token(user)
        return {
            "success": True,
            "token": token,
            "username": user,
            "role": "PARTNER",
            "displayName": "Samuel (Socio Comercial)" if user == "samuel" else "Administrador"
        }
    raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos.")

@app.get("/api/client/auth/me")
def partner_me(authorization: Optional[str] = Header(None)):
    """Check active partner session."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        user = verify_partner_token(token)
        if user:
            return {
                "authenticated": True,
                "username": user,
                "role": "PARTNER",
                "displayName": "Samuel (Socio Comercial)" if user == "samuel" else "Administrador"
            }
    return {"authenticated": False, "role": "PUBLIC"}

class ClientCompareRequest(BaseModel):
    ids: List[str]

@app.get("/api/client/vehicles")
def get_client_vehicles(
    model: Optional[str] = None,
    make: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    mileage_max: Optional[int] = None,
    fuel: Optional[str] = None,
    drivetrain: Optional[str] = None,
    sunroof: Optional[int] = None,
    sort_by: Optional[str] = "price_asc",
    q: Optional[str] = None,
    limit: int = Query(default=24, ge=1, le=60),
    offset: int = Query(default=0, ge=0),
    authorization: Optional[str] = Header(None)
):
    """Returns vehicles filtered and sanitized for client display."""
    partner = is_partner_request(authorization)
    db_price_min = max(0, price_min - 800) if price_min is not None else None
    db_price_max = max(0, price_max - 800) if price_max is not None else None

    res = query_vehicles(
        model=model,
        make=make,
        year_min=year_min,
        year_max=year_max,
        price_min=db_price_min,
        price_max=db_price_max,
        mileage_max=mileage_max,
        fuel=fuel,
        drivetrain=drivetrain,
        sunroof=sunroof,
        status="ACTIVE",
        sort_by=sort_by,
        search_query=q,
        limit=limit,
        offset=offset
    )

    clean_items = [clean_vehicle_for_client(item, is_partner=partner) for item in res["items"]]
    return {"total": res["total"], "items": clean_items, "is_partner": partner}

@app.get("/api/client/vehicles/{vehicle_id}")
def get_client_vehicle_detail(vehicle_id: str, authorization: Optional[str] = Header(None)):
    """Returns single vehicle sanitized details."""
    partner = is_partner_request(authorization)
    v = get_vehicle_by_id(vehicle_id)
    if not v or v.get("status") != "ACTIVE":
        raise HTTPException(status_code=404, detail="Vehículo no disponible.")
    return clean_vehicle_for_client(v, is_partner=partner)

@app.get("/api/client/image")
def proxy_image(img: str):
    """Proxies, sanitizes, and caches images for the client portal."""
    try:
        raw_url = base64.urlsafe_b64decode(img.encode("utf-8")).decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image token")

    if not (raw_url.startswith("http://") or raw_url.startswith("https://")):
        raise HTTPException(status_code=400, detail="Invalid protocol")

    # Check memory cache
    if raw_url in IMAGE_CACHE:
        return Response(content=IMAGE_CACHE[raw_url], media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    try:
        req = urllib.request.Request(raw_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = sanitize_image(resp.read())
            if len(IMAGE_CACHE) < 500:
                IMAGE_CACHE[raw_url] = content
            return Response(content=content, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
    except Exception as e:
        raise HTTPException(status_code=404, detail="Image not accessible")

@app.post("/api/client/compare")
def compare_client_vehicles(payload: ClientCompareRequest, authorization: Optional[str] = Header(None)):
    """Compares 2 to 4 vehicles for the client."""
    partner = is_partner_request(authorization)
    if len(payload.ids) < 2 or len(payload.ids) > 4:
        raise HTTPException(status_code=400, detail="Selecciona entre 2 y 4 vehículos para comparar.")

    clean_vehicles = []
    for vid in payload.ids:
        v = get_vehicle_by_id(vid)
        if v and v.get("status") == "ACTIVE":
            clean_vehicles.append(clean_vehicle_for_client(v, is_partner=partner))

    if len(clean_vehicles) < 2:
        raise HTTPException(status_code=404, detail="No se encontraron suficientes vehículos válidos.")

    attributes = [
        {"key": "photo", "label": "Foto"},
        {"key": "name", "label": "Vehículo"}
    ]

    if partner:
        attributes.extend([
            {"key": "precio_fuente", "label": "Costo Real (Fuente)"},
            {"key": "cargo_margen", "label": "Margen Agregado (+)"}
        ])

    attributes.extend([
        {"key": "precio_vehiculo_cliente", "label": "Precio FOB"},
        {"key": "roro_usd", "label": "RoRo"},
        {"key": "ciguena_usd", "label": "Cigüeña en Paraguay"},
        {"key": "total_estimado", "label": "TOTAL ESTIMADO"},
        {"key": "year", "label": "Año Modelo"},
        {"key": "mileage_km", "label": "Kilometraje"},
        {"key": "fuel", "label": "Combustible"},
        {"key": "displacement_cc", "label": "Cilindrada (Motor)"},
        {"key": "transmission", "label": "Transmisión"},
        {"key": "drivetrain", "label": "Tracción"},
        {"key": "sunroof", "label": "Techo Solar / Panorámico"},
        {"key": "inspected_desc", "label": "Certificación Técnica"},
        {"key": "color", "label": "Color"},
        {"key": "seller", "label": "Origen del Vehículo"},
        {"key": "ref_code", "label": "Código de Referencia"}
    ])

    return {"vehicles": clean_vehicles, "attributes": attributes, "is_partner": partner}

# Static client files
app.mount("/static", StaticFiles(directory=str(PORTAL_DIR)), name="static")

@app.get("/")
def serve_client_index():
    return FileResponse(PORTAL_DIR / "index.html")

@app.get("/{catchall:path}")
def serve_client_files(catchall: str):
    file_path = PORTAL_DIR / catchall
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(PORTAL_DIR / "index.html")
