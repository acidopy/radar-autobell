import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from backend.database import (
    init_db, query_vehicles, get_vehicle_by_id, update_vehicle_photos,
    get_setting, set_setting, delete_setting,
    set_favorite, remove_favorite, get_favorites,
    save_search, get_saved_searches, delete_saved_search,
    get_dashboard_stats, get_configured_models, add_configured_model
)
from backend.connector import AutobellConnector
from backend.scheduler import (
    run_full_scan, start_scan_in_background, start_auto_scheduler, CURRENT_SCAN_STATE
)

# Initialize database
init_db()

# Start background auto-scheduler (initial scan if empty + periodic updates)
start_auto_scheduler(interval_minutes=60)

app = FastAPI(
    title="RADAR AUTOBELL API",
    description="Motor de búsqueda, comparación y monitoreo de SUVs coreanas en Autobell Global",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Request Models
class FavoriteRequest(BaseModel):
    tag: str = "guardado"
    notes: str = ""

class SavedSearchRequest(BaseModel):
    name: str
    filters: Dict[str, Any]

class CompareRequest(BaseModel):
    ids: List[str]

class AddModelRequest(BaseModel):
    make: str
    model: str
    search_keyword: str


class ConnectAuthRequest(BaseModel):
    token: str
    user_key: Optional[str] = ""
    member_code: Optional[str] = ""


# ── Image proxy with logo sanitization ──────────────────────────────────────
_img_cache: dict = {}

@app.get("/api/img")
async def proxy_image(url: str = Query(...), v: Optional[str] = Query(None)):
    """Fetch an Autobell vehicle photo, blur supplier logos, and return the result."""
    # Bump the sanitizer cache version when mask geometry changes so existing
    # in-memory entries cannot keep serving images with the old masks.
    cache_key = f"{url}_{v or 'v23'}"
    if cache_key in _img_cache:
        return Response(content=_img_cache[cache_key], media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=0, must-revalidate"})
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers={"Referer": "https://www.autobellglobal.com/"})
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="upstream error")
        content = r.content
        try:
            from backend.image_sanitizer import sanitize_image
            content = sanitize_image(content)
        except Exception:
            pass  # Fallback si falla
        if len(_img_cache) < 500:
            _img_cache[cache_key] = content
        return Response(content=content, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=0, must-revalidate"})
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/auth/status")
def get_auth_status():
    """Returns the status of the connected Autobell account."""
    token = get_setting("autobell_access_token")
    user_key = get_setting("autobell_user_key", "")
    member_code = get_setting("autobell_member_code", "")

    if token:
        preview = token[:8] + "..." + token[-6:] if len(token) > 14 else "***"
        return {
            "is_connected": True,
            "user_key": user_key,
            "member_code": member_code,
            "token_preview": preview
        }
    return {
        "is_connected": False,
        "user_key": "",
        "member_code": "",
        "token_preview": ""
    }

@app.post("/api/auth/connect")
def connect_auth(payload: ConnectAuthRequest):
    """Validates and saves the user's logged-in Autobell session token."""
    raw_token = payload.token.strip()

    # If user pasted the whole Vuex JSON string, parse it automatically!
    if "{" in raw_token and "auth" in raw_token:
        try:
            parsed = json.loads(raw_token)
            if "auth" in parsed:
                auth_obj = parsed["auth"]
                headers = auth_obj.get("authHeaders", {})
                raw_token = headers.get("accessToken") or raw_token
                payload.user_key = headers.get("userKey") or payload.user_key
                payload.member_code = auth_obj.get("datas", {}).get("memberCode") or payload.member_code
        except Exception:
            pass

    if not raw_token:
        raise HTTPException(status_code=400, detail="Token no proporcionado o inválido.")

    # Test token against Autobell
    connector = AutobellConnector()
    test_res = connector.test_token(raw_token)
    if not test_res.get("valid"):
        raise HTTPException(status_code=400, detail=f"No se pudo validar el token con Autobell Global: {test_res.get('error')}")

    set_setting("autobell_access_token", raw_token)
    if payload.user_key:
        set_setting("autobell_user_key", payload.user_key)
    if payload.member_code:
        set_setting("autobell_member_code", payload.member_code)

    return {
        "status": "SUCCESS",
        "message": "Cuenta de Autobell Global vinculada exitosamente.",
        "user_key": payload.user_key,
        "member_code": payload.member_code
    }

@app.delete("/api/auth/disconnect")
def disconnect_auth():
    """Disconnects the Autobell account session."""
    delete_setting("autobell_access_token")
    delete_setting("autobell_user_key")
    delete_setting("autobell_member_code")
    return {"status": "SUCCESS", "message": "Cuenta desvinculada."}

@app.get("/api/stats")
def get_stats():
    """Returns dashboard global counters and model market statistics."""
    stats = get_dashboard_stats()
    stats["current_scan"] = CURRENT_SCAN_STATE
    return stats

@app.get("/api/vehicles")
def list_vehicles(
    model: Optional[str] = None,
    make: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    mileage_max: Optional[int] = None,
    displacement_min: Optional[int] = None,
    displacement_max: Optional[int] = None,
    fuel: Optional[str] = None,
    transmission: Optional[str] = None,
    drivetrain: Optional[str] = None,
    sunroof: Optional[int] = None,
    inspected: Optional[int] = None,
    opportunity_only: Optional[bool] = False,
    favorites_only: Optional[bool] = False,
    status: Optional[str] = "ACTIVE",
    sort_by: Optional[str] = "best_deal",
    q: Optional[str] = None,
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    Search and filter normalized vehicles with multi-criteria sorting.
    """
    results = query_vehicles(
        model=model,
        make=make,
        year_min=year_min,
        year_max=year_max,
        price_min=price_min,
        price_max=price_max,
        mileage_max=mileage_max,
        displacement_min=displacement_min,
        displacement_max=displacement_max,
        fuel=fuel,
        transmission=transmission,
        drivetrain=drivetrain,
        sunroof=sunroof,
        inspected=inspected,
        opportunity_only=opportunity_only,
        favorites_only=favorites_only,
        status=status,
        sort_by=sort_by,
        search_query=q,
        limit=limit,
        offset=offset
    )
    return results

@app.get("/api/vehicles/{vehicle_id}")
def get_vehicle(vehicle_id: str):
    """Returns complete vehicle details, options, and price history."""
    vehicle = get_vehicle_by_id(vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehículo con ID {vehicle_id} no encontrado.")
    
    # On demand: If fewer than 3 photos, fetch all detail photos from supplier
    existing_photos = vehicle.get("photos") or []
    if len(existing_photos) < 3:
        try:
            connector = AutobellConnector()
            raw_photos = connector.get_vehicle_photos(vehicle_id)
            new_photos = list(existing_photos)
            for item in raw_photos:
                if isinstance(item, dict):
                    # extract resource or thumbnailResource
                    rel = item.get("resource") or item.get("thumbnailResource", {}).get("LARGE") or item.get("thumbnailResource", {}).get("MEDIUM")
                    if rel:
                        url = rel if rel.startswith("http") else f"https://cdn.autobellglobal.com/images/{rel.lstrip('/')}"
                        if url not in new_photos:
                            new_photos.append(url)
                elif isinstance(item, str):
                    url = item if item.startswith("http") else f"https://cdn.autobellglobal.com/images/{item.lstrip('/')}"
                    if url not in new_photos:
                        new_photos.append(url)
            if len(new_photos) > len(existing_photos):
                update_vehicle_photos(vehicle_id, new_photos)
                vehicle["photos"] = new_photos
                if new_photos:
                    vehicle["thumbnail_url"] = new_photos[0]
        except Exception as e:
            pass

    return vehicle

@app.post("/api/scan")
def trigger_scan(max_per_model: int = 2000, sync: bool = False, background_tasks: BackgroundTasks = None):
    """
    Triggers 'BUSCAR NUEVOS ANUNCIOS' across all configured models.
    """
    if CURRENT_SCAN_STATE["is_scanning"]:
        return {
            "status": "ALREADY_RUNNING",
            "message": "Un escaneo ya está en progreso.",
            "state": CURRENT_SCAN_STATE
        }

    if sync:
        res = run_full_scan(max_per_model=max_per_model)
        return res
    else:
        start_scan_in_background(max_per_model=max_per_model)
        return {
            "status": "STARTED",
            "message": "Escaneo iniciado en segundo plano.",
            "state": CURRENT_SCAN_STATE
        }

@app.get("/api/scan/status")
def get_scan_status():
    """Returns the live progress and status of the current or latest scan."""
    return CURRENT_SCAN_STATE

@app.get("/api/models")
def get_models():
    """Returns currently configured models to monitor."""
    return get_configured_models()

@app.post("/api/models")
def create_model(payload: AddModelRequest):
    """Allows adding a new brand/model to the radar."""
    add_configured_model(
        make=payload.make,
        model=payload.model,
        search_keyword=payload.search_keyword
    )
    return {"status": "SUCCESS", "message": f"Modelo {payload.make} {payload.model} configurado."}

@app.get("/api/favorites")
def list_favorites():
    """Returns all saved vehicles in 'MI LISTA'."""
    return get_favorites()

@app.post("/api/favorites/{vehicle_id}")
def add_favorite(vehicle_id: str, payload: FavoriteRequest):
    """Adds or updates a vehicle in 'MI LISTA'."""
    set_favorite(vehicle_id, tag=payload.tag, notes=payload.notes)
    return {"status": "SUCCESS", "vehicle_id": vehicle_id}

@app.patch("/api/favorites/{vehicle_id}")
def update_favorite(vehicle_id: str, payload: FavoriteRequest):
    """Updates tag status and private notes for a favorite."""
    set_favorite(vehicle_id, tag=payload.tag, notes=payload.notes)
    return {"status": "SUCCESS", "vehicle_id": vehicle_id}

@app.delete("/api/favorites/{vehicle_id}")
def delete_favorite(vehicle_id: str):
    """Removes a vehicle from 'MI LISTA'."""
    remove_favorite(vehicle_id)
    return {"status": "SUCCESS", "vehicle_id": vehicle_id}

@app.get("/api/saved-searches")
def list_saved_searches():
    """Returns user's saved searches."""
    return get_saved_searches()

@app.post("/api/saved-searches")
def create_saved_search(payload: SavedSearchRequest):
    """Saves a search filter preset."""
    search_id = save_search(name=payload.name, filters=payload.filters)
    return {"status": "SUCCESS", "id": search_id}

@app.delete("/api/saved-searches/{search_id}")
def remove_saved_search(search_id: int):
    """Deletes a saved search."""
    delete_saved_search(search_id)
    return {"status": "SUCCESS", "id": search_id}

@app.post("/api/compare")
def compare_vehicles(payload: CompareRequest):
    """
    Compares 2 to 4 vehicles side-by-side.
    """
    if len(payload.ids) < 2 or len(payload.ids) > 4:
        raise HTTPException(status_code=400, detail="Debe seleccionar entre 2 y 4 vehículos para comparar.")

    vehicles = []
    for vid in payload.ids:
        v = get_vehicle_by_id(vid)
        if v:
            vehicles.append(v)

    if len(vehicles) < 2:
        raise HTTPException(status_code=404, detail="No se encontraron suficientes vehículos válidos para comparar.")

    attributes = [
        {"key": "photo", "label": "Foto principal"},
        {"key": "name", "label": "Vehículo"},
        {"key": "price_usd", "label": "Precio USD"},
        {"key": "year", "label": "Año"},
        {"key": "month_year", "label": "Mes / Año Registro"},
        {"key": "mileage_km", "label": "Kilometraje"},
        {"key": "displacement_cc", "label": "Cilindrada (cc)"},
        {"key": "fuel", "label": "Combustible"},
        {"key": "transmission", "label": "Transmisión"},
        {"key": "drivetrain", "label": "Tracción"},
        {"key": "sunroof", "label": "Techo Solar / Panorámico"},
        {"key": "inspected", "label": "Inspección Técnica"},
        {"key": "accidents", "label": "Historial de Daños"},
        {"key": "color", "label": "Color"},
        {"key": "market_score", "label": "Relación Precio/Año/Km"},
        {"key": "opportunity_flag", "label": "¿Oportunidad Detectada?"},
        {"key": "opportunity_reason", "label": "Razón de Oportunidad"},
        {"key": "seller", "label": "Vendedor"},
        {"key": "url", "label": "Enlace Autobell"}
    ]

    return {
        "vehicles": vehicles,
        "attributes": attributes
    }

# Mount frontend static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/{catchall:path}")
def serve_frontend(catchall: str):
    file_path = FRONTEND_DIR / catchall
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(FRONTEND_DIR / "index.html")
