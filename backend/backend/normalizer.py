import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from backend.config import AUTOBELL_BASE_URL, AUTOBELL_IMAGE_BASE

def normalize_vehicle(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes an item returned by Autobell Global search or detail API.
    Ensures that any missing field explicitly returns 'Sin información'.
    """
    # Key / ID
    car_key = raw.get("carKey") or raw.get("itemKey") or ""
    if not car_key:
        raise ValueError("Vehicle data missing carKey")

    url = f"{AUTOBELL_BASE_URL}/usedcar/info/{car_key}"

    # Make & Model
    make = (raw.get("makerName") or raw.get("maker_name") or "").strip().upper()
    model = (raw.get("modelName") or raw.get("model_name") or "").strip()
    detail_model = (raw.get("detailModelName") or raw.get("modelDetailName") or "").strip()
    grade = (raw.get("gradeName") or "").strip()
    car_name = (raw.get("carName") or "").strip()

    if not make:
        car_name_upper = car_name.upper()
        if "KIA" in car_name_upper:
            make = "KIA"
        elif "HYUNDAI" in car_name_upper:
            make = "HYUNDAI"
        else:
            make = "Sin información"

    if not model:
        car_name_upper = car_name.upper()
        if "SPORTAGE" in car_name_upper:
            model = "Sportage"
        elif "SORENTO" in car_name_upper:
            model = "Sorento"
        elif "TUCSON" in car_name_upper:
            model = "Tucson"
        elif "SANTA FE" in car_name_upper or "SANTAFE" in car_name_upper:
            model = "Santa Fe"
        else:
            model = "Sin información"

    trim = grade or detail_model or car_name or "Sin información"

    # Year & Month/Year
    year = raw.get("year")
    try:
        year = int(year) if year else 0
    except (ValueError, TypeError):
        year = 0

    month_year = "Sin información"
    first_reg_date = raw.get("firstRegDate")
    if first_reg_date:
        try:
            # timestamp in milliseconds
            dt = datetime.utcfromtimestamp(int(first_reg_date) / 1000)
            month_year = dt.strftime("%m/%Y")
        except Exception:
            month_year = str(year) if year > 0 else "Sin información"
    elif year > 0:
        month_year = str(year)

    # Mileage
    mileage = raw.get("mileage")
    try:
        mileage_km = int(mileage) if mileage is not None else 0
    except (ValueError, TypeError):
        mileage_km = 0

    # Fuel
    fuel = raw.get("carFuelName") or raw.get("fuelTypeName") or ""
    if not fuel:
        fuel = "Sin información"

    # Displacement cc
    cc = raw.get("displacementVolume") or raw.get("displacement")
    try:
        displacement_cc = int(cc) if cc else 0
    except (ValueError, TypeError):
        displacement_cc = 0

    # Transmission
    trans = raw.get("transmissionName") or raw.get("gearBoxName") or ""
    if not trans:
        trans = "Sin información"

    # Drivetrain (2WD, 4WD, AWD)
    combined_specs = f"{car_name} {trim} {detail_model}".upper()
    if "4WD" in combined_specs or "AWD" in combined_specs:
        drivetrain = "4WD"
    elif "2WD" in combined_specs:
        drivetrain = "2WD"
    else:
        drivetrain = "Sin información"

    # Color
    color = raw.get("colorName") or ""
    if not color:
        color = "Sin información"

    # Price USD
    price = raw.get("salePrice")
    try:
        price_usd = int(price) if price is not None else 0
    except (ValueError, TypeError):
        price_usd = 0

    currency = "USD"

    # Seller
    seller = ""
    if raw.get("autobellStock") is True:
        seller = "Autobell Oficial Stock"
    elif raw.get("dealerKey"):
        seller = f"Dealer ({raw.get('dealerKey')})"
    elif raw.get("stockType"):
        seller = f"Autobell Stock ({raw.get('stockType')})"
    else:
        seller = "Sin información"

    # Status
    raw_status = (raw.get("carStatus") or raw.get("businessStatus") or "").upper()
    if raw_status in ["PURCHASE_APPLY", "ACTIVE", "NORMAL"]:
        status = "ACTIVE"
    elif raw_status in ["SOLD", "FINISH", "CANCEL"]:
        status = "SOLD_OR_REMOVED"
    elif raw_status:
        status = raw_status
    else:
        status = "ACTIVE"

    # Photos
    photos: List[str] = []
    # From thumbnailResource
    for res_key in ["thumbnailResource", "thumbnailResource2"]:
        thumb_obj = raw.get(res_key)
        if isinstance(thumb_obj, dict):
            # prefer LARGE, then MEDIUM, then THUMB
            for size_key in ["LARGE", "MEDIUM", "THUMB"]:
                rel_path = thumb_obj.get(size_key)
                if rel_path:
                    img_url = rel_path if rel_path.startswith("http") else f"{AUTOBELL_IMAGE_BASE}/{rel_path.lstrip('/')}"
                    if img_url not in photos:
                        photos.append(img_url)

    # From car_detail_images if present
    for img_item in raw.get("detail_images", []):
        if isinstance(img_item, dict):
            rel = img_item.get("resource") or img_item.get("thumbnailResource", {}).get("LARGE")
            if rel:
                img_url = rel if rel.startswith("http") else f"{AUTOBELL_IMAGE_BASE}/{rel.lstrip('/')}"
                if img_url not in photos:
                    photos.append(img_url)
        elif isinstance(img_item, str):
            img_url = img_item if img_item.startswith("http") else f"{AUTOBELL_IMAGE_BASE}/{img_item.lstrip('/')}"
            if img_url not in photos:
                photos.append(img_url)

    thumbnail_url = photos[0] if photos else ""

    # Sunroof detection from option lists
    sunroof = 0
    all_options_text = []

    for opt_group in ["exteriorOptionData", "interiorOptionData", "safetyOptionData", "multimediaOptionData", "etcOptionData"]:
        items = raw.get(opt_group, [])
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    c2 = (it.get("code2EngName") or "").lower()
                    c3 = (it.get("code3EngName") or "").lower()
                    full = f"{c2} {c3}"
                    all_options_text.append(full)
                    if "sunroof" in full or "panoramic" in full or "roof" in c2:
                        if "sunroof" in full:
                            sunroof = 1

    # Also check carName/trim for sunroof mentions
    if "SUNROOF" in combined_specs or "PANORAMA" in combined_specs:
        sunroof = 1

    # Accidents and inspection
    inspected = 1 if raw.get("inspected") is True else 0
    accidents = "Sin información"
    if raw.get("accidentYn") is not None:
        accidents = "Con accidentes previos" if raw.get("accidentYn") == "Y" else "Sin accidentes reportados"
    elif inspected == 1:
        accidents = "Inspeccionado por Autobell (Ver informe en sitio)"
    else:
        accidents = "Sin inspección registrada"

    options_summary = {
        "exterior": [it.get("code2EngName", "") for it in raw.get("exteriorOptionData", []) if isinstance(it, dict)],
        "interior": [it.get("code2EngName", "") for it in raw.get("interiorOptionData", []) if isinstance(it, dict)],
        "safety": [it.get("code2EngName", "") for it in raw.get("safetyOptionData", []) if isinstance(it, dict)],
        "multimedia": [it.get("code2EngName", "") for it in raw.get("multimediaOptionData", []) if isinstance(it, dict)],
        "etc": [it.get("code2EngName", "") for it in raw.get("etcOptionData", []) if isinstance(it, dict)]
    }

    return {
        "id": car_key,
        "url": url,
        "make": make,
        "model": model,
        "trim": trim,
        "detail_model": detail_model or "Sin información",
        "year": year,
        "month_year": month_year,
        "mileage_km": mileage_km,
        "fuel": fuel,
        "displacement_cc": displacement_cc,
        "transmission": trans,
        "drivetrain": drivetrain,
        "color": color,
        "price_usd": price_usd,
        "currency": currency,
        "seller": seller,
        "status": status,
        "photos": photos,
        "thumbnail_url": thumbnail_url,
        "accidents": accidents,
        "inspected": inspected,
        "sunroof": sunroof,
        "options_json": options_summary,
        "raw_data": raw
    }
