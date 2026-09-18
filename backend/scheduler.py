import logging
import threading
from typing import Dict, Any, Optional
from backend.database import (
    get_configured_models, create_scan_run, finish_scan_run,
    upsert_vehicle, get_latest_scan_run
)
from backend.connector import AutobellConnector
from backend.normalizer import normalize_vehicle
from backend.analyzer import update_all_opportunities

logger = logging.getLogger("radar_autobell.scheduler")

# Global scan state for live UI tracking
CURRENT_SCAN_STATE = {
    "is_scanning": False,
    "progress_percent": 0,
    "current_model": "",
    "total_found": 0,
    "new_count": 0,
    "updated_count": 0,
    "price_drop_count": 0,
    "error_count": 0,
    "message": "En espera"
}
SCAN_LOCK = threading.Lock()

def run_full_scan(max_per_model: int = 1500, fetch_detail_photos: bool = False) -> Dict[str, Any]:
    """
    Executes a complete scan across all active configured models on Autobell Global.
    Can be run synchronously or in a background thread.
    """
    global CURRENT_SCAN_STATE
    with SCAN_LOCK:
        if CURRENT_SCAN_STATE["is_scanning"]:
            return {"status": "ALREADY_RUNNING", "state": CURRENT_SCAN_STATE}
        CURRENT_SCAN_STATE["is_scanning"] = True
        CURRENT_SCAN_STATE["progress_percent"] = 5
        CURRENT_SCAN_STATE["message"] = "Iniciando escaneo de Autobell Global..."
        CURRENT_SCAN_STATE["total_found"] = 0
        CURRENT_SCAN_STATE["new_count"] = 0
        CURRENT_SCAN_STATE["updated_count"] = 0
        CURRENT_SCAN_STATE["price_drop_count"] = 0
        CURRENT_SCAN_STATE["error_count"] = 0

    run_id = create_scan_run()
    connector = AutobellConnector()
    models = get_configured_models()

    total_found = 0
    new_count = 0
    updated_count = 0
    price_drop_count = 0
    error_count = 0
    error_logs = []

    try:
        num_models = len(models)
        for idx, m in enumerate(models):
            model_name = f"{m['make']} {m['model']}"
            CURRENT_SCAN_STATE["current_model"] = model_name
            CURRENT_SCAN_STATE["message"] = f"Buscando {model_name} en Autobell..."
            step_progress = int(10 + (idx / max(1, num_models)) * 75)
            CURRENT_SCAN_STATE["progress_percent"] = step_progress

            keyword = m.get("search_keyword") or m["model"]
            try:
                raw_cars = connector.fetch_all_for_model(keyword=keyword, max_items=max_per_model)
                total_found += len(raw_cars)
                CURRENT_SCAN_STATE["total_found"] = total_found

                for raw in raw_cars:
                    try:
                        norm = normalize_vehicle(raw)
                        # Optionally attach extra photos if requested and not present
                        if fetch_detail_photos and len(norm.get("photos", [])) < 3:
                            extra_photos = connector.get_vehicle_photos(norm["id"])
                            if extra_photos:
                                norm["raw_data"]["detail_images"] = extra_photos
                                norm = normalize_vehicle(norm["raw_data"])

                        result = upsert_vehicle(norm)
                        action = result.get("action")

                        if action == "NEW":
                            new_count += 1
                        elif action == "PRICE_DROP":
                            price_drop_count += 1
                            updated_count += 1
                        elif action == "PRICE_RISE":
                            updated_count += 1

                    except Exception as e:
                        error_count += 1
                        error_logs.append(f"Error normalizing vehicle: {e}")

                CURRENT_SCAN_STATE["new_count"] = new_count
                CURRENT_SCAN_STATE["updated_count"] = updated_count
                CURRENT_SCAN_STATE["price_drop_count"] = price_drop_count

            except Exception as e:
                error_count += 1
                error_logs.append(f"Error fetching model {model_name}: {e}")

        # Recalculate market intelligence and opportunities
        CURRENT_SCAN_STATE["message"] = "Calculando oportunidades y estadísticas de mercado..."
        CURRENT_SCAN_STATE["progress_percent"] = 90
        update_all_opportunities()

        status = "SUCCESS" if error_count == 0 else ("PARTIAL" if total_found > 0 else "FAILED")
        finish_scan_run(
            run_id=run_id,
            status=status,
            total_found=total_found,
            new_count=new_count,
            updated_count=updated_count,
            price_drop_count=price_drop_count,
            error_count=error_count,
            error_log="; ".join(error_logs[:10])
        )

        CURRENT_SCAN_STATE["progress_percent"] = 100
        CURRENT_SCAN_STATE["message"] = f"Escaneo completado: {total_found} vehículos analizados ({new_count} nuevos, {price_drop_count} bajas de precio)"
        return {
            "status": status,
            "run_id": run_id,
            "total_found": total_found,
            "new_count": new_count,
            "updated_count": updated_count,
            "price_drop_count": price_drop_count,
            "error_count": error_count
        }

    except Exception as e:
        logger.exception(f"Fatal error during scan: {e}")
        finish_scan_run(
            run_id=run_id,
            status="FAILED",
            total_found=total_found,
            new_count=new_count,
            updated_count=updated_count,
            price_drop_count=price_drop_count,
            error_count=error_count + 1,
            error_log=str(e)
        )
        CURRENT_SCAN_STATE["message"] = f"Error en escaneo: {e}"
        return {"status": "FAILED", "error": str(e)}

    finally:
        with SCAN_LOCK:
            CURRENT_SCAN_STATE["is_scanning"] = False

def start_scan_in_background(max_per_model: int = 2000, fetch_detail_photos: bool = False):
    """Launches scan in background thread."""
    t = threading.Thread(target=run_full_scan, kwargs={"max_per_model": max_per_model, "fetch_detail_photos": fetch_detail_photos}, daemon=True)
    t.start()
    return {"status": "STARTED"}

def start_auto_scheduler(interval_minutes: int = 60):
    """
    Starts an automatic background loop that scans vehicles on startup
    and refreshes the catalog periodically every interval_minutes.
    """
    def _loop():
        import time
        from backend.database import get_dashboard_stats
        time.sleep(3)
        try:
            stats = get_dashboard_stats()
            # If database is empty or has few items, scan immediately
            if stats.get("total_active", 0) < 500:
                logger.info("Auto-scheduler: Starting initial full scan on startup...")
                run_full_scan(max_per_model=2000)
        except Exception as e:
            logger.error(f"Auto-scheduler initial scan error: {e}")

        while True:
            time.sleep(interval_minutes * 60)
            try:
                logger.info("Auto-scheduler: Starting periodic catalog update...")
                run_full_scan(max_per_model=2000)
            except Exception as e:
                logger.error(f"Auto-scheduler periodic scan error: {e}")

    t = threading.Thread(target=_loop, daemon=True, name="auto_scheduler_thread")
    t.start()
