import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.config import DB_PATH, DEFAULT_MODELS

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS configured_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        make TEXT NOT NULL,
        model TEXT NOT NULL,
        brand_code INTEGER,
        car_model INTEGER,
        search_keyword TEXT NOT NULL,
        is_active INTEGER DEFAULT 1,
        UNIQUE(make, model)
    );

    CREATE TABLE IF NOT EXISTS vehicles (
        id TEXT PRIMARY KEY,
        url TEXT NOT NULL,
        make TEXT NOT NULL,
        model TEXT NOT NULL,
        trim TEXT DEFAULT 'Sin información',
        detail_model TEXT DEFAULT 'Sin información',
        year INTEGER DEFAULT 0,
        month_year TEXT DEFAULT 'Sin información',
        mileage_km INTEGER DEFAULT 0,
        fuel TEXT DEFAULT 'Sin información',
        displacement_cc INTEGER DEFAULT 0,
        transmission TEXT DEFAULT 'Sin información',
        drivetrain TEXT DEFAULT 'Sin información',
        color TEXT DEFAULT 'Sin información',
        price_usd INTEGER DEFAULT 0,
        currency TEXT DEFAULT 'USD',
        seller TEXT DEFAULT 'Sin información',
        first_seen_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        status TEXT DEFAULT 'ACTIVE',
        photos TEXT DEFAULT '[]',
        thumbnail_url TEXT DEFAULT '',
        accidents TEXT DEFAULT 'Sin información',
        inspected INTEGER DEFAULT 0,
        sunroof INTEGER DEFAULT 0,
        options_json TEXT DEFAULT '{}',
        raw_data_json TEXT DEFAULT '{}',
        opportunity_flag INTEGER DEFAULT 0,
        opportunity_reason TEXT DEFAULT '',
        market_score REAL DEFAULT 0.0,
        price_change_status TEXT DEFAULT 'NEW',
        price_diff INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_vehicles_model_year ON vehicles(model, year);
    CREATE INDEX IF NOT EXISTS idx_vehicles_price ON vehicles(price_usd);
    CREATE INDEX IF NOT EXISTS idx_vehicles_mileage ON vehicles(mileage_km);
    CREATE INDEX IF NOT EXISTS idx_vehicles_status ON vehicles(status);
    CREATE INDEX IF NOT EXISTS idx_vehicles_opp ON vehicles(opportunity_flag);

    CREATE TABLE IF NOT EXISTS price_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_id TEXT NOT NULL,
        old_price INTEGER NOT NULL,
        new_price INTEGER NOT NULL,
        change_type TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        FOREIGN KEY(vehicle_id) REFERENCES vehicles(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_price_history_vehicle ON price_history(vehicle_id);

    CREATE TABLE IF NOT EXISTS favorites (
        vehicle_id TEXT PRIMARY KEY,
        tag TEXT DEFAULT 'guardado',
        notes TEXT DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(vehicle_id) REFERENCES vehicles(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS saved_searches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        filters_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    
    CREATE TABLE IF NOT EXISTS app_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    
    CREATE TABLE IF NOT EXISTS scan_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        status TEXT NOT NULL,
        total_found INTEGER DEFAULT 0,
        new_count INTEGER DEFAULT 0,
        updated_count INTEGER DEFAULT 0,
        price_drop_count INTEGER DEFAULT 0,
        error_count INTEGER DEFAULT 0,
        error_log TEXT DEFAULT ''
    );
    """)

    # Seed default models if table is empty
    cursor.execute("SELECT COUNT(*) as cnt FROM configured_models")
    if cursor.fetchone()["cnt"] == 0:
        for m in DEFAULT_MODELS:
            cursor.execute("""
                INSERT OR IGNORE INTO configured_models (make, model, brand_code, car_model, search_keyword, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (m["make"], m["model"], m.get("brand_code"), m.get("car_model"), m["search_keyword"], 1))

    conn.commit()
    conn.close()

def get_configured_models() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM configured_models WHERE is_active = 1")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def add_configured_model(make: str, model: str, search_keyword: str, brand_code: Optional[int] = None, car_model: Optional[int] = None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO configured_models (make, model, brand_code, car_model, search_keyword, is_active)
        VALUES (?, ?, ?, ?, ?, 1)
    """, (make.upper(), model, brand_code, car_model, search_keyword))
    conn.commit()
    conn.close()

def create_scan_run() -> int:
    conn = get_connection()
    cursor = conn.cursor()
    now_iso = datetime.utcnow().isoformat()
    cursor.execute("""
        INSERT INTO scan_runs (started_at, status)
        VALUES (?, 'RUNNING')
    """, (now_iso,))
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id

def finish_scan_run(run_id: int, status: str, total_found: int, new_count: int, updated_count: int, price_drop_count: int, error_count: int, error_log: str = ""):
    conn = get_connection()
    cursor = conn.cursor()
    now_iso = datetime.utcnow().isoformat()
    cursor.execute("""
        UPDATE scan_runs
        SET ended_at = ?, status = ?, total_found = ?, new_count = ?, updated_count = ?,
            price_drop_count = ?, error_count = ?, error_log = ?
        WHERE id = ?
    """, (now_iso, status, total_found, new_count, updated_count, price_drop_count, error_count, error_log, run_id))
    conn.commit()
    conn.close()

def get_latest_scan_run() -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scan_runs ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def upsert_vehicle(v: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upserts vehicle and detects:
    - NEW: first time seen
    - PRICE_DROP: price went down
    - PRICE_RISE: price went up
    - SAME: no price change
    Records in price_history when price changes.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now_iso = datetime.utcnow().isoformat()

    cursor.execute("SELECT * FROM vehicles WHERE id = ?", (v["id"],))
    existing = cursor.fetchone()

    action = "NEW"
    price_diff = 0

    if existing:
        old_price = existing["price_usd"]
        new_price = v["price_usd"]
        first_seen = existing["first_seen_at"]

        if new_price < old_price:
            action = "PRICE_DROP"
            price_diff = new_price - old_price
            cursor.execute("""
                INSERT INTO price_history (vehicle_id, old_price, new_price, change_type, recorded_at)
                VALUES (?, ?, ?, 'DROP', ?)
            """, (v["id"], old_price, new_price, now_iso))
        elif new_price > old_price:
            action = "PRICE_RISE"
            price_diff = new_price - old_price
            cursor.execute("""
                INSERT INTO price_history (vehicle_id, old_price, new_price, change_type, recorded_at)
                VALUES (?, ?, ?, 'RISE', ?)
            """, (v["id"], old_price, new_price, now_iso))
        else:
            action = existing["price_change_status"] or "SAME"
            price_diff = existing["price_diff"] or 0

        cursor.execute("""
            UPDATE vehicles
            SET url = ?, make = ?, model = ?, trim = ?, detail_model = ?, year = ?,
                month_year = ?, mileage_km = ?, fuel = ?, displacement_cc = ?,
                transmission = ?, drivetrain = ?, color = ?, price_usd = ?, currency = ?,
                seller = ?, last_seen_at = ?, status = 'ACTIVE', photos = ?, thumbnail_url = ?,
                accidents = ?, inspected = ?, sunroof = ?, options_json = ?, raw_data_json = ?,
                opportunity_flag = ?, opportunity_reason = ?, market_score = ?,
                price_change_status = ?, price_diff = ?
            WHERE id = ?
        """, (
            v["url"], v["make"], v["model"], v.get("trim", "Sin información"), v.get("detail_model", "Sin información"),
            v.get("year", 0), v.get("month_year", "Sin información"), v.get("mileage_km", 0), v.get("fuel", "Sin información"),
            v.get("displacement_cc", 0), v.get("transmission", "Sin información"), v.get("drivetrain", "Sin información"),
            v.get("color", "Sin información"), v.get("price_usd", 0), v.get("currency", "USD"), v.get("seller", "Sin información"),
            now_iso, json.dumps(v.get("photos", [])), v.get("thumbnail_url", ""), v.get("accidents", "Sin información"),
            v.get("inspected", 0), v.get("sunroof", 0), json.dumps(v.get("options_json", {})), json.dumps(v.get("raw_data", {})),
            v.get("opportunity_flag", 0), v.get("opportunity_reason", ""), v.get("market_score", 0.0),
            action, price_diff, v["id"]
        ))
    else:
        # First insertion
        action = "NEW"
        cursor.execute("""
            INSERT INTO vehicles (
                id, url, make, model, trim, detail_model, year, month_year, mileage_km,
                fuel, displacement_cc, transmission, drivetrain, color, price_usd, currency,
                seller, first_seen_at, last_seen_at, status, photos, thumbnail_url,
                accidents, inspected, sunroof, options_json, raw_data_json,
                opportunity_flag, opportunity_reason, market_score, price_change_status, price_diff
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, 'ACTIVE', ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
        """, (
            v["id"], v["url"], v["make"], v["model"], v.get("trim", "Sin información"), v.get("detail_model", "Sin información"),
            v.get("year", 0), v.get("month_year", "Sin información"), v.get("mileage_km", 0), v.get("fuel", "Sin información"),
            v.get("displacement_cc", 0), v.get("transmission", "Sin información"), v.get("drivetrain", "Sin información"),
            v.get("color", "Sin información"), v.get("price_usd", 0), v.get("currency", "USD"), v.get("seller", "Sin información"),
            now_iso, now_iso, json.dumps(v.get("photos", [])), v.get("thumbnail_url", ""), v.get("accidents", "Sin información"),
            v.get("inspected", 0), v.get("sunroof", 0), json.dumps(v.get("options_json", {})), json.dumps(v.get("raw_data", {})),
            v.get("opportunity_flag", 0), v.get("opportunity_reason", ""), v.get("market_score", 0.0),
            action, 0
        ))
        cursor.execute("""
            INSERT INTO price_history (vehicle_id, old_price, new_price, change_type, recorded_at)
            VALUES (?, ?, ?, 'INITIAL', ?)
        """, (v["id"], v.get("price_usd", 0), v.get("price_usd", 0), now_iso))

    conn.commit()
    conn.close()
    return {"action": action, "price_diff": price_diff}

def mark_missing_as_sold(active_keys: List[str], model: str):
    """Marks active vehicles for this model not present in the current scan as SOLD_OR_REMOVED."""
    if not active_keys:
        return
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join(["?"] * len(active_keys))
    cursor.execute(f"""
        UPDATE vehicles
        SET status = 'SOLD_OR_REMOVED'
        WHERE model = ? AND status = 'ACTIVE' AND id NOT IN ({placeholders})
    """, [model] + active_keys)
    conn.commit()
    conn.close()

def query_vehicles(
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
    search_query: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
        SELECT v.*,
               f.tag as favorite_tag,
               f.notes as favorite_notes,
               CASE WHEN f.vehicle_id IS NOT NULL THEN 1 ELSE 0 END as is_favorite
        FROM vehicles v
        LEFT JOIN favorites f ON v.id = f.vehicle_id
        WHERE 1=1
    """
    params = []

    if status and status != 'ALL':
        sql += " AND v.status = ?"
        params.append(status)

    if model:
        sql += " AND LOWER(v.model) = LOWER(?)"
        params.append(model)

    if make:
        sql += " AND LOWER(v.make) = LOWER(?)"
        params.append(make)

    if year_min is not None and year_min > 0:
        sql += " AND v.year >= ?"
        params.append(year_min)

    if year_max is not None and year_max > 0:
        sql += " AND v.year <= ?"
        params.append(year_max)

    if price_min is not None and price_min > 0:
        sql += " AND v.price_usd >= ?"
        params.append(price_min)

    if price_max is not None and price_max > 0:
        sql += " AND v.price_usd <= ?"
        params.append(price_max)

    if mileage_max is not None and mileage_max > 0:
        sql += " AND v.mileage_km <= ?"
        params.append(mileage_max)

    if displacement_min is not None and displacement_min > 0:
        sql += " AND v.displacement_cc >= ?"
        params.append(displacement_min)

    if displacement_max is not None and displacement_max > 0:
        sql += " AND v.displacement_cc <= ?"
        params.append(displacement_max)

    if fuel and fuel != 'ALL':
        sql += " AND LOWER(v.fuel) = LOWER(?)"
        params.append(fuel)

    if transmission and transmission != 'ALL':
        sql += " AND LOWER(v.transmission) LIKE LOWER(?)"
        params.append(f"%{transmission}%")

    if drivetrain and drivetrain != 'ALL':
        sql += " AND v.drivetrain = ?"
        params.append(drivetrain)

    if sunroof is not None:
        if sunroof == 1:
            sql += " AND v.sunroof = 1"
        elif sunroof == 0:
            sql += " AND (v.sunroof = 0 OR v.sunroof IS NULL)"

    if inspected is not None and inspected == 1:
        sql += " AND v.inspected = 1"

    if opportunity_only:
        sql += " AND v.opportunity_flag = 1"

    if favorites_only:
        sql += " AND f.vehicle_id IS NOT NULL"

    if search_query:
        sql += " AND (v.make LIKE ? OR v.model LIKE ? OR v.trim LIKE ? OR v.detail_model LIKE ? OR v.color LIKE ?)"
        q_pattern = f"%{search_query}%"
        params.extend([q_pattern, q_pattern, q_pattern, q_pattern, q_pattern])

    # Count total matching query
    count_sql = f"SELECT COUNT(*) as total FROM ({sql})"
    cursor.execute(count_sql, params)
    total = cursor.fetchone()["total"]

    # Sorting
    if sort_by == "best_deal":
        sql += " ORDER BY v.market_score DESC, v.price_usd ASC"
    elif sort_by == "price_asc":
        sql += " ORDER BY v.price_usd ASC"
    elif sort_by == "price_desc":
        sql += " ORDER BY v.price_usd DESC"
    elif sort_by == "year_desc":
        sql += " ORDER BY v.year DESC, v.price_usd ASC"
    elif sort_by == "mileage_asc":
        sql += " ORDER BY v.mileage_km ASC"
    elif sort_by == "newest_seen":
        sql += " ORDER BY v.first_seen_at DESC"
    else:
        sql += " ORDER BY v.market_score DESC"

    sql += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(sql, params)
    rows = []
    for r in cursor.fetchall():
        row_dict = dict(r)
        if isinstance(row_dict.get("photos"), str):
            try:
                row_dict["photos"] = json.loads(row_dict["photos"])
            except:
                row_dict["photos"] = []
        if isinstance(row_dict.get("options_json"), str):
            try:
                row_dict["options_json"] = json.loads(row_dict["options_json"])
            except:
                row_dict["options_json"] = {}
        rows.append(row_dict)

    conn.close()
    return {"total": total, "items": rows}

def get_vehicle_by_id(vehicle_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v.*,
               f.tag as favorite_tag,
               f.notes as favorite_notes,
               CASE WHEN f.vehicle_id IS NOT NULL THEN 1 ELSE 0 END as is_favorite
        FROM vehicles v
        LEFT JOIN favorites f ON v.id = f.vehicle_id
        WHERE v.id = ?
    """, (vehicle_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    v = dict(row)
    if isinstance(v.get("photos"), str):
        try: v["photos"] = json.loads(v["photos"])
        except: v["photos"] = []
    if isinstance(v.get("options_json"), str):
        try: v["options_json"] = json.loads(v["options_json"])
        except: v["options_json"] = {}

    # Get price history
    cursor.execute("SELECT * FROM price_history WHERE vehicle_id = ? ORDER BY id ASC", (vehicle_id,))
    v["price_history"] = [dict(ph) for ph in cursor.fetchall()]

    conn.close()
    return v

def update_vehicle_photos(vehicle_id: str, photos: List[str]) -> bool:
    """Updates the photos list and thumbnail for a vehicle."""
    if not photos:
        return False
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE vehicles
        SET photos = ?, thumbnail_url = ?
        WHERE id = ?
    """, (json.dumps(photos), photos[0] if photos else "", vehicle_id))
    conn.commit()
    conn.close()
    return True

def set_favorite(vehicle_id: str, tag: str = "guardado", notes: str = "") -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    now_iso = datetime.utcnow().isoformat()
    cursor.execute("""
        INSERT INTO favorites (vehicle_id, tag, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(vehicle_id) DO UPDATE SET
            tag = excluded.tag,
            notes = excluded.notes,
            updated_at = excluded.updated_at
    """, (vehicle_id, tag, notes, now_iso, now_iso))
    conn.commit()
    conn.close()
    return True

def remove_favorite(vehicle_id: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM favorites WHERE vehicle_id = ?", (vehicle_id,))
    conn.commit()
    conn.close()
    return True

def get_favorites() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v.*, f.tag as favorite_tag, f.notes as favorite_notes, f.created_at as favorited_at
        FROM favorites f
        JOIN vehicles v ON f.vehicle_id = v.id
        ORDER BY f.created_at DESC
    """)
    rows = []
    for r in cursor.fetchall():
        d = dict(r)
        if isinstance(d.get("photos"), str):
            try: d["photos"] = json.loads(d["photos"])
            except: d["photos"] = []
        rows.append(d)
    conn.close()
    return rows

def save_search(name: str, filters: Dict[str, Any]) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    now_iso = datetime.utcnow().isoformat()
    cursor.execute("""
        INSERT INTO saved_searches (name, filters_json, created_at)
        VALUES (?, ?, ?)
    """, (name, json.dumps(filters), now_iso))
    search_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return search_id

def get_saved_searches() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM saved_searches ORDER BY id DESC")
    rows = []
    for r in cursor.fetchall():
        d = dict(r)
        try: d["filters"] = json.loads(d["filters_json"])
        except: d["filters"] = {}
        rows.append(d)
    conn.close()
    return rows

def delete_saved_search(search_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM saved_searches WHERE id = ?", (search_id,))
    conn.commit()
    conn.close()
    return True

def get_dashboard_stats() -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as cnt FROM vehicles WHERE status = 'ACTIVE'")
    total_active = cursor.fetchone()["cnt"]

    today_prefix = datetime.utcnow().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) as cnt FROM vehicles WHERE first_seen_at LIKE ?", (f"{today_prefix}%",))
    new_today = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(*) as cnt FROM vehicles WHERE opportunity_flag = 1 AND status = 'ACTIVE'")
    opportunities = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(*) as cnt FROM favorites")
    favorites_count = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(*) as cnt FROM vehicles WHERE price_change_status = 'PRICE_DROP' AND status = 'ACTIVE'")
    price_drops = cursor.fetchone()["cnt"]

    cursor.execute("""
        SELECT model, COUNT(*) as count, AVG(price_usd) as avg_price, MIN(price_usd) as min_price, MAX(price_usd) as max_price
        FROM vehicles WHERE status = 'ACTIVE' AND price_usd > 0
        GROUP BY model
    """)
    models_stats = [dict(r) for r in cursor.fetchall()]

    latest_scan = get_latest_scan_run()

    conn.close()
    return {
        "total_active": total_active,
        "new_today": new_today,
        "opportunities": opportunities,
        "favorites_count": favorites_count,
        "price_drops": price_drops,
        "models_stats": models_stats,
        "latest_scan": latest_scan
    }


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row["value"] if row else default

def set_setting(key: str, value: str):
    conn = get_connection()
    cursor = conn.cursor()
    now_iso = datetime.utcnow().isoformat()
    cursor.execute("""
        INSERT INTO app_settings (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
    """, (key, value, now_iso))
    conn.commit()
    conn.close()

def delete_setting(key: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM app_settings WHERE key = ?", (key,))
    conn.commit()
    conn.close()
