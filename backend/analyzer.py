import statistics
from typing import Dict, List, Any, Tuple
from backend.database import get_connection

def calculate_cohort_stats() -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Calculates median, mean, min, max price and mean mileage for each (model, year).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model, year, price_usd, mileage_km
        FROM vehicles
        WHERE status = 'ACTIVE' AND price_usd > 0
    """)
    rows = cursor.fetchall()
    conn.close()

    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    model_grouped: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        key = (r["model"], r["year"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append({"price": r["price_usd"], "mileage": r["mileage_km"]})

        m_key = r["model"]
        if m_key not in model_grouped:
            model_grouped[m_key] = []
        model_grouped[m_key].append({"price": r["price_usd"], "mileage": r["mileage_km"]})

    stats: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for key, items in grouped.items():
        prices = [x["price"] for x in items]
        mileages = [x["mileage"] for x in items]
        stats[key] = {
            "count": len(items),
            "median_price": statistics.median(prices),
            "mean_price": round(statistics.mean(prices), 2),
            "min_price": min(prices),
            "max_price": max(prices),
            "mean_mileage": round(statistics.mean(mileages), 1) if mileages else 0
        }

    # Model-level fallbacks
    for m_key, items in model_grouped.items():
        prices = [x["price"] for x in items]
        mileages = [x["mileage"] for x in items]
        stats[(m_key, 0)] = {
            "count": len(items),
            "median_price": statistics.median(prices),
            "mean_price": round(statistics.mean(prices), 2),
            "min_price": min(prices),
            "max_price": max(prices),
            "mean_mileage": round(statistics.mean(mileages), 1) if mileages else 0
        }

    return stats

def evaluate_vehicle_opportunity(v: Dict[str, Any], stats: Dict[Tuple[str, int], Dict[str, Any]]) -> Tuple[int, str, float]:
    """
    Evaluates whether a vehicle is a 'POSIBLE OPORTUNIDAD' and computes its
    'Mejor relación precio/año/km' market score.
    Returns: (opportunity_flag, opportunity_reason, market_score)
    """
    price = v.get("price_usd") or 0
    year = v.get("year") or 0
    mileage = v.get("mileage_km") or 0
    model = v.get("model") or ""
    sunroof = v.get("sunroof") or 0
    inspected = v.get("inspected") or 0

    if price <= 0:
        return 0, "Precio no disponible para análisis", 10.0

    cohort = stats.get((model, year))
    if not cohort or cohort["count"] < 2:
        # fallback to model overall
        cohort = stats.get((model, 0))

    if not cohort:
        return 0, "Insuficientes datos de mercado para comparar este modelo", 50.0

    median_price = cohort["median_price"]
    mean_mileage = cohort["mean_mileage"]

    # Price difference percentage vs cohort median
    price_diff_pct = round(((median_price - price) / median_price) * 100, 1)

    mileage_diff_pct = 0.0
    if mean_mileage > 0:
        mileage_diff_pct = round(((mean_mileage - mileage) / mean_mileage) * 100, 1)

    is_opportunity = 0
    reasons = []

    if price_diff_pct >= 10.0:
        is_opportunity = 1
        reasons.append(f"⭐ {price_diff_pct}% por debajo del promedio de mercado ({model} {year or ''})")
        if mileage_diff_pct > 15.0:
            diff_km = int(abs(mean_mileage - mileage))
            reasons.append(f"{diff_km:,} km menos que el promedio de su categoría")
    elif price_diff_pct >= 5.0 and mileage_diff_pct >= 25.0:
        is_opportunity = 1
        reasons.append(f"⭐ Excelente relación precio con kilometraje excepcionalmente bajo ({mileage:,} km)")

    reason_str = " · ".join(reasons) if reasons else "Precio acorde al rango de mercado."

    # Score formula: 0 to 100 (RADAR AUTOBELL index)
    score = 50.0
    score += max(-30.0, min(35.0, price_diff_pct * 1.2))
    score += max(-15.0, min(15.0, mileage_diff_pct * 0.4))
    if year > 2012:
        score += min(10.0, (year - 2012) * 1.2)
    if sunroof:
        score += 3.5
    if inspected:
        score += 2.5

    final_score = round(max(5.0, min(99.0, score)), 1)
    return is_opportunity, reason_str, final_score

def update_all_opportunities():
    """
    Recalculates stats and updates all active vehicles in the database.
    """
    stats = calculate_cohort_stats()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, model, year, price_usd, mileage_km, sunroof, inspected
        FROM vehicles
        WHERE status = 'ACTIVE'
    """)
    vehicles = [dict(r) for r in cursor.fetchall()]

    for v in vehicles:
        is_opp, reason, score = evaluate_vehicle_opportunity(v, stats)
        cursor.execute("""
            UPDATE vehicles
            SET opportunity_flag = ?, opportunity_reason = ?, market_score = ?
            WHERE id = ?
        """, (is_opp, reason, score, v["id"]))

    conn.commit()
    conn.close()
