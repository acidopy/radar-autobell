import json
import time
import urllib.request
import urllib.parse
import urllib.error
import logging
from typing import Dict, List, Any, Optional
from backend.config import AUTOBELL_API_URL, AUTOBELL_HEADERS
from backend.database import get_setting

logger = logging.getLogger("radar_autobell.connector")

class AutobellConnector:
    """
    Legitimate client for Autobell Global public and authenticated API endpoints.
    Employs respectful rate limiting, request headers, authenticated user sessions,
    and exponential backoff retries.
    """
    def __init__(self, delay_between_requests: float = 0.25, max_retries: int = 3, timeout: int = 15):
        self.delay = delay_between_requests
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request_time = 0.0

    def _wait_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()

    def _get_headers(self) -> Dict[str, str]:
        headers = dict(AUTOBELL_HEADERS)
        token = get_setting("autobell_access_token")
        if token:
            headers["Authorization"] = f"Bearer {token.strip()}"
        return headers

    def _http_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._wait_rate_limit()
        query_str = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"{AUTOBELL_API_URL}{endpoint}{query_str}"

        headers = self._get_headers()
        req = urllib.request.Request(url, headers=headers)

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        raw = resp.read().decode("utf-8")
                        return json.loads(raw)
                    else:
                        raise RuntimeError(f"HTTP {resp.status} from {url}")
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                wait_sec = attempt * 1.5
                logger.warning(f"Attempt {attempt} failed for {url}: {e}. Retrying in {wait_sec}s...")
                time.sleep(wait_sec)

        logger.error(f"All {self.max_retries} attempts failed for {url}. Last error: {last_err}")
        raise RuntimeError(f"Failed to fetch from Autobell Global after {self.max_retries} attempts: {last_err}")

    def test_token(self, token: str) -> Dict[str, Any]:
        """
        Tests whether an access token is recognized by Autobell Global.
        """
        headers = dict(AUTOBELL_HEADERS)
        headers["Authorization"] = f"Bearer {token.strip()}"
        url = f"{AUTOBELL_API_URL}/buyerCarGoods/isAlive?carStockKey=GV169113&countryCode=en"
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return {"valid": True, "data": data}
                return {"valid": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def search_vehicles(
        self,
        keyword: str,
        page: int = 1,
        page_size: int = 20,
        country_code: str = "en"
    ) -> Dict[str, Any]:
        """
        Searches vehicles via /search/carFilterMobileList endpoint.
        Returns {'total': int, 'items': List[Dict]}
        """
        params = {
            "searchKeyword": keyword,
            "page": page,
            "size": page_size,
            "countryCode": country_code
        }
        res = self._http_get("/search/carFilterMobileList", params)
        total = res.get("total", 0)
        items = res.get("data", [])
        return {"total": total, "items": items}

    def fetch_all_for_model(
        self,
        keyword: str,
        max_items: int = 1500,
        page_size: int = 50,
        country_code: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Paginates through search results for a model up to max_items.
        """
        all_items = []
        page = 1

        while len(all_items) < max_items:
            req_size = min(page_size, max_items - len(all_items))
            result = self.search_vehicles(keyword=keyword, page=page, page_size=req_size, country_code=country_code)
            items = result.get("items", [])
            total = result.get("total", 0)

            if not items:
                break

            all_items.extend(items)

            if len(all_items) >= total or len(items) < req_size:
                break

            page += 1

        return all_items

    def get_vehicle_details(self, car_key: str) -> Optional[Dict[str, Any]]:
        """
        Fetches detailed info via /buyerCarGoods/info
        """
        try:
            res = self._http_get("/buyerCarGoods/info", {"carKey": car_key})
            return res.get("data")
        except Exception as e:
            logger.warning(f"Could not fetch details for {car_key}: {e}")
            return None

    def get_vehicle_photos(self, car_key: str) -> List[Dict[str, Any]]:
        """
        Fetches photo gallery via /file/list
        """
        try:
            res = self._http_get("/file/list", {
                "referenceKey": car_key,
                "fileBusinessType": "CAR_DETAIL_IMAGE"
            })
            return res.get("data", [])
        except Exception as e:
            logger.warning(f"Could not fetch photos for {car_key}: {e}")
            return []

    def check_is_alive(self, car_key: str, country_code: str = "en") -> bool:
        """
        Checks if vehicle is still available on Autobell via /buyerCarGoods/isAlive
        """
        try:
            res = self._http_get("/buyerCarGoods/isAlive", {
                "carStockKey": car_key,
                "countryCode": country_code
            })
            data = res.get("data", {})
            return bool(data.get("alive", False))
        except Exception as e:
            logger.warning(f"Could not check isAlive for {car_key}: {e}")
            return True
