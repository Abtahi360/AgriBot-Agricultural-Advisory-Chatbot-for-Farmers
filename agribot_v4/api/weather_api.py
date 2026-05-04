"""
api/weather_api.py
OpenWeatherMap API Integration — NEW

Provides live weather data for crop/planting advice.
The existing static weather handling in chatbot.py is preserved.
This module is ADDITIVE — it enriches queries when a location is provided.

Usage:
  from api.weather_api import WeatherAPI
  weather = WeatherAPI()                    # reads API key from env
  data = weather.get_current(city="Delhi") # returns dict with temp, humidity, etc.

Configuration (set as environment variables):
  OPENWEATHER_API_KEY=your_key_here     (required for live data)
  OPENWEATHER_UNITS=metric              (optional: metric/imperial/standard)

Get a free API key at: https://openweathermap.org/api (60 calls/min free tier)
"""

import os
import json
import time
import warnings
from typing import Optional, Dict, Any
warnings.filterwarnings("ignore")

# Try to import requests (needed for live API calls)
try:
    import requests as _requests
    _requests_available = True
except ImportError:
    _requests_available = False


# ── Configuration ─────────────────────────────────────────────────────────────
OPENWEATHER_BASE  = "https://api.openweathermap.org/data/2.5"
DEFAULT_UNITS     = "metric"   # Celsius
CACHE_TTL_SECONDS = 600        # 10-minute cache to avoid hitting rate limits


class WeatherAPIError(Exception):
    """Raised when the Weather API call fails."""
    pass


class WeatherCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl: int = CACHE_TTL_SECONDS):
        self._store: Dict[str, Dict] = {}
        self._ttl   = ttl

    def get(self, key: str) -> Optional[Dict]:
        if key in self._store:
            entry = self._store[key]
            if time.time() - entry["ts"] < self._ttl:
                return entry["data"]
            del self._store[key]
        return None

    def set(self, key: str, data: Dict):
        self._store[key] = {"data": data, "ts": time.time()}


class WeatherAPI:
    """
    OpenWeatherMap API client with caching and graceful fallback.

    Falls back to None (instead of crashing) when:
      - API key is not set
      - Network is unavailable
      - API quota exceeded
    """

    def __init__(self, api_key: Optional[str] = None, units: str = DEFAULT_UNITS):
        self.api_key = api_key or os.environ.get("OPENWEATHER_API_KEY", "")
        self.units   = units or os.environ.get("OPENWEATHER_UNITS", DEFAULT_UNITS)
        self._cache  = WeatherCache()

    @property
    def _available(self) -> bool:
        return bool(self.api_key) and _requests_available

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        """Make API call with error handling."""
        if not self._available:
            return None
        params["appid"] = self.api_key
        params["units"] = self.units
        try:
            resp = _requests.get(
                f"{OPENWEATHER_BASE}/{endpoint}",
                params=params,
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                raise WeatherAPIError("Invalid API key. Get one at openweathermap.org/api")
            elif resp.status_code == 429:
                raise WeatherAPIError("API rate limit exceeded. Retry after 1 minute.")
            elif resp.status_code == 404:
                raise WeatherAPIError(f"Location not found: {params.get('q', params.get('id', ''))}")
            else:
                return None
        except WeatherAPIError:
            raise
        except Exception:
            return None

    def get_current(
        self,
        city:    Optional[str]   = None,
        lat:     Optional[float] = None,
        lon:     Optional[float] = None,
        country: Optional[str]   = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch current weather for a city or coordinates.

        Returns standardized dict:
          {
            "city":        "Delhi",
            "country":     "IN",
            "temperature": 32.4,    # °C
            "feels_like":  35.1,
            "humidity":    68,       # %
            "pressure":    1005,     # hPa
            "wind_speed":  12.3,     # km/h
            "wind_dir":    180,      # degrees
            "rainfall_1h": 0.0,      # mm (last 1h)
            "condition":   "Haze",
            "icon":        "50d",
            "timestamp":   1715000000,
            "source":      "OpenWeatherMap live",
          }
        Returns None if API key missing or request fails.
        """
        if city and country:
            cache_key = f"current:{city}:{country}"
            query_key = f"{city},{country}"
        elif city:
            cache_key = f"current:{city}"
            query_key = city
        elif lat is not None and lon is not None:
            cache_key = f"current:{lat:.3f},{lon:.3f}"
            query_key = None
        else:
            return None

        cached = self._cache.get(cache_key)
        if cached:
            return cached

        params = {}
        if query_key:
            params["q"] = query_key
        elif lat is not None:
            params["lat"] = lat
            params["lon"] = lon

        raw = self._get("weather", params)
        if not raw:
            return None

        result = self._parse_current(raw)
        self._cache.set(cache_key, result)
        return result

    def _parse_current(self, raw: dict) -> Dict[str, Any]:
        """Parse OpenWeatherMap API response into standard format."""
        wind_ms = raw.get("wind", {}).get("speed", 0)
        wind_kmh = round(wind_ms * 3.6, 1)

        rain = raw.get("rain", {})
        rain_1h = rain.get("1h", 0.0) if isinstance(rain, dict) else 0.0

        return {
            "city":        raw.get("name", "Unknown"),
            "country":     raw.get("sys", {}).get("country", ""),
            "temperature": raw.get("main", {}).get("temp"),
            "feels_like":  raw.get("main", {}).get("feels_like"),
            "humidity":    raw.get("main", {}).get("humidity"),
            "pressure":    raw.get("main", {}).get("pressure"),
            "wind_speed":  wind_kmh,
            "wind_dir":    raw.get("wind", {}).get("deg"),
            "rainfall_1h": rain_1h,
            "condition":   raw.get("weather", [{}])[0].get("main", ""),
            "description": raw.get("weather", [{}])[0].get("description", ""),
            "icon":        raw.get("weather", [{}])[0].get("icon", ""),
            "timestamp":   raw.get("dt"),
            "source":      "OpenWeatherMap live",
        }

    def get_forecast_5day(
        self,
        city:    Optional[str]   = None,
        lat:     Optional[float] = None,
        lon:     Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get 5-day / 3-hour forecast.

        Returns:
          {
            "city": "...",
            "forecast": [
              {"dt_txt": "2025-05-01 12:00:00", "temp": 32.4,
               "humidity": 68, "condition": "Rain", ...},
              ...
            ],
            "daily_summary": {
              "2025-05-01": {"avg_temp": 31.2, "total_rain": 12.5, ...},
              ...
            }
          }
        """
        if city:
            cache_key = f"forecast5:{city}"
            params    = {"q": city}
        elif lat is not None and lon is not None:
            cache_key = f"forecast5:{lat:.3f},{lon:.3f}"
            params    = {"lat": lat, "lon": lon}
        else:
            return None

        cached = self._cache.get(cache_key)
        if cached:
            return cached

        raw = self._get("forecast", params)
        if not raw:
            return None

        result = self._parse_forecast(raw)
        self._cache.set(cache_key, result)
        return result

    def _parse_forecast(self, raw: dict) -> Dict[str, Any]:
        """Parse 5-day forecast into structured daily summary."""
        forecast_list = []
        daily: Dict[str, list] = {}

        for item in raw.get("list", []):
            dt_txt = item.get("dt_txt", "")
            date   = dt_txt.split(" ")[0]
            temp   = item.get("main", {}).get("temp")
            hum    = item.get("main", {}).get("humidity")
            rain   = item.get("rain", {})
            rain3h = rain.get("3h", 0.0) if isinstance(rain, dict) else 0.0
            cond   = item.get("weather", [{}])[0].get("main", "")

            forecast_list.append({
                "dt_txt":    dt_txt,
                "temp":      temp,
                "humidity":  hum,
                "rainfall":  rain3h,
                "condition": cond,
            })

            if date not in daily:
                daily[date] = {"temps": [], "humidities": [], "rainfall": 0.0, "conditions": []}
            daily[date]["temps"].append(temp or 0)
            daily[date]["humidities"].append(hum or 0)
            daily[date]["rainfall"] += rain3h
            daily[date]["conditions"].append(cond)

        daily_summary = {}
        for date, values in daily.items():
            daily_summary[date] = {
                "avg_temp":    round(sum(values["temps"]) / max(len(values["temps"]), 1), 1),
                "max_temp":    round(max(values["temps"]), 1),
                "min_temp":    round(min(values["temps"]), 1),
                "avg_humidity":round(sum(values["humidities"]) / max(len(values["humidities"]), 1), 1),
                "total_rain":  round(values["rainfall"], 1),
                "conditions":  list(set(values["conditions"])),
            }

        return {
            "city":          raw.get("city", {}).get("name", ""),
            "country":       raw.get("city", {}).get("country", ""),
            "forecast":      forecast_list,
            "daily_summary": daily_summary,
        }

    def get_planting_advice(
        self,
        city: str,
        crop: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        High-level helper: fetch weather and return planting-specific summary.

        Returns dict with:
          - current conditions
          - 5-day forecast summary
          - planting_window: "good" / "caution" / "delay"
          - recommendation text
        """
        current  = self.get_current(city=city)
        forecast = self.get_forecast_5day(city=city)

        advice = {
            "city":            city,
            "crop":            crop,
            "current_weather": current,
            "forecast":        forecast,
            "planting_window": "unknown",
            "recommendation":  "",
            "api_status":      "live" if current else "unavailable",
        }

        if not current:
            advice["recommendation"] = (
                f"Live weather data unavailable for {city}. "
                "Please check local weather and use the general planting guide."
            )
            return advice

        temp    = current.get("temperature", 25)
        humidity= current.get("humidity", 65)
        cond    = current.get("condition", "")

        # Simple planting window logic
        is_raining   = "rain" in cond.lower() or "storm" in cond.lower()
        too_hot      = temp > 40
        too_cold     = temp < 8

        if too_hot or too_cold:
            window = "delay"
            note   = f"Temperature {temp}°C is not suitable for most crops."
        elif is_raining:
            window = "caution"
            note   = "Current rain detected. Wait 1–2 days for soil to drain before transplanting."
        elif humidity > 90:
            window = "caution"
            note   = f"Very high humidity ({humidity}%) increases disease risk. Apply preventive fungicide."
        else:
            window = "good"
            note   = f"Conditions look suitable. Temp={temp}°C, Humidity={humidity}%."

        advice["planting_window"] = window
        advice["recommendation"]  = note

        # Add forecast summary for next 5 days
        if forecast and forecast.get("daily_summary"):
            days = list(forecast["daily_summary"].items())[:5]
            advice["5day_summary"] = {
                date: {
                    "temp_range": f"{v['min_temp']}–{v['max_temp']}°C",
                    "rain_mm":    v["total_rain"],
                    "conditions": v["conditions"],
                }
                for date, v in days
            }

        return advice

    def status(self) -> Dict[str, str]:
        """Return API configuration status."""
        return {
            "api_key_set":    "yes" if self.api_key else "no (set OPENWEATHER_API_KEY)",
            "requests_lib":   "available" if _requests_available else "missing (pip install requests)",
            "units":          self.units,
            "cache_ttl_sec":  str(CACHE_TTL_SECONDS),
            "base_url":       OPENWEATHER_BASE,
        }


# ── Singleton for import ───────────────────────────────────────────────────────
weather_api = WeatherAPI()


# ── Convenience functions ──────────────────────────────────────────────────────
def get_weather_for_city(city: str) -> Optional[Dict]:
    """One-liner: get current weather for any city."""
    return weather_api.get_current(city=city)


def get_planting_advice(city: str, crop: Optional[str] = None) -> Dict:
    """One-liner: get planting advice based on live weather."""
    return weather_api.get_planting_advice(city=city, crop=crop)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api = WeatherAPI()
    print("Weather API Status:")
    print(json.dumps(api.status(), indent=2))

    if not api._available:
        print("\n[INFO] Set OPENWEATHER_API_KEY environment variable to test live calls.")
        print("       Example: export OPENWEATHER_API_KEY=your_key_here")
        print("       Get free key at: https://openweathermap.org/api")
    else:
        print("\n[TEST] Fetching weather for Delhi ...")
        data = api.get_current(city="Delhi")
        print(json.dumps(data, indent=2))

        print("\n[TEST] Planting advice for Mumbai, wheat ...")
        advice = api.get_planting_advice(city="Mumbai", crop="wheat")
        print(json.dumps(advice, indent=2))
