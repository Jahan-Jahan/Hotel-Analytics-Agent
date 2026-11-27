"""predict_cancellation tool: calls the Hotel-Booking-Demand prediction API."""
from __future__ import annotations

from typing import Any

import httpx

# Mirrors BookingFeatures in the prediction API. Optional fields use the API defaults.
REQUIRED = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "adr",
]
OPTIONAL = [
    "children", "babies", "meal", "country", "market_segment", "distribution_channel",
    "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "reserved_room_type", "booking_changes", "deposit_type", "agent", "company",
    "customer_type", "required_car_parking_spaces", "total_of_special_requests",
]
ALL_FIELDS = REQUIRED + OPTIONAL
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


class PredictClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        # trust_env=False: ignore HTTP(S)_PROXY / ALL_PROXY. The prediction API is normally local,
        # and a system proxy or VPN would otherwise intercept localhost and answer 503.
        self.client = client or httpx.Client(timeout=timeout, trust_env=False)

    def predict(self, **fields: Any) -> dict[str, Any]:
        unknown = sorted(k for k in fields if k not in ALL_FIELDS)
        payload = {k: v for k, v in fields.items() if k in ALL_FIELDS and v is not None}
        missing = [f for f in REQUIRED if f not in payload]
        if missing:
            message = f"Missing required fields: {', '.join(missing)}."
            if unknown:
                message += (f" These field names are not valid and were ignored: {', '.join(unknown)}. "
                            f"Use the exact names from the tool schema.")
            message += (" If the user already gave these values, call the tool again now with the exact "
                        "field names above; only ask the user for values they really did not provide. "
                        "Never guess a value.")
            return {"ok": False, "error": message, "retryable": True}
        try:
            response = self.client.post(f"{self.base_url}/predict", json=payload)
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"Prediction API unreachable at {self.base_url}: {type(exc).__name__}"}

        if response.status_code == 422:
            problems = [f"{'.'.join(map(str, e.get('loc', [])[1:]))}: {e.get('msg')}"
                        for e in response.json().get("detail", [])]
            return {"ok": False, "error": "Invalid input: " + "; ".join(problems), "retryable": True}
        if response.status_code != 200:
            return {"ok": False, "error": f"Prediction API returned HTTP {response.status_code}."}

        body = response.json()
        return {
            "ok": True,
            "cancel_probability": body["cancel_probability"],
            "will_cancel": body["will_cancel"],
            "threshold_used": body["threshold_used"],
            "inputs_used": payload,
            **({"ignored_fields": unknown} if unknown else {}),
        }
