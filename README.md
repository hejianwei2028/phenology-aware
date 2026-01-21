"""Phenology pipeline utilities.

This module provides a small, testable pipeline for turning raw index/weather
series into phenological event features. The functions are intentionally
lightweight so they can be unit-tested without heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PhenologyCurve:
    """Container for a fitted phenology curve."""

    dates: List[date]
    values: List[float]
    smoothed: List[float]
    metadata: Dict[str, object]


@dataclass(frozen=True)
class PhenologyEvent:
    """Structured phenological event."""

    event: str
    date: date
    doy: int
    value: float
    quality: str


def _parse_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    raise TypeError(f"Unsupported date type: {type(value)!r}")


def _parse_series(
    records: Iterable[Mapping[str, object]],
    date_key: str,
    value_key: str,
) -> List[Tuple[date, float]]:
    series: List[Tuple[date, float]] = []
    for record in records:
        if date_key not in record or value_key not in record:
            raise KeyError(
                f"Record missing required keys: {date_key!r}, {value_key!r}"
            )
        record_date = _parse_date(record[date_key])
        try:
            record_value = float(record[value_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {value_key!r}: {record[value_key]!r}") from exc
        series.append((record_date, record_value))
    return series


def preprocess_series(
    D_index: Iterable[Mapping[str, object]],
    D_weather: Iterable[Mapping[str, object]],
    meta: Mapping[str, object],
    *,
    index_date_key: str = "date",
    index_value_key: str = "value",
    weather_date_key: str = "date",
    weather_fields: Sequence[str] = ("tmean", "precip"),
) -> Dict[str, object]:
    """Parse, validate, and align index + weather series.

    Args:
        D_index: Iterable of index observations (date + value).
        D_weather: Iterable of weather observations (date + weather fields).
        meta: Metadata dictionary. Expected keys: ``site_id`` and optional
            ``timezone`` / ``elevation``.
        index_date_key: Key name for the index observation date.
        index_value_key: Key name for the index observation value.
        weather_date_key: Key name for the weather observation date.
        weather_fields: Weather field names to extract.

    Returns:
        Dict with keys: ``meta``, ``index_series``, ``weather_series``,
        ``aligned_dates``, and ``aligned_weather``.
    """

    if "site_id" not in meta:
        raise KeyError("meta must include 'site_id'")

    index_series = _parse_series(D_index, index_date_key, index_value_key)
    weather_series: Dict[date, Dict[str, float]] = {}

    for record in D_weather:
        if weather_date_key not in record:
            raise KeyError(f"Weather record missing {weather_date_key!r}")
        record_date = _parse_date(record[weather_date_key])
        weather_values: Dict[str, float] = {}
        for field in weather_fields:
            if field not in record:
                raise KeyError(f"Weather record missing {field!r}")
            try:
                weather_values[field] = float(record[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid value for {field!r}: {record[field]!r}") from exc
        weather_series[record_date] = weather_values

    aligned_dates: List[date] = []
    aligned_values: List[float] = []
    aligned_weather: Dict[str, List[float]] = {field: [] for field in weather_fields}

    for record_date, record_value in sorted(index_series, key=lambda item: item[0]):
        if record_date not in weather_series:
            continue
        aligned_dates.append(record_date)
        aligned_values.append(record_value)
        for field in weather_fields:
            aligned_weather[field].append(weather_series[record_date][field])

    if not aligned_dates:
        raise ValueError("No overlapping dates between index and weather series")

    return {
        "meta": dict(meta),
        "index_series": index_series,
        "weather_series": weather_series,
        "aligned_dates": aligned_dates,
        "aligned_values": aligned_values,
        "aligned_weather": aligned_weather,
    }


def _moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    smoothed: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window)
        end = min(len(values), idx + window + 1)
        smoothed.append(mean(values[start:end]))
    return smoothed


def fit_phenology_curve(
    dates: Sequence[date],
    values: Sequence[float],
    *,
    smoothing_window: int = 3,
    metadata: Optional[Mapping[str, object]] = None,
) -> PhenologyCurve:
    """Fit a phenology curve using a simple smoothing strategy.

    Args:
        dates: Ordered sequence of dates.
        values: Sequence of index values matching ``dates``.
        smoothing_window: Window size (in observations) for moving average.
        metadata: Optional metadata to attach to the curve.

    Returns:
        PhenologyCurve with raw and smoothed values.
    """

    if len(dates) != len(values):
        raise ValueError("dates and values must have the same length")

    smoothed = _moving_average(values, smoothing_window)
    return PhenologyCurve(
        dates=list(dates),
        values=list(values),
        smoothed=smoothed,
        metadata=dict(metadata or {}),
    )


def extract_events(
    curve: PhenologyCurve,
    *,
    onset_threshold: float = 0.2,
    peak_threshold: float = 0.8,
) -> List[PhenologyEvent]:
    """Extract phenology events (onset/peak/senescence) from a curve.

    Args:
        curve: Fitted phenology curve.
        onset_threshold: Fraction of max value used to define onset.
        peak_threshold: Fraction of max value used to define peak.

    Returns:
        List of PhenologyEvent objects.
    """

    if not curve.smoothed:
        raise ValueError("Curve contains no values")

    max_value = max(curve.smoothed)
    if max_value <= 0:
        raise ValueError("Curve max value must be positive")

    onset_value = max_value * onset_threshold
    peak_value = max_value * peak_threshold

    onset_idx = next(
        (i for i, value in enumerate(curve.smoothed) if value >= onset_value),
        0,
    )
    peak_idx = next(
        (i for i, value in enumerate(curve.smoothed) if value >= peak_value),
        len(curve.smoothed) - 1,
    )
    senescence_idx = next(
        (
            i
            for i in range(len(curve.smoothed) - 1, -1, -1)
            if curve.smoothed[i] >= onset_value
        ),
        len(curve.smoothed) - 1,
    )

    events = [
        PhenologyEvent(
            event="onset",
            date=curve.dates[onset_idx],
            doy=curve.dates[onset_idx].timetuple().tm_yday,
            value=curve.smoothed[onset_idx],
            quality="derived",
        ),
        PhenologyEvent(
            event="peak",
            date=curve.dates[peak_idx],
            doy=curve.dates[peak_idx].timetuple().tm_yday,
            value=curve.smoothed[peak_idx],
            quality="derived",
        ),
        PhenologyEvent(
            event="senescence",
            date=curve.dates[senescence_idx],
            doy=curve.dates[senescence_idx].timetuple().tm_yday,
            value=curve.smoothed[senescence_idx],
            quality="derived",
        ),
    ]

    return events


def assemble_features(
    meta: Mapping[str, object],
    events: Sequence[PhenologyEvent],
    *,
    weather_summary: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    """Assemble features list output for downstream modeling.

    Args:
        meta: Metadata dictionary. Must include ``site_id``.
        events: Sequence of phenology events.
        weather_summary: Optional aggregated weather summary values.

    Returns:
        Dict with keys: ``site_id``, ``features_list``. ``features_list`` is a
        list of dictionaries with keys: ``event``, ``date``, ``doy``,
        ``value``, ``quality``, and ``weather``.
    """

    if "site_id" not in meta:
        raise KeyError("meta must include 'site_id'")

    features_list: List[Dict[str, object]] = []
    for event in events:
        features_list.append(
            {
                "event": event.event,
                "date": event.date.isoformat(),
                "doy": event.doy,
                "value": event.value,
                "quality": event.quality,
                "weather": dict(weather_summary or {}),
            }
        )

    return {
        "site_id": meta["site_id"],
        "features_list": features_list,
    }


def summarize_weather(
    aligned_weather: Mapping[str, Sequence[float]],
) -> Dict[str, float]:
    """Summarize weather fields with their mean values."""

    summary: Dict[str, float] = {}
    for field, values in aligned_weather.items():
        if not values:
            raise ValueError(f"Weather field {field!r} has no values")
        summary[field] = mean(values)
    return summary


def run_pipeline(
    D_index: Iterable[Mapping[str, object]],
    D_weather: Iterable[Mapping[str, object]],
    meta: Mapping[str, object],
    *,
    index_date_key: str = "date",
    index_value_key: str = "value",
    weather_date_key: str = "date",
    weather_fields: Sequence[str] = ("tmean", "precip"),
    smoothing_window: int = 3,
    onset_threshold: float = 0.2,
    peak_threshold: float = 0.8,
) -> Dict[str, object]:
    """End-to-end phenology pipeline with aligned weather summary."""

    parsed = preprocess_series(
        D_index,
        D_weather,
        meta,
        index_date_key=index_date_key,
        index_value_key=index_value_key,
        weather_date_key=weather_date_key,
        weather_fields=weather_fields,
    )
    curve = fit_phenology_curve(
        parsed["aligned_dates"],
        parsed["aligned_values"],
        smoothing_window=smoothing_window,
        metadata={"site_id": meta.get("site_id")},
    )
    events = extract_events(
        curve,
        onset_threshold=onset_threshold,
        peak_threshold=peak_threshold,
    )
    weather_summary = summarize_weather(parsed["aligned_weather"])
    return assemble_features(meta, events, weather_summary=weather_summary)
