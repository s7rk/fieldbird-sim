from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_WEATHER_COLUMNS = (
    "temp_min_C",
    "temp_max_C",
    "precip_mm",
    "wind_gust_max_ms",
)


@dataclass(frozen=True)
class WeatherDataStore:
    frame: pd.DataFrame

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        required_columns: tuple[str, ...] = REQUIRED_WEATHER_COLUMNS,
    ) -> "WeatherDataStore":
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Weather CSV not found: {path}")

        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError("Weather CSV must contain a 'date' column.")

        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Weather CSV missing required columns: {missing}. Found: {list(df.columns)}"
            )

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            bad = int(df["date"].isna().sum())
            raise ValueError(f"Weather CSV has {bad} invalid dates in 'date' column.")

        df = df.sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index).normalize()
        df = df.sort_index()

        return cls(frame=df)

    @property
    def start_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame.index.min()).normalize()

    @property
    def end_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame.index.max()).normalize()

    def get_row(self, sim_date: Any, *, missing_date_policy: str = "raise") -> pd.Series:
        ts = pd.Timestamp(sim_date).normalize()
        if ts in self.frame.index:
            row = self.frame.loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            return row

        policy = str(missing_date_policy or "raise").strip().lower()
        if policy == "raise":
            raise KeyError(
                f"Weather date {ts.date()} missing. "
                "Use weather_module.missing_date_policy=ffill|bfill to allow gaps."
            )

        if policy == "ffill":
            sub = self.frame.loc[:ts]
            if sub.empty:
                raise KeyError(f"Weather date {ts.date()} missing and no earlier data for ffill.")
            return sub.iloc[-1]

        if policy == "bfill":
            sub = self.frame.loc[ts:]
            if sub.empty:
                raise KeyError(f"Weather date {ts.date()} missing and no later data for bfill.")
            return sub.iloc[0]

        raise ValueError(
            f"Unsupported missing_date_policy={missing_date_policy!r}. "
            "Expected one of: raise, ffill, bfill."
        )
