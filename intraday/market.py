from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


_NY_TZ = ZoneInfo("America/New_York")


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    current += timedelta(weeks=n - 1)
    return current


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _observed_holiday(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@dataclass(slots=True)
class MarketWindow:
    opens_at: datetime
    closes_at: datetime
    session: str


class USMarketCalendar:
    regular_open = time(9, 30)
    regular_close = time(16, 0)
    half_day_close = time(13, 0)

    def trading_window(self, moment: datetime) -> MarketWindow | None:
        localized = self._localize(moment)
        session_day = localized.date()
        if self.is_holiday(session_day) or localized.weekday() >= 5:
            return None

        close_time = self.half_day_close if self.is_half_day(session_day) else self.regular_close
        return MarketWindow(
            opens_at=datetime.combine(session_day, self.regular_open, tzinfo=_NY_TZ),
            closes_at=datetime.combine(session_day, close_time, tzinfo=_NY_TZ),
            session="half_day" if close_time == self.half_day_close else "regular",
        )

    def is_market_open(self, moment: datetime) -> bool:
        window = self.trading_window(moment)
        if window is None:
            return False
        localized = self._localize(moment)
        return window.opens_at <= localized < window.closes_at

    def slot_start(self, moment: datetime, interval_minutes: int) -> datetime:
        localized = self._localize(moment)
        minutes = (localized.hour * 60) + localized.minute
        floored = minutes - (minutes % interval_minutes)
        hour, minute = divmod(floored, 60)
        return localized.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def next_slot(self, moment: datetime, interval_minutes: int) -> datetime:
        return self.slot_start(moment, interval_minutes) + timedelta(minutes=interval_minutes)

    def is_holiday(self, session_day: date) -> bool:
        year = session_day.year
        holidays = {
            _observed_holiday(date(year, 1, 1)),
            _nth_weekday_of_month(year, 1, 0, 3),
            _nth_weekday_of_month(year, 2, 0, 3),
            _easter_sunday(year) - timedelta(days=2),
            _last_weekday_of_month(year, 5, 0),
            _observed_holiday(date(year, 6, 19)),
            _observed_holiday(date(year, 7, 4)),
            _nth_weekday_of_month(year, 9, 0, 1),
            _nth_weekday_of_month(year, 11, 3, 4),
            _observed_holiday(date(year, 12, 25)),
        }
        return session_day in holidays

    def is_half_day(self, session_day: date) -> bool:
        thanksgiving = _nth_weekday_of_month(session_day.year, 11, 3, 4)
        day_after_thanksgiving = thanksgiving + timedelta(days=1)
        christmas_eve = date(session_day.year, 12, 24)
        independence_eve = date(session_day.year, 7, 3)
        half_days = {
            day_after_thanksgiving,
        }
        if christmas_eve.weekday() < 5 and not self.is_holiday(christmas_eve):
            half_days.add(christmas_eve)
        if independence_eve.weekday() < 5 and not self.is_holiday(independence_eve):
            half_days.add(independence_eve)
        return session_day in half_days

    def _localize(self, moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=_NY_TZ)
        return moment.astimezone(_NY_TZ)
