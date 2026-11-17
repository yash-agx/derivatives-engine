"""Download one trading day's NSE F&O bhavcopy and keep the NIFTY rows.

    python analysis/fetch_phase6_data.py 2026-11-16

Writes ``data/nifty_fo_bhavcopy_<YYYYMMDD>.csv``: the exchange's own file with
the same header and the rows copied verbatim, filtered to ``TckrSymb ==
"NIFTY"``. Nothing is recomputed or rounded here -- the filter is only so the
repository carries 0.2 MB of the instrument we price instead of 8 MB of every
stock future and option on the exchange.

Why the bhavcopy and not the live option chain
----------------------------------------------
The bhavcopy is the end-of-day settlement file: one row per contract, with a
close and a settlement price for *every* listed strike and expiry, plus volume
and open interest, and the underlying's close in ``UndrlygPric``. It is a
single consistent snapshot taken at one instant (the 15:30 IST close), which is
exactly what an implied-volatility smile needs -- quotes taken at different
times would give a smile smeared by moves in the index itself.

The file lives at

    https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_<date>_F_0000.csv.zip

and the host rejects requests without a browser user agent, so one is sent.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

URL = ("https://nsearchives.nseindia.com/content/fo/"
       "BhavCopy_NSE_FO_0_0_0_{stamp}_F_0000.csv.zip")

# Akamai in front of nseindia.com serves 403 to anything that does not look
# like a browser. Nothing here depends on the exact string.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

SYMBOL = "NIFTY"


def download(trade_date: date) -> bytes:
    stamp = trade_date.strftime("%Y%m%d")
    request = urllib.request.Request(URL.format(stamp=stamp), headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def extract_symbol(zipped: bytes, symbol: str) -> tuple[list[str], list[list[str]]]:
    with zipfile.ZipFile(io.BytesIO(zipped)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        text = archive.read(name).decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    symbol_column = header.index("TckrSymb")
    rows = [row for row in reader if row[symbol_column] == symbol]
    return header, rows


def main(argv: list[str]) -> int:
    trade_date = date.fromisoformat(argv[1]) if len(argv) > 1 else date.today()
    stamp = trade_date.strftime("%Y%m%d")

    print(f"downloading NSE F&O bhavcopy for {trade_date.isoformat()} "
          f"({trade_date.strftime('%A')})")
    header, rows = extract_symbol(download(trade_date), SYMBOL)
    if not rows:
        print(f"no {SYMBOL} rows -- not a trading day?")
        return 1

    DATA.mkdir(exist_ok=True)
    out = DATA / f"nifty_fo_bhavcopy_{stamp}.csv"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    spots = {row[header.index("UndrlygPric")] for row in rows}
    print(f"wrote {out.relative_to(ROOT)}  ({len(rows)} {SYMBOL} contracts)")
    print(f"underlying close carried in the file: {sorted(spots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
