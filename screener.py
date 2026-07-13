"""Zero-input market screener.

Pulls the current constituents of the S&P 500, Nasdaq-100, and Dow Jones 30,
downloads ~1 year of daily prices for the combined universe, runs the
trend + momentum buy-signal heuristic from ``stock_analyzer`` on each, and
prints a ranking of the strongest BUY candidates "today".

Run it with no arguments and no prompts:

    python screener.py

This is a simple heuristic (20/50-day MA crossover + RSI), not a validated
predictive model. No backtesting. Not financial advice.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime

try:
    import pandas as pd
    import requests
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    print(f"Missing dependency: {exc}. Activate the venv and install requirements.")
    sys.exit(1)

from stock_analyzer import generate_buy_signal

HEADERS = {"User-Agent": "Mozilla/5.0 (stock-data-puller screener)"}
LOOKBACK = "1y"          # enough for the 50-day moving average with buffer
MIN_HISTORY = 50         # generate_buy_signal needs >= long_window (50) prices

# --------------------------------------------------------------------------- #
# Baked-in offline fallbacks (captured 2026-07). yfinance uses '-' not '.'.
# Live fetch is preferred; these only kick in if every network source fails.
# --------------------------------------------------------------------------- #
STATIC_SP500 = ["NVDA","AAPL","MSFT","AMZN","GOOGL","GOOG","AVGO","META","TSLA","BRK-B","LLY","MU","WMT","JPM","AMD","V","JNJ","XOM","INTC","MA","CSCO","AMAT","ABBV","CAT","BAC","LRCX","COST","UNH","ORCL","GE","CVX","KO","MS","PG","HD","NFLX","PLTR","GS","MRK","KLAC","PM","GEV","DELL","TXN","IBM","PANW","WFC","RTX","SNDK","LIN","AXP","C","ANET","TMUS","MRVL","TMO","AMGN","STX","QCOM","MCD","APH","CRWD","WDC","PEP","ADI","NEE","VZ","SCHW","UNP","BA","DIS","TJX","WELL","GILD","ABT","BLK","GLW","DE","ETN","UBER","T","BX","APP","ISRG","DHR","PFE","CRM","CB","COP","PGR","BKNG","CVS","PLD","SPGI","SYK","COF","VRTX","SBUX","BMY","LMT","MO","PH","VRT","FTNT","LOW","NOW","SO","HWM","MDT","TT","CDNS","BNY","EQIX","PNC","GD","ADP","NEM","DUK","HOOD","PWR","USB","MAR","UPS","MCK","MNST","WM","CSX","ELV","CEG","ADBE","CMI","DDOG","WMB","CME","VLO","JCI","MRSH","HCA","KKR","MPC","ABNB","MCO","FCX","CMCSA","ACN","SNPS","DASH","MMM","SHW","CI","PSX","INTU","AMT","AON","ITW","ICE","RCL","NOC","MDLZ","ECL","EMR","FDX","EOG","CL","AEP","NSC","CTAS","HLT","TRV","ORLY","KMI","SPG","NXPI","SLB","ROST","HON","MSI","REGN","GM","TDG","RSG","APO","CRH","URI","WBD","APD","AJG","HONA","BSX","DLR","ALL","GWW","PCAR","NKE","TFC","MPWR","CIEN","AFL","HPE","D","SRE","TGT","FIX","COHR","TRGP","LITE","MET","O","COR","OKE","TEL","CTVA","BKR","CARR","DAL","PSA","F","CAH","KEYS","OXY","FANG","LHX","TER","ETR","FAST","AME","NUE","EW","VST","FITB","EA","ROK","EBAY","NDAQ","DVN","XEL","AZO","STT","HUM","ODFL","EXC","FLEX","CMG","GRMN","XYZ","CVNA","AMP","MCHP","TTWO","MSCI","VTR","YUM","ADSK","IDXX","WAB","AXON","LYV","KDP","AIG","BDX","DHI","PYPL","IBKR","ED","COIN","CBRE","PRU","PEG","SYY","ADM","PAYX","UAL","HIG","PCG","VMC","A","WEC","KVUE","WAT","KR","KMB","HBAN","ROP","CCL","IRM","ACGL","WDAY","ON","MTB","HSY","CCI","IQV","MLM","NTRS","EME","STLD","CNC","JBL","RJF","NTAP","VEEV","EXPE","CASY","ZTS","AEE","DTE","BIIB","EQT","EXR","IR","LVS","ATO","KHC","CFG","GEHC","FICO","NRG","Q","HAL","DXCM","EL","EIX","CBOE","VICI","TDY","CNP","DOV","RMD","XYL","ES","TPL","CINF","OTIS","FE","WTW","AVB","WRB","TPR","FISV","DG","PPL","ARES","JBHT","MRNA","RF","MTD","EQR","AWK","WSM","VRSK","CPRT","WST","PPG","HUBB","KEY","VRSN","SYF","PFG","DLTR","TROW","L","FFIV","FSLR","CPAY","OMC","PHM","BRO","CMS","LUV","CHRW","EXPD","DGX","CHD","STZ","INCY","VLTO","BG","LH","SW","DRI","NI","HPQ","FDXF","RL","DOW","FIS","ROL","STE","GPN","CTSH","EXE","SNA","EFX","TSN","ULTA","PKG","EVRG","LEN","SBAC","LNT","AMCR","GIS","IP","IFF","LII","ESS","VTRS","FTV","LYB","CF","CDW","AKAM","ZBH","SMCI","DD","INVH","BR","NVR","BBY","GPC","BEN","KIM","WY","IEX","BALL","CHTR","NDSN","MAA","TSCO","HST","GEN","TXT","MAS","DVA","DOC","J","EG","DECK","ALB","REG","PTC","MKC","GL","AIZ","COO","LULU","TKO","SOLV","HRL","SWK","LDOS","GNRC","PNW","UDR","ALGN","ERIE","TYL","ZBRA","IVZ","APTV","RVTY","PNR","APA","TRMB","AVY","MGM","GDDY","BF-B","SJM","CSGP","ALLE","BAX","CLX","CPT","HAS","HII","PODD","TECH","FOXA","FOX","CRL","JKHY","BXP","PSKY","AES","FRT","DPZ","WYNN","NWSA","HSIC","IT","FDS","TTD","UHS","NCLH","SWKS","ARE","AOS","BLDR","TAP","MOS","NWS","SATS"]
STATIC_NDX = ["NVDA","AAPL","MSFT","AMZN","GOOGL","GOOG","AVGO","SPCX","META","TSLA","MU","WMT","AMD","ASML","INTC","CSCO","AMAT","LRCX","COST","ARM","NFLX","PLTR","KLAC","TXN","PANW","SNDK","LIN","TMUS","MRVL","AMGN","STX","QCOM","CRWD","WDC","PEP","ADI","GILD","SHOP","APP","ISRG","BKNG","VRTX","SBUX","PDD","FTNT","CDNS","ADP","MAR","MNST","MELI","CSX","CEG","ADBE","DDOG","ABNB","CMCSA","SNPS","DASH","INTU","MDLZ","AEP","CTAS","ORLY","NXPI","ROST","HON","REGN","WBD","HONA","PCAR","MPWR","ALAB","LITE","BKR","FANG","TER","FAST","NBIS","EA","XEL","ODFL","EXC","RKLB","CCEP","MCHP","CRWV","FER","TTWO","ADSK","IDXX","AXON","KDP","PYPL","TRI","PAYX","ALNY","ROP","WDAY","MSTR","KHC","GEHC","DXCM","CPRT"]
STATIC_DJIA = ["GS","CAT","UNH","MSFT","AMGN","V","AXP","GOOGL","TRV","HD","JPM","SHW","AAPL","IBM","MCD","JNJ","AMZN","HON","BA","NVDA","CVX","CRM","MMM","PG","MRK","CSCO","WMT","DIS","KO","NKE"]

INDEX_LABELS = {"SP500": "S&P 500", "NDX": "Nasdaq-100", "DJIA": "Dow 30"}


def _fetch_symbols(url: str) -> list[str]:
    """Pull the 'Symbol' column from the first table at a slickcharts/wiki URL."""
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    table = pd.read_html(io.StringIO(resp.text))[0]
    return [str(s).strip().upper().replace(".", "-") for s in table["Symbol"].tolist()]


def _load_index(name: str, sources: list[str], static: list[str]) -> list[str]:
    """Try each live source in turn; fall back to the baked static list."""
    for url in sources:
        try:
            symbols = _fetch_symbols(url)
            if symbols:
                print(f"  {INDEX_LABELS[name]:<11} {len(symbols):>3} tickers (live)")
                return symbols
        except Exception:  # noqa: BLE001 - any network/parse failure -> next source
            continue
    print(f"  {INDEX_LABELS[name]:<11} {len(static):>3} tickers (offline fallback)")
    return static


def build_universe() -> dict[str, set[str]]:
    """Return {ticker: {index labels}} for the combined index universe."""
    print("Fetching index constituents...")
    members: dict[str, list[str]] = {
        "SP500": _load_index(
            "SP500",
            ["https://www.slickcharts.com/sp500",
             "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"],
            STATIC_SP500,
        ),
        "NDX": _load_index(
            "NDX",
            ["https://www.slickcharts.com/nasdaq100"],
            STATIC_NDX,
        ),
        "DJIA": _load_index(
            "DJIA",
            ["https://www.slickcharts.com/dowjones",
             "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"],
            STATIC_DJIA,
        ),
    }
    universe: dict[str, set[str]] = {}
    for index_name, tickers in members.items():
        for ticker in tickers:
            universe.setdefault(ticker, set()).add(index_name)
    print(f"  Combined universe: {len(universe)} unique tickers\n")
    return universe


def download_closes(tickers: list[str]) -> "pd.DataFrame":
    """Batch-download daily closes; return a DataFrame indexed by date, cols=tickers."""
    print(f"Downloading {LOOKBACK} of daily prices for {len(tickers)} tickers...")
    data = yf.download(
        tickers,
        period=LOOKBACK,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data is None or data.empty:
        raise RuntimeError("No price data returned from yfinance.")
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"]
    # Single ticker edge case: wrap into a one-column frame.
    return data[["Close"]].rename(columns={"Close": tickers[0]})


def screen(universe: dict[str, set[str]], closes: "pd.DataFrame") -> list[dict]:
    """Run the buy-signal heuristic on every ticker with enough history."""
    results: list[dict] = []
    skipped = 0
    for ticker in closes.columns:
        prices = closes[ticker].dropna().tolist()
        if len(prices) < MIN_HISTORY:
            skipped += 1
            continue
        try:
            signal = generate_buy_signal(prices)
        except (ValueError, TypeError):
            skipped += 1
            continue
        results.append(
            {
                "ticker": ticker,
                "indices": ",".join(
                    sorted(universe.get(ticker, set()), key=lambda k: list(INDEX_LABELS).index(k))
                ),
                "recommendation": signal["recommendation"],
                "signal_score": signal["signal_score"],
                "rsi": signal["rsi"],
                "trend_strength": signal["trend_strength"],
                "price": prices[-1],
            }
        )
    results.sort(key=lambda row: row["signal_score"], reverse=True)
    if skipped:
        print(f"  Skipped {skipped} tickers (insufficient history or no data)\n")
    return results


def _print_table(title: str, rows: list[dict]) -> None:
    print(title)
    if not rows:
        print("  (none)\n")
        return
    print(f"  {'#':>3}  {'Ticker':<7} {'Indices':<14} {'Signal':>7} {'RSI':>6} {'Price':>10}")
    print(f"  {'-'*3}  {'-'*7} {'-'*14} {'-'*7} {'-'*6} {'-'*10}")
    for rank, row in enumerate(rows, start=1):
        rsi = f"{row['rsi']:.1f}" if row["rsi"] is not None else "n/a"
        print(
            f"  {rank:>3}  {row['ticker']:<7} {row['indices']:<14} "
            f"{row['signal_score']:>+7.2f} {rsi:>6} {row['price']:>10.2f}"
        )
    print()


def main() -> None:
    now = datetime.now()
    print("=" * 66)
    print("Market Buy-Signal Screener  (S&P 500 + Nasdaq-100 + Dow 30)")
    print(f"As of {now:%Y-%m-%d %H:%M} local time")
    print("=" * 66)
    print("Heuristic: 20/50-day MA crossover + RSI. Not financial advice.\n")

    universe = build_universe()
    closes = download_closes(sorted(universe))
    results = screen(universe, closes)

    buys = [r for r in results if r["recommendation"] == "BUY"]
    holds = [r for r in results if r["recommendation"] == "HOLD"]
    sells = [r for r in results if r["recommendation"] == "SELL"]

    print("=" * 66)
    print(f"Scored {len(results)} stocks:  {len(buys)} BUY / {len(holds)} HOLD / {len(sells)} SELL")
    print("=" * 66 + "\n")

    _print_table(f"TOP BUY CANDIDATES ({len(buys)} total, strongest first):", buys[:25])
    _print_table("STRONGEST SELL / AVOID (weakest first):", sells[-10:][::-1])

    csv_path = f"screener_results_{now:%Y%m%d}.csv"
    try:
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"Full ranking of all {len(results)} stocks saved to {csv_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"(Could not write CSV: {exc})")

    print("\nReminder: this is a mechanical trend/momentum score, not a forecast.")
    print("Do your own research before trading.")


if __name__ == "__main__":
    main()
