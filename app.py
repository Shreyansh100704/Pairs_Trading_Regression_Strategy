from flask import (
    Flask, render_template, jsonify, request,
    redirect, url_for, flash
)
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.secret_key = "a-very-secret-key-that-should-be-changed"


DATA_DIR = os.path.join(app.root_path, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CORRELATED_FILE = os.path.join(DATA_DIR, "correlated_pairs.csv")
FINAL_RESULTS_FILE = os.path.join(DATA_DIR, "final_results.csv")
CORRELATED_SECTOR_FILE = os.path.join(DATA_DIR, "correlated_pairs_by_sector.csv")
FINAL_RESULTS_SECTOR_FILE = os.path.join(DATA_DIR, "correlated_pairs_by_sector_backtest_results.csv")

ALLOWED_EXTENSIONS = {"csv"}

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _read_csv_flexible(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "ISO-8859-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return pd.DataFrame()

def _normalize_and_validate_cols(df, required_cols, optional_cols=[]):
    canonical_map = {
        'stocky': 'Stock_Y', 'stockx': 'Stock_X', 'sector': 'Sector',
        'correlation': 'Correlation', 'sign': 'Sign', 'exitdate': 'Exit_Date',
        'tradepnl': 'Trade_PnL', 'entrydate': 'Entry_Date',
        'adfpvalueentry': 'ADF_PValue_Entry', 'zscoreentry': 'ZScore_Entry',
        'beta': 'Beta', 'position': 'Position', 'adfpvalueexit': 'ADF_PValue_Exit',
        'zscoreexit': 'ZScore_Exit', 'intercept': 'Intercept',
        'stderresidual': 'StdErr_Residual', 'entryopenpricex': 'Entry_Open_Price_X',
        'exitopenpricex': 'Exit_Open_Price_X', 'entryopenpricey': 'Entry_Open_Price_Y',
        'exitopenpricey': 'Exit_Open_Price_Y', 'lotsizex': 'Lot_Size_X',
        'lotsizey': 'Lot_Size_Y', 'returns': 'Returns', 'cumulativepnl': 'Cumulative_PnL'
    }
    rename_dict = {col: canonical_map[key] for col in df.columns for key in [col.lower().replace(" ", "").replace("_", "")] if key in canonical_map}
    df_renamed = df.rename(columns=rename_dict)
    
    missing = [col for col in required_cols if col not in df_renamed.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}. Original columns found: {list(df.columns)}")
    
    return df_renamed


def download_data(stocks, start_date, end_date):
    try:
        df = yf.download(stocks, start=start_date, end=end_date, progress=False, auto_adjust=True, actions=False)["Close"]
    except Exception:
        return None
    if isinstance(df, pd.Series):
        df = df.to_frame(name=stocks[0] if len(stocks)==1 else None)
    df.ffill(inplace=True)
    df.dropna(axis=0, how="any", inplace=True)
    if df.empty: return None
    df.index = pd.to_datetime(df.index)
    return df


Z_ENTRY, Z_EXIT = 2.5, 1

def evaluate_pair_for_signals(price_df, stock_x, stock_y, asof_dt, all_signals=True, sector=None):
    if price_df is None or price_df.empty: return None
    price_df = price_df.loc[price_df.index <= asof_dt]
    if len(price_df) < 30 or stock_x not in price_df.columns or stock_y not in price_df.columns: return None
    y, X = price_df[stock_y], sm.add_constant(price_df[stock_x])
    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None
    resid, std_resid = model.resid, float(np.std(model.resid))
    if std_resid < 1e-8: return None
    latest_day = price_df.index[-1]
    y_i, x_i = float(price_df.loc[latest_day, stock_y]), float(price_df.loc[latest_day, stock_x])
    intercept, beta = model.params
    z_score = (y_i - (intercept + beta * x_i)) / std_resid
    if z_score > Z_ENTRY: signal = "Entry Short"
    elif z_score < -Z_ENTRY: signal = "Entry Long"
    elif -Z_EXIT < z_score < Z_EXIT: signal = "Exit"
    else: return None
    try:
        p_adf = adfuller(resid)[1]
        if p_adf >= 0.05: return None
    except Exception:
        return None
    signal_data = {
        "Date": latest_day.strftime("%d-%m-%Y"), "Stock_Y": stock_y, "Stock_X": stock_x, 
        "Signal": signal, "Z_Score": round(float(z_score), 2), 
        "Beta": round(float(beta), 2), "Intercept": round(float(intercept), 2), "ADF_P_Value": f"{p_adf:.4f}"
    }
    if sector: signal_data["Sector"] = sector
    return signal_data

def run_check_for_pair_worker(pair_tuple, asof_dt):
    stock_y, stock_x, sector = pair_tuple[0], pair_tuple[1], (pair_tuple[2] if len(pair_tuple) > 2 else None)
    end_date, start_date = asof_dt + timedelta(days=1), asof_dt - timedelta(days=500)
    price_df = download_data([stock_x, stock_y], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    return evaluate_pair_for_signals(price_df, stock_x, stock_y, asof_dt, sector) if price_df is not None else None

def _safe_format(value, format_spec):
    if pd.isna(value) or value == '': return ''
    try: return f"{float(value):{format_spec}}"
    except (ValueError, TypeError): return str(value)

#Flask Routes
@app.route("/")
def index():
    files_exist = {f: os.path.exists(p) for f, p in {'correlated': CORRELATED_FILE, 'final': FINAL_RESULTS_FILE, 'correlated_sector': CORRELATED_SECTOR_FILE, 'final_sector': FINAL_RESULTS_SECTOR_FILE}.items()}
    unique_pairs, unique_sector_pairs = [], []
    if files_exist['correlated']:
        try:
            df = _read_csv_flexible(CORRELATED_FILE)
            if not df.empty: unique_pairs = _normalize_and_validate_cols(df, ['Stock_Y', 'Stock_X'])[['Stock_Y', 'Stock_X']].drop_duplicates().to_dict('records')
        except ValueError as e: flash(f"Error in correlated_pairs.csv: {e}", "error")
    if files_exist['correlated_sector']:
        try:
            df = _read_csv_flexible(CORRELATED_SECTOR_FILE)
            if not df.empty: unique_sector_pairs = _normalize_and_validate_cols(df, ['Stock_Y', 'Stock_X', 'Sector'])[['Stock_Y', 'Stock_X', 'Sector']].drop_duplicates().to_dict('records')
        except ValueError as e: flash(f"Error in correlated_pairs_by_sector.csv: {e}", "error")
    return render_template("index.html", files_exist=files_exist, unique_pairs=unique_pairs, unique_sector_pairs=unique_sector_pairs)

@app.route("/upload_data", methods=["POST"])
def upload_data():
    files_to_upload = {'correlated_file': CORRELATED_FILE, 'final_file': FINAL_RESULTS_FILE, 'correlated_sector_file': CORRELATED_SECTOR_FILE, 'final_sector_file': FINAL_RESULTS_SECTOR_FILE}
    saved_any = False
    for form_name, file_path in files_to_upload.items():
        if (file := request.files.get(form_name)) and file.filename:
            if _allowed_file(file.filename):
                file.save(file_path); saved_any = True; flash(f"Uploaded {os.path.basename(file_path)} successfully.", "success")
            else: flash(f"File for {form_name} must be a CSV.", "error")
    if not saved_any: flash("No valid files were selected for upload.", "warning")
    return redirect(url_for("index"))

@app.route("/get_live_signals")
def get_live_signals():
    source = request.args.get('source', 'all')
    period = request.args.get('period', 'week')
    signal_type = request.args.get('signal_type', 'entry').lower()
    date_str = request.args.get("date", "").strip()

    try:
        asof_dt = datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime.now()
    except ValueError:
        return jsonify({"error": f"Invalid date format: {date_str}"}), 400

    is_sector_mode = source == 'sector'
    file_path = CORRELATED_SECTOR_FILE if is_sector_mode else CORRELATED_FILE
    required_cols = ['Stock_Y', 'Stock_X', 'Sector'] if is_sector_mode else ['Stock_Y', 'Stock_X']
    df_raw = _read_csv_flexible(file_path)
    if df_raw.empty:
        return jsonify({"error": f"{os.path.basename(file_path)} not found or is empty."}), 404

    try:
        df = _normalize_and_validate_cols(df_raw, required_cols)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    stock_y_q = request.args.get('stock_y')
    stock_x_q = request.args.get('stock_x')
    if stock_y_q and stock_x_q:
        if is_sector_mode:
            match = df[(df['Stock_Y'] == stock_y_q) & (df['Stock_X'] == stock_x_q)]
            sector = match['Sector'].iloc[0] if not match.empty else None
            pairs_to_process = [(stock_y_q, stock_x_q, sector)]
        else:
            pairs_to_process = [(stock_y_q, stock_x_q)]
    else:
        pairs_to_process = list(df[required_cols].itertuples(index=False, name=None))

    #Entry/Exit Logic
    all_signals = []

    # Collect all stocks for bulk download
    unique_stocks = set([p[0] for p in pairs_to_process] + [p[1] for p in pairs_to_process])
    price_start_date = (asof_dt - timedelta(days=500)).strftime("%Y-%m-%d")
    price_end_date = (asof_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    price_df_all = download_data(list(unique_stocks), price_start_date, price_end_date)

    if price_df_all is None or price_df_all.empty:
        return jsonify({"error": "Price data could not be loaded."}), 404

    def evaluate_for_day(day):
        results = []
        for p in pairs_to_process:
            stock_y, stock_x = p[0], p[1]
            sector = p[2] if len(p) > 2 else None
            sub_df = price_df_all[[stock_x, stock_y]].copy()
            signal = evaluate_pair_for_signals(sub_df, stock_x, stock_y, day, sector=sector)
            if signal:
                results.append(signal)
        return results

    if period == 'today':
        all_signals = evaluate_for_day(asof_dt)
    else:
        # Past 7 trading days (skip weekends)
        current_day = asof_dt
        for _ in range(7):
            if current_day.weekday() < 5 and current_day in price_df_all.index:
                all_signals.extend(evaluate_for_day(current_day))
            current_day -= timedelta(days=1)

    filtered_signals = [s for s in all_signals if signal_type in s['Signal'].lower()] if signal_type != 'all' else all_signals

    if period == 'today':
        final_signals = [s for s in filtered_signals if datetime.strptime(s['Date'], '%d-%m-%Y').date() == asof_dt.date()]
    else:
        final_signals = filtered_signals  

    return jsonify(final_signals)


@app.route("/get_pair_details")
def get_pair_details():
    stock_y = request.args.get('stock_y')
    stock_x = request.args.get('stock_x')
    date_str = request.args.get("date", "").strip()
    source = request.args.get('source', 'all')

    if not stock_y or not stock_x or not date_str:
        return jsonify({"error": "Missing stock_y, stock_x or date parameter."}), 400

    try:
        asof_dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": f"Invalid date format: {date_str}"}), 400

    # Load price data for both stocks
    start_date = (asof_dt - timedelta(days=500)).strftime("%Y-%m-%d")
    end_date = (asof_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    price_df = download_data([stock_x, stock_y], start_date, end_date)
    if price_df is None or price_df.empty:
        return jsonify({"error": "No price data found."}), 404

    if asof_dt not in price_df.index:
        return jsonify({"error": f"No data for {asof_dt.strftime('%d-%m-%Y')}"}), 404

    # OLS regression
    y = price_df[stock_y]
    X = sm.add_constant(price_df[stock_x])
    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        return jsonify({"error": f"OLS regression failed: {e}"}), 500

    resid = model.resid
    try:
        p_adf = adfuller(resid)[1]
    except Exception:
        p_adf = None

    y_val = float(price_df.loc[asof_dt, stock_y])
    x_val = float(price_df.loc[asof_dt, stock_x])
    intercept, beta = model.params
    std_resid = float(np.std(resid))
    z_score = (y_val - (intercept + beta * x_val)) / std_resid if std_resid > 1e-8 else 0.0

    details = {
        "Date": asof_dt.strftime("%d-%m-%Y"),
        "Stock_Y": stock_y,
        "Stock_X": stock_x,
        "ADF_Value": round(p_adf, 4) if p_adf is not None else "N/A",
        "Z_Score": round(z_score, 2),
        "Beta": round(beta, 2),
        "Intercept": round(intercept, 2)
    }
    return jsonify([details])


@app.route("/analyze/<stock_y>/<stock_x>")
def analyze_pair(stock_y, stock_x):
    source = request.args.get('source', 'all')
    is_sector_mode = source == 'sector'
    file_path = FINAL_RESULTS_SECTOR_FILE if is_sector_mode else FINAL_RESULTS_FILE
    df_raw = _read_csv_flexible(file_path)
    if df_raw.empty: return f"Historical results file '{os.path.basename(file_path)}' not found or empty.", 404
    
    required = ['Stock_Y', 'Stock_X', 'Exit_Date', 'Trade_PnL']
    optional = [
        'Entry_Date', 'ADF_PValue_Entry', 'ZScore_Entry', 'Beta', 'Position', 
        'ADF_PValue_Exit', 'ZScore_Exit', 'Intercept', 'StdErr_Residual', 
        'Entry_Open_Price_X', 'Exit_Open_Price_X', 'Entry_Open_Price_Y', 
        'Exit_Open_Price_Y', 'Lot_Size_X', 'Lot_Size_Y', 'Returns'
    ]
    if is_sector_mode: required.append('Sector')
    
    try:
        df = _normalize_and_validate_cols(df_raw, required, optional)
    except ValueError as e:
        return str(e), 400
        
    want_y, want_x = str(stock_y).strip().upper(), str(stock_x).strip().upper()
    pair_df = df.loc[(df["Stock_Y"].astype(str).str.strip().str.upper() == want_y) & (df["Stock_X"].astype(str).str.strip().str.upper() == want_x)].copy()
    if pair_df.empty: return f"No historical trades found for {want_y} / {want_x} in the selected data source.", 404
    
    
    def _parse_date_series(s: pd.Series):
        # First, try the explicit DD-MM-YYYY format.
        dt = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
        # For any that failed, fall back to a more general parser.
        if dt.isna().any():
            dt[dt.isna()] = pd.to_datetime(s[dt.isna()], errors="coerce", dayfirst=True)
        return dt

    for date_col in ['Exit_Date', 'Entry_Date']:
        if date_col in pair_df.columns and not pd.api.types.is_datetime64_any_dtype(pair_df[date_col]):
            pair_df[date_col] = _parse_date_series(pair_df[date_col])
            
    pair_df["_pnl"] = pd.to_numeric(pair_df["Trade_PnL"], errors="coerce")
    rows_dropped = len(pair_df)
    pair_df.dropna(subset=["Exit_Date", "_pnl"], inplace=True)
    rows_dropped -= len(pair_df)
    if pair_df.empty: return "All matched rows had invalid Exit_Date or Trade_PnL.", 422
    
    pair_df.sort_values("Exit_Date", inplace=True, ignore_index=True)
    pair_df["_cum_pnl"] = pair_df["_pnl"].cumsum()
    pnl = pair_df["_pnl"]
    summary = {"pair": f"{want_y} / {want_x}", "total_pnl": f"{pnl.sum():,.2f}", "total_trades": len(pnl), "win_rate": f"{(pnl > 0).sum() / len(pnl) * 100:.2f}%" if len(pnl) > 0 else "0.00%", "avg_pnl": f"{pnl.mean():,.2f}" if len(pnl) > 0 else "0.00", "rows_dropped": rows_dropped, "source": "Sector-wise" if is_sector_mode else "All Pairs"}
    chart_data = {"labels": pair_df["Exit_Date"].dt.strftime("%Y-%m-%d").tolist(), "cumulative_pnl": pair_df["_cum_pnl"].round(2).tolist(), "trade_pnl": pair_df["_pnl"].round(2).tolist()}
    
    table_rows = []
    for _, r in pair_df.iterrows():
        rec = {
            'Stock_Y': r.get('Stock_Y'), 'Stock_X': r.get('Stock_X'),
            'Entry_Date': r['Entry_Date'].strftime('%d-%m-%Y') if pd.notna(r.get('Entry_Date')) else '',
            'Exit_Date': r['Exit_Date'].strftime('%d-%m-%Y') if pd.notna(r.get('Exit_Date')) else '',
            'Trade_PnL': _safe_format(r.get('Trade_PnL'), ',.2f'),
            'Cumulative_PnL': _safe_format(r.get('_cum_pnl'), ',.2f'),
            'ADF_Entry': _safe_format(r.get('ADF_PValue_Entry'), '.4f'),
            'ADF_Exit': _safe_format(r.get('ADF_PValue_Exit'), '.4f'),
            'Z_Entry': _safe_format(r.get('ZScore_Entry'), ',.2f'),
            'Z_Exit': _safe_format(r.get('ZScore_Exit'), ',.2f'),
            'Beta': _safe_format(r.get('Beta'), ',.2f'),
            'Position': r.get('Position'),
            'Intercept': _safe_format(r.get('Intercept'), ',.2f'),
            'StdErr_Residual': _safe_format(r.get('StdErr_Residual'), ',.2f'),
            'Entry_Open_Price_X': _safe_format(r.get('Entry_Open_Price_X'), ',.2f'),
            'Exit_Open_Price_X': _safe_format(r.get('Exit_Open_Price_X'), ',.2f'),
            'Entry_Open_Price_Y': _safe_format(r.get('Entry_Open_Price_Y'), ',.2f'),
            'Exit_Open_Price_Y': _safe_format(r.get('Exit_Open_Price_Y'), ',.2f'),
            'Lot_Size_X': _safe_format(r.get('Lot_Size_X'), ',.0f'),
            'Lot_Size_Y': _safe_format(r.get('Lot_Size_Y'), ',.0f'),
            'Returns': _safe_format(r.get('Returns'), '.4f'),
        }
        if is_sector_mode: rec['Sector'] = r.get('Sector')
        table_rows.append(rec)
    return render_template("analysis.html", summary=summary, chart_data=chart_data, table_rows=table_rows, is_sector_mode=is_sector_mode)



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
