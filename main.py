import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import uvicorn
import yfinance as yf
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from statsmodels.tsa.stattools import adfuller
from werkzeug.utils import secure_filename

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="a-very-secret-key-that-should-be-changed")


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CORRELATED_FILE = os.path.join(DATA_DIR, "correlated_pairs.csv")
FINAL_RESULTS_FILE = os.path.join(DATA_DIR, "final_results.csv")
CORRELATED_SECTOR_FILE = os.path.join(DATA_DIR, "correlated_pairs_by_sector.csv")
FINAL_RESULTS_SECTOR_FILE = os.path.join(DATA_DIR, "correlated_pairs_by_sector_backtest_results.csv")

ALLOWED_EXTENSIONS = {"csv"}

templates = Jinja2Templates(directory="templates")


from jinja2 import pass_context


def flash(request: Request, message: str, category: str = "message"):
    if "_messages" not in request.session:
        request.session["_messages"] = []
    request.session["_messages"].append({"message": message, "category": category})

@pass_context
def get_flashed_messages(context, with_categories: bool = False):
    request = context.get("request")
    if request and "_messages" in request.session:
        messages = request.session.pop("_messages")
        if not with_categories:
            return [m["message"] for m in messages]
        return [(m["category"], m["message"]) for m in messages]
    return []

templates.env.globals["get_flashed_messages"] = get_flashed_messages


class SignalParams(BaseModel):
    z_entry: float = Query(2.5, description="Z-score threshold for entry")
    z_exit: float = Query(1.0, description="Z-score threshold for exit")
    z_sl: float = Query(3.0, description="Z-score threshold for stop loss")
    window_size: int = Query(500, description="Window size for regression")


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

def _safe_format(value, format_spec):
    if pd.isna(value) or value == '': return ''
    try: return f"{float(value):{format_spec}}"
    except (ValueError, TypeError): return str(value)



async def download_data(stocks: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    def _fetch():
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

    return await asyncio.to_thread(_fetch)

def evaluate_pair_for_signals(price_df, stock_x, stock_y, asof_dt, params: SignalParams, sector=None, want_entry=False):
    if price_df is None or price_df.empty:
        return None

    # Use params.window_size
    price_df = price_df.loc[price_df.index <= asof_dt].iloc[-params.window_size:]
    if len(price_df) < 30 or stock_x not in price_df.columns or stock_y not in price_df.columns:
        return None

    y = price_df[stock_y]
    X = sm.add_constant(price_df[stock_x])
    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    resid = model.resid
    std_resid = float(np.std(resid))
    if std_resid < 1e-8:
        return None

    latest_day = pd.to_datetime(asof_dt)
    if latest_day not in price_df.index:
        return None

    try:
        y_i = float(price_df.loc[latest_day, stock_y])
        x_i = float(price_df.loc[latest_day, stock_x])
    except Exception:
        return None

    intercept, beta = model.params
    z_score = (y_i - (intercept + beta * x_i)) / std_resid

    # Always try to compute ADF value
    try:
        p_adf = adfuller(resid)[1]
    except Exception:
        p_adf = None

    # If just computing details (want_entry = False), return unconditionally
    if not want_entry:
        return {
            "Date": latest_day.strftime("%d-%m-%Y"),
            "Stock_Y": stock_y,
            "Stock_X": stock_x,
            "Signal": None,
            "Z_Score": round(float(z_score), 2),
            "Beta": round(float(beta), 2),
            "Intercept": round(float(intercept), 2),
            "ADF_P_Value": f"{p_adf:.4f}" if p_adf is not None else "N/A"
        }

    # Signal logic using params
    if abs(z_score) > params.z_sl:
        signal = "Stop Loss"
    elif z_score > params.z_entry:
        signal = "Entry Short"
    elif z_score < -params.z_entry:
        signal = "Entry Long"
    elif -params.z_exit < z_score < params.z_exit:
        signal = "Exit"
    else:
        return None

    # For entry signals, check ADF condition
    if signal.lower().startswith("entry") and (p_adf is None or p_adf >= 0.05):
        return None

    signal_data = {
        "Date": latest_day.strftime("%d-%m-%Y"),
        "Stock_Y": stock_y,
        "Stock_X": stock_x,
        "Signal": signal,
        "Z_Score": round(float(z_score), 2),
        "Beta": round(float(beta), 2),
        "Intercept": round(float(intercept), 2),
        "ADF_P_Value": f"{p_adf:.4f}" if p_adf is not None else "N/A"
    }
    if sector:
        signal_data["Sector"] = sector

    return signal_data

async def evaluate_for_day(pairs_to_process, price_df_all, day, params: SignalParams, want_entry=True):
   
    def _process():
        results = []
        for p in pairs_to_process:
            stock_y, stock_x = p[0], p[1]
            sector = p[2] if len(p) > 2 else None
            sub_df = price_df_all[[stock_x, stock_y]].copy()
            signal = evaluate_pair_for_signals(sub_df, stock_x, stock_y, day, params, sector=sector, want_entry=want_entry)
            if signal:
                results.append(signal)
        return results
    
    return await asyncio.to_thread(_process)



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    files_exist = {f: os.path.exists(p) for f, p in {'correlated': CORRELATED_FILE, 'final': FINAL_RESULTS_FILE, 'correlated_sector': CORRELATED_SECTOR_FILE, 'final_sector': FINAL_RESULTS_SECTOR_FILE}.items()}
    unique_pairs, unique_sector_pairs = [], []
    
    if files_exist['correlated']:
        try:
            df = await asyncio.to_thread(_read_csv_flexible, CORRELATED_FILE)
            if not df.empty: 
                df_norm = _normalize_and_validate_cols(df, ['Stock_Y', 'Stock_X'])
                unique_pairs = df_norm[['Stock_Y', 'Stock_X']].drop_duplicates().to_dict('records')
        except ValueError as e: 
            flash(request, f"Error in correlated_pairs.csv: {e}", "error")
            
    if files_exist['correlated_sector']:
        try:
            df = await asyncio.to_thread(_read_csv_flexible, CORRELATED_SECTOR_FILE)
            if not df.empty: 
                df_norm = _normalize_and_validate_cols(df, ['Stock_Y', 'Stock_X', 'Sector'])
                unique_sector_pairs = df_norm[['Stock_Y', 'Stock_X', 'Sector']].drop_duplicates().to_dict('records')
        except ValueError as e: 
            flash(request, f"Error in correlated_pairs_by_sector.csv: {e}", "error")
            
    return templates.TemplateResponse("index.html", {"request": request, "files_exist": files_exist, "unique_pairs": unique_pairs, "unique_sector_pairs": unique_sector_pairs})

@app.post("/upload_data")
async def upload_data(
    request: Request,
    correlated_file: UploadFile = File(None),
    final_file: UploadFile = File(None),
    correlated_sector_file: UploadFile = File(None),
    final_sector_file: UploadFile = File(None)
):
    files_map = {
        'correlated_file': (correlated_file, CORRELATED_FILE),
        'final_file': (final_file, FINAL_RESULTS_FILE),
        'correlated_sector_file': (correlated_sector_file, CORRELATED_SECTOR_FILE),
        'final_sector_file': (final_sector_file, FINAL_RESULTS_SECTOR_FILE)
    }
    
    saved_any = False
    for form_name, (file_obj, dest_path) in files_map.items():
        if file_obj and file_obj.filename:
            if _allowed_file(file_obj.filename):
                content = await file_obj.read()
                with open(dest_path, "wb") as f:
                    f.write(content)
                saved_any = True
                flash(request, f"Uploaded {os.path.basename(dest_path)} successfully.", "success")
            else:
                flash(request, f"File for {form_name} must be a CSV.", "error")
    
    if not saved_any:
        flash(request, "No valid files were selected for upload.", "warning")
        
    return RedirectResponse(url="/", status_code=303)

@app.get("/get_live_signals")
async def get_live_signals(
    request: Request,
    source: str = "all",
    period: str = "week",
    signal_type: str = "entry",
    date: str = "",
    stock_y: Optional[str] = None,
    stock_x: Optional[str] = None,
    params: SignalParams = Depends()
):
    signal_type = signal_type.lower()
    date_str = date.strip()
    
    if date_str:
        try:
            asof_dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return JSONResponse({"error": f"Invalid date format: {date_str}"}, status_code=400)
    else:
        asof_dt = datetime.now()

    is_sector_mode = source == 'sector'
    file_path = CORRELATED_SECTOR_FILE if is_sector_mode else CORRELATED_FILE
    required_cols = ['Stock_Y', 'Stock_X', 'Sector'] if is_sector_mode else ['Stock_Y', 'Stock_X']
    
    df_raw = await asyncio.to_thread(_read_csv_flexible, file_path)
    if df_raw.empty:
        return JSONResponse({"error": f"{os.path.basename(file_path)} not found or is empty."}, status_code=404)

    try:
        df = _normalize_and_validate_cols(df_raw, required_cols)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if stock_y and stock_x:
        if is_sector_mode:
            match = df[(df['Stock_Y'] == stock_y) & (df['Stock_X'] == stock_x)]
            sector = match['Sector'].iloc[0] if not match.empty else None
            pairs_to_process = [(stock_y, stock_x, sector)]
        else:
            pairs_to_process = [(stock_y, stock_x)]
    else:
        pairs_to_process = list(df[required_cols].itertuples(index=False, name=None))

    all_signals = []
    unique_stocks = set([p[0] for p in pairs_to_process] + [p[1] for p in pairs_to_process])
    

    price_start_date = (asof_dt - timedelta(days=params.window_size + 20)).strftime("%Y-%m-%d")
    price_end_date = (asof_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    
    price_df_all = await download_data(list(unique_stocks), price_start_date, price_end_date)

    if price_df_all is None or price_df_all.empty:
        return JSONResponse({"error": "Price data could not be loaded."}, status_code=404)

    if period == 'today':
        all_signals = await evaluate_for_day(pairs_to_process, price_df_all, asof_dt, params, want_entry=True)
    else:
        current_day = asof_dt
        tasks = []
        for _ in range(7):
            if current_day.weekday() < 5 and current_day in price_df_all.index:
                tasks.append(evaluate_for_day(pairs_to_process, price_df_all, current_day, params, want_entry=True))
            current_day -= timedelta(days=1)
        
        results = await asyncio.gather(*tasks)
        for res in results:
            all_signals.extend(res)

    if signal_type == 'all':
        filtered_signals = [s for s in all_signals if s and s.get('Signal')]
    elif signal_type == 'exit':
        filtered_signals = [s for s in all_signals if s and s.get('Signal') and ('exit' in s['Signal'].lower() or 'stop' in s['Signal'].lower())]
    else:
        filtered_signals = [s for s in all_signals if s and s.get('Signal') and signal_type in s['Signal'].lower()]

    if period == 'today':
        final_signals = [s for s in filtered_signals if datetime.strptime(s['Date'], '%d-%m-%Y').date() == asof_dt.date()]
    else:
        final_signals = filtered_signals

    return final_signals

@app.get("/get_pair_details")
async def get_pair_details(
    request: Request,
    stock_y: str,
    stock_x: str,
    date: str,
    period: str = "today",
    params: SignalParams = Depends()
):
    date_str = date.strip()
    if not stock_y or not stock_x or not date_str:
        return JSONResponse({"error": "Missing stock_y, stock_x or date parameter."}, status_code=400)

    try:
        asof_dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return JSONResponse({"error": f"Invalid date format: {date_str}"}, status_code=400)

    async def get_details_for_date(target_date):
        start_date = (target_date - timedelta(days=params.window_size + 20)).strftime("%Y-%m-%d")
        end_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")
        price_df = await download_data([stock_x, stock_y], start_date, end_date)
        
        if price_df is None or price_df.empty:
            return None
            
        def _process():
            price_df.index = pd.to_datetime(price_df.index)
            valid_days = price_df.index[price_df.index <= target_date]
            if valid_days.empty:
                return None
            latest_valid_day = valid_days[-1]
            return evaluate_pair_for_signals(price_df, stock_x, stock_y, latest_valid_day, params, want_entry=False)
            
        return await asyncio.to_thread(_process)

    if period == "week":
        details_list = []
        current_day = asof_dt
        days_checked = 0
       
        
        start_date_all = (asof_dt - timedelta(days=params.window_size + 30)).strftime("%Y-%m-%d")
        end_date_all = (asof_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        price_df_all = await download_data([stock_x, stock_y], start_date_all, end_date_all)
        
        if price_df_all is None or price_df_all.empty:
             return JSONResponse({"error": "No valid trading data found for the selected week."}, status_code=404)
        
        price_df_all.index = pd.to_datetime(price_df_all.index)
        
        while len(details_list) < 5 and days_checked < 10:
            if current_day.weekday() < 5:
                # Check if we have data for this day
                valid_days = price_df_all.index[price_df_all.index <= current_day]
                if not valid_days.empty:
                    latest_valid_day = valid_days[-1]
                    
                    detail = await asyncio.to_thread(evaluate_pair_for_signals, price_df_all, stock_x, stock_y, latest_valid_day, params, None, False)
                    if detail:
                         # Check if we already have this date (to avoid duplicates from holidays)
                         if not any(d['Date'] == detail['Date'] for d in details_list):
                             details_list.append(detail)
            current_day -= timedelta(days=1)
            days_checked += 1

        if not details_list:
            return JSONResponse({"error": "No valid trading data found for the selected week."}, status_code=404)
        return details_list

    # Default: period == today
    detail = await get_details_for_date(asof_dt)
    if not detail:
        for offset in range(1, 5):
            fallback_day = asof_dt - timedelta(days=offset)
            detail = await get_details_for_date(fallback_day)
            if detail:
                break
    if not detail:
        return JSONResponse({"error": f"No valid data found near {asof_dt.strftime('%d-%m-%Y')}"}, status_code=404)
    return [detail]

@app.get("/analyze/{stock_y}/{stock_x}", response_class=HTMLResponse)
async def analyze_pair(request: Request, stock_y: str, stock_x: str, source: str = "all"):
    is_sector_mode = source == 'sector'
    file_path = FINAL_RESULTS_SECTOR_FILE if is_sector_mode else FINAL_RESULTS_FILE
    
    df_raw = await asyncio.to_thread(_read_csv_flexible, file_path)
    if df_raw.empty: 
        return HTMLResponse(f"Historical results file '{os.path.basename(file_path)}' not found or empty.", status_code=404)
    
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
        return HTMLResponse(str(e), status_code=400)
        
    want_y, want_x = str(stock_y).strip().upper(), str(stock_x).strip().upper()
    
    
    def _process_analysis():
        pair_df = df.loc[(df["Stock_Y"].astype(str).str.strip().str.upper() == want_y) & (df["Stock_X"].astype(str).str.strip().str.upper() == want_x)].copy()
        if pair_df.empty: return None
        
        def _parse_date_series(s: pd.Series):
            dt = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
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
        if pair_df.empty: return "All matched rows had invalid Exit_Date or Trade_PnL."
        
        pair_df.sort_values("Exit_Date", inplace=True, ignore_index=True)
        pair_df["_cum_pnl"] = pair_df["_pnl"].cumsum()
        pnl = pair_df["_pnl"]
        
        summary = {
            "pair": f"{want_y} / {want_x}", 
            "total_pnl": f"{pnl.sum():,.2f}", 
            "total_trades": len(pnl), 
            "win_rate": f"{(pnl > 0).sum() / len(pnl) * 100:.2f}%" if len(pnl) > 0 else "0.00%", 
            "avg_pnl": f"{pnl.mean():,.2f}" if len(pnl) > 0 else "0.00", 
            "rows_dropped": rows_dropped, 
            "source": "Sector-wise" if is_sector_mode else "All Pairs"
        }
        chart_data = {
            "labels": pair_df["Exit_Date"].dt.strftime("%d-%m-%Y").tolist(), 
            "cumulative_pnl": pair_df["_cum_pnl"].round(2).tolist(), 
            "trade_pnl": pair_df["_pnl"].round(2).tolist()
        }
        
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
        return summary, chart_data, table_rows

    result = await asyncio.to_thread(_process_analysis)
    
    if result is None:
        return HTMLResponse(f"No historical trades found for {want_y} / {want_x} in the selected data source.", status_code=404)
    if isinstance(result, str):
        return HTMLResponse(result, status_code=422)
        
    summary, chart_data, table_rows = result
    return templates.TemplateResponse("analysis.html", {"request": request, "summary": summary, "chart_data": chart_data, "table_rows": table_rows, "is_sector_mode": is_sector_mode})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
