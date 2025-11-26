import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from nsepython import option_chain
from datetime import datetime

# ---------- Black–Scholes function ----------
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ---------- Step 1: Fetch current option chain ----------
symbol = "NIFTY"
expiry = "28-AUG-2025"  # change to next expiry available on NSE

oc_data = option_chain(symbol)

# Filter for desired expiry after fetching
expiry = "28-Aug-2025"  # note the case & format from NSE
records = oc_data['records']['data']
df_list = []

for rec in records:
    strike = rec['strikePrice']
    if rec['expiryDate'] != expiry:
        continue  # skip if not the desired expiry

    if 'CE' in rec and rec['CE'] is not None:
        df_list.append({
            "strike": strike,
            "type": "call",
            "last_price": rec['CE']['lastPrice'],
            "iv": rec['CE']['impliedVolatility'] / 100,  # % → decimal
            "expiry": rec['CE']['expiryDate']
        })
    if 'PE' in rec and rec['PE'] is not None:
        df_list.append({
            "strike": strike,
            "type": "put",
            "last_price": rec['PE']['lastPrice'],
            "iv": rec['PE']['impliedVolatility'] / 100,
            "expiry": rec['PE']['expiryDate']
        })

options_df = pd.DataFrame(df_list)


# ---------- Step 2: Fetch current spot price ----------
nifty = yf.Ticker("^NSEI")
spot_price = nifty.history(period="1d")["Close"].iloc[-1]

# ---------- Step 3: Compute time to expiry ----------
today = datetime.now()
options_df["expiry_dt"] = pd.to_datetime(options_df["expiry"], format="%d-%b-%Y")
options_df["time_to_expiry"] = (options_df["expiry_dt"] - today).dt.days / 252

# ---------- Step 4: Apply Black–Scholes ----------
risk_free_rate = 0.055  # Repo rate

options_df["bs_price"] = options_df.apply(
    lambda row: black_scholes_price(
        S=spot_price,
        K=row["strike"],
        T=row["time_to_expiry"],
        r=risk_free_rate,
        sigma=row["iv"],
        option_type=row["type"]
    ),
    axis=1
)

# ---------- Step 5: Evaluate ----------
mae = mean_absolute_error(options_df["last_price"], options_df["bs_price"])
rmse = np.sqrt(mean_squared_error(options_df["last_price"], options_df["bs_price"]))

print(f"Spot Price: {spot_price:.2f}")
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# ---------- Step 6: Save results ----------
options_df.to_csv("data/nifty50_bs_comparison.csv", index=False)
print("Saved results to nifty50_bs_comparison.csv")
