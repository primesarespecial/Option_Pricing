import matplotlib.pyplot as plt
import pandas as pd

# Load results
df = pd.read_csv("data/nifty50_bs_comparison.csv")

# Filter out zero IV (illiquid contracts)
df_filtered = df[df['iv'] > 0].copy()

# ---------- Plot 1: Actual vs Predicted ----------
plt.figure(figsize=(8, 6))
plt.scatter(
    df_filtered[df_filtered['type'] == 'call']['last_price'],
    df_filtered[df_filtered['type'] == 'call']['bs_price'],
    alpha=0.6, label='Calls', color='blue'
)
plt.scatter(
    df_filtered[df_filtered['type'] == 'put']['last_price'],
    df_filtered[df_filtered['type'] == 'put']['bs_price'],
    alpha=0.6, label='Puts', color='green'
)

max_val = max(df_filtered['last_price'].max(), df_filtered['bs_price'].max())
plt.plot([0, max_val], [0, max_val], 'r--', label="Perfect Prediction")
plt.xlabel("Market Price (₹)")
plt.ylabel("Black–Scholes Price (₹)")
plt.title("Actual vs Black–Scholes Predicted Prices")
plt.legend()
plt.grid(True)
plt.show()

# ---------- Plot 2: Error vs Strike Price ----------
df_filtered['error'] = df_filtered['bs_price'] - df_filtered['last_price']

plt.figure(figsize=(10, 6))
plt.scatter(
    df_filtered[df_filtered['type'] == 'call']['strike'],
    df_filtered[df_filtered['type'] == 'call']['error'],
    alpha=0.6, label='Calls', color='blue'
)
plt.scatter(
    df_filtered[df_filtered['type'] == 'put']['strike'],
    df_filtered[df_filtered['type'] == 'put']['error'],
    alpha=0.6, label='Puts', color='green'
)

plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Strike Price (₹)")
plt.ylabel("Prediction Error (₹)")
plt.title("Black–Scholes Prediction Error vs Strike Price")
plt.legend()
plt.grid(True)
plt.show()
