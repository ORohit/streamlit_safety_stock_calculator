import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# -------------------------------
# Safety Stock Calculation
# -------------------------------
def safety_stock(service_level, demand_mean, demand_std, lt_mean, lt_std):
    """Calculates safety stock based on normal distribution."""
    z = norm.ppf(service_level)  # z-score for service level
    std_dlt = np.sqrt((lt_mean * demand_std**2) + ((demand_mean**2) * (lt_std**2)))
    return z * std_dlt

# -------------------------------
# Streamlit App
# -------------------------------

st.set_page_config(page_title="Safety Stock Calculator", page_icon="o")

st.title("2-Tier Inventory Flow Network: Safety Stock Calculator")

st.markdown("""
This app calculates safety stock for **2-tier network** represented below:
""")

st.image("2-tier network example.jpg", use_container_width=True)

st.subheader("Assumptions:")

st.markdown("""
1. Customer demand at stores and lead times follow normal distributions.
2. All stores have the same demand and lead time distribution from DC.

We calculate **safety stock** at each location using normal distribution assumptions.  
👉 The **DC’s service level affects the effective lead time distribution at Stores**.
""")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Input Parameters")

# Service levels
dc_sl = st.sidebar.slider("DC Service Level", 0.5, 0.99, 0.80)
store_sl = st.sidebar.slider("Store Service Level", 0.5, 0.99, 0.95)

# Demand & Lead time assumptions
st.sidebar.subheader("Store Assumptions")
store_demand_mean = st.sidebar.number_input("Store Mean Daily Demand (units)", 10, 5000, 100)
store_demand_std = st.sidebar.number_input("Store Demand Std Dev (units)", 1, 500, 20)
store_lt_mean_base = st.sidebar.number_input("Store Base Lead Time Mean (days)", 1, 20, 5)
store_lt_std_base = st.sidebar.number_input("Store Base Lead Time Std Dev (days)", 0, 10, 2)

st.sidebar.subheader("DC Assumptions")
dc_demand_mean = store_demand_mean * 3  # aggregate demand from 3 Stores
dc_demand_std = np.sqrt(3) * store_demand_std
dc_lt_mean = st.sidebar.number_input("DC Lead Time Mean (days)", 1, 60, 10)   # default updated
dc_lt_std = st.sidebar.number_input("DC Lead Time Std Dev (days)", 0, 30, 4)  # default updated

# -------------------------------
# Incorporating DC service level impact on Store lead time
# -------------------------------
store_lt_mean_effective = store_lt_mean_base + (1 - dc_sl) * dc_lt_mean
store_lt_var_effective = (store_lt_std_base**2) + ((1 - dc_sl)**2) * (dc_lt_std**2)
store_lt_std_effective = np.sqrt(store_lt_var_effective)

# -------------------------------
# Calculations
# -------------------------------
store_ss = [safety_stock(store_sl, store_demand_mean, store_demand_std,
                         store_lt_mean_effective, store_lt_std_effective) for _ in range(3)]
dc_ss = safety_stock(dc_sl, dc_demand_mean, dc_demand_std, dc_lt_mean, dc_lt_std)
total_ss = dc_ss + sum(store_ss)

# -------------------------------
# Results
# -------------------------------

st.subheader("Safety Stock Results: ")

ss_data = {
    "Node": ["DC", "Store 1", "Store 2", "Store 3", "Total"],
    "Safety Stock (Units)": [int(dc_ss)] + [int(ss) for ss in store_ss] + [int(total_ss)]
}
df_ss = pd.DataFrame(ss_data)

# Drop index so no row numbers show
df_ss = df_ss.reset_index(drop=True)

# Function to style the Total row
def highlight_total(row):
    if row["Node"] == "Total":
        return ["background-color: #F8F9FB; color: black; font-weight: bold; text-align: center"]*2
    else:
        return ["text-align: center"]*2

# Apply styling
styled_df = (
    df_ss.style
    .format({"SAFETY STOCK (UNITS)": "{:,.0f}"})
    .apply(highlight_total, axis=1)
    .set_table_styles([
        {"selector": "th", "props": [("text-align", "center"),
                                     ("font-weight", "bold")]}  # bold & center headers
    ])
)

st.dataframe(styled_df, use_container_width=True, hide_index=True)


## Lead Time Table
# st.subheader("Store Lead Time Statistics: ")

# lt_data = {
    # "Metric": ["Mean (days)", "Std Dev (days)"],
    # "Base": [f"{store_lt_mean_base:,.2f}", f"{store_lt_std_base:,.2f}"],
    # "Effective": [f"{store_lt_mean_effective:,.2f}", f"{store_lt_std_effective:,.2f}"]
# }
# df_lt = pd.DataFrame(lt_data)

# # Drop index
# df_lt = df_lt.reset_index(drop=True)

# # Style: center everything
# styled_lt = (
    # df_lt.style
    # .set_table_styles(
        # [{"selector": "th", "props": [("text-align", "center")]},
         # {"selector": "td", "props": [("text-align", "center")]}],
        # overwrite=False
    # )
# )

# st.dataframe(styled_lt, use_container_width=True, hide_index=True)


# -------------------------------
# Bar Chart with Labels
# -------------------------------
labels = ["DC", "Store 1", "Store 2", "Store 3", "Total"]
values = [dc_ss] + store_ss + [total_ss]

fig, ax = plt.subplots()
bars = ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:orange", "tab:orange", "tab:green"])

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom')

ax.set_ylabel("Safety Stock (units)")
# ax.set_title("Safety Stock Comparison (Impact of DC SL on Store Lead Times)")
st.pyplot(fig)

# -------------------------------
# Histogram of Base vs Effective Store Lead Times
# -------------------------------
st.subheader("Distribution of Store Lead Times (Base vs Effective):")

# Simulate samples
base_samples = np.random.normal(store_lt_mean_base, store_lt_std_base, 1000)
effective_samples = np.random.normal(store_lt_mean_effective, store_lt_std_effective, 1000)

fig2, ax2 = plt.subplots()
ax2.hist(base_samples, bins=20, alpha=0.6, color="orange", edgecolor="black", label="Base LT")
ax2.hist(effective_samples, bins=20, alpha=0.6, color="skyblue", edgecolor="black", label="Effective LT")
ax2.set_xlabel("Lead Time (days)")
ax2.set_ylabel("Frequency")
ax2.set_title("Histogram of Store Lead Times")
ax2.legend()
st.pyplot(fig2)

