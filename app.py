import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Data Analyst Pro", page_icon="📈", layout="wide")

st.title("🤖 AI Data Analyst: From Messy CSV to Business Insights")
st.markdown("""
**System Workflow:**
1.  **Audit:** Identify Nulls & Errors.
2.  **Clean:** Apply Mean/Median (Numeric) and Mode (Categorical).
3.  **Visualize:** Analyze Trends, Profit, and Correlations.
4.  **Recommend:** AI-driven business improvements.
""")

# --- HELPER: DATA HEALTH CHECK ---
def get_health_check(df):
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    return nulls

# --- MAIN APP ---
uploaded_file = st.file_uploader("📂 Upload your Raw Business Data (CSV)", type=["csv"])

if uploaded_file is not None:
    # 1. LOAD DATA
    df = pd.read_csv(uploaded_file)
    
    # Standardize headers immediately for easier processing
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    st.header("Step 1: Data Audit & Cleaning Log")
    
    # --- STEP-BY-STEP CLEANING LOGIC ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔴 Before Cleaning")
        initial_nulls = get_health_check(df)
        if not initial_nulls.empty:
            st.write(initial_nulls)
        else:
            st.success("No missing values found initially!")
            
    # AUTO-CLEANING ENGINE
    with st.expander("⚙️ View Cleaning Operations (How we fixed it)", expanded=True):
        
        # 1. Handling Numeric Data (Mean/Median)
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                # Check for outliers to decide Mean vs Median
                if df[col].skew() > 1:
                    fill_val = df[col].median()
                    method = "Median (due to skew)"
                else:
                    fill_val = df[col].mean()
                    method = "Mean (normal distribution)"
                
                df[col] = df[col].fillna(fill_val)
                st.write(f"✅ **{col}**: Filled missing values with **{method}** : `{round(fill_val, 2)}`")

        # 2. Handling Categorical Data (Mode)
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                st.write(f"✅ **{col}**: Filled missing values with **Mode (Most Frequent)** : `'{mode_val}'`")

        # 3. Date Detection (for Trends)
        date_cols = [col for col in df.columns if 'date' in col or 'time' in col]
        if date_cols:
            st.write(f"📅 Detected Time Column: `{date_cols[0]}` - Converting to DateTime format.")
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df = df.sort_values(by=date_cols[0])

    with col2:
        st.subheader("🟢 After Cleaning")
        final_nulls = get_health_check(df)
        if final_nulls.empty:
            st.success("Data is 100% Clean! (0 Nulls)")
        else:
            st.warning("Some complex nulls remain.")

    st.markdown("---")

    # --- STEP 2: ADVANCED VISUALIZATION ---
    st.header("Step 2: Business Intelligence & Visualization")
    
    tab1, tab2, tab3 = st.tabs(["📊 Market Trends", "💰 Profit & Loss Analysis", "🔬 Correlations"])

    # Attempt to identify key business columns
    profit_col = next((col for col in df.columns if 'profit' in col), None)
    sales_col = next((col for col in df.columns if 'sales' in col or 'revenue' in col or 'amount' in col), None)
    category_col = next((col for col in cat_cols if 'category' in col or 'segment' in col or 'region' in col), None)

    with tab1:
        st.subheader("Market Trends Over Time")
        if date_cols and sales_col:
            # Resample by Month to show Trend
            monthly_sales = df.set_index(date_cols[0])[sales_col].resample('M').sum()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=monthly_sales, marker='o', color='purple', ax=ax)
            ax.set_title("Sales Trend (Monthly Aggregated)")
            ax.set_ylabel("Total Sales")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            
            st.caption("💡 **Insight:** Peaks in the line chart indicate high-demand seasons. Valleys indicate off-seasons where marketing should be increased.")
        else:
            st.info("Could not detect 'Date' or 'Sales' columns automatically. Ensure columns are named clearly (e.g., 'Order Date', 'Sales').")

    with tab2:
        st.subheader("Profitability & Category Analysis")
        if profit_col and category_col:
            fig, ax = plt.subplots(figsize=(10, 5))
            # Calculate total profit by category
            cat_profit = df.groupby(category_col)[profit_col].sum().sort_values()
            
            # Color code: Red for loss, Green for profit
            colors = ['red' if x < 0 else 'green' for x in cat_profit.values]
            sns.barplot(x=cat_profit.values, y=cat_profit.index, palette=colors, ax=ax)
            ax.set_title(f"Total Profit by {category_col.title()}")
            st.pyplot(fig)
            
            st.caption("💡 **Insight:** Categories in Red are loss-making. Consider discontinuing them or raising prices. Categories in Green are your 'Cash Cows'.")
        else:
            st.info("Need 'Profit' and 'Category' columns to generate P&L charts.")

    with tab3:
        st.subheader("Deep Dive: Correlations")
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric data for correlation.")

    # --- STEP 3: STRATEGIC RECOMMENDATIONS ---
    st.markdown("---")
    st.header("Step 3: Strategic Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.info("📌 **Data Quality Recommendations**")
        st.markdown("""
        * **Standardization:** Ensure all text inputs are standardized (e.g., 'NY', 'N.Y.', 'New York' should all be 'New York').
        * **Date Formats:** Enforce `YYYY-MM-DD` format at the point of data entry to avoid errors.
        """)

    with rec_col2:
        st.success("🚀 **Business Recommendations**")
        if profit_col and (df[profit_col] < 0).any():
             loss_count = (df[profit_col] < 0).sum()
             st.write(f"⚠️ **Alert:** Found {loss_count} transactions with negative profit.")
             st.markdown("* **Action:** Audit discounts. Are sales teams giving too much discount to close deals?")
        if date_cols:
             st.markdown("* **Action:** Use the Time Series chart to plan inventory 2 months before the peak season starts.")

    # --- STEP 4: EXPORT ---
    st.markdown("---")
    st.header("⬇️ Final Step: Power BI Export")
    st.write("This file is now **Optimized for Power BI**. Nulls are handled, dates are formatted, and headers are clean.")
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Optimized CSV for Power BI",
        data=csv,
        file_name="PowerBI_Ready_Data.csv",
        mime="text/csv"
    )
