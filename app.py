import streamlit as st
import sqlite3
import pandas as pd
import json

def load_data():
    """Load data from SQLite database"""
    conn = sqlite3.connect('output/pokemon_cards.db')
    df = pd.read_sql_query("SELECT * FROM pokemon_cards", conn)
    conn.close()
    return df

def main():
    st.set_page_config(page_title="Pokemon Card Database", layout="wide")
    st.title("Pokemon Card Database")

    # Load data
    df = load_data()

    # Basic stats
    st.header("Collection Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Cards", len(df))
    with col2:
        st.metric("Total URLs", df['search_urls'].apply(lambda x: len(json.loads(x))).sum())
    with col3:
        avg_low = df['price_float_low'].mean()
        st.metric("Average Min Price", f"${avg_low:.2f}")
    with col4:
        avg_mid = df['price_float'].mean()
        st.metric("Average Price", f"${avg_mid:.2f}")
    with col5:
        avg_high = df['price_float_high'].mean()
        st.metric("Average Max Price", f"${avg_high:.2f}")

    # Price Range Analysis
    st.header("Price Distribution")
    price_range = st.slider(
        "Filter by Price Range",
        float(df['price_float_low'].min()),
        float(df['price_float_high'].max()),
        (float(df['price_float_low'].min()), float(df['price_float_high'].max()))
    )

    # Main data view
    st.header("Card Details")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        name_filter = st.text_input("Filter by Name")
    with col2:
        type_filter = st.selectbox("Filter by Type", ["All"] + sorted(df['type'].unique().tolist()))
    with col3:
        sort_by = st.selectbox("Sort by", ["Name", "Low Price", "Mid Price", "High Price"])

    # Apply filters
    filtered_df = df.copy()
    if name_filter:
        filtered_df = filtered_df[filtered_df['name'].str.contains(name_filter, case=False, na=False)]
    if type_filter != "All":
        filtered_df = filtered_df[filtered_df['type'] == type_filter]
    
    # Apply price range filter
    filtered_df = filtered_df[
        (filtered_df['price_float_low'] >= price_range[0]) &
        (filtered_df['price_float_high'] <= price_range[1])
    ]

    # Sort
    if sort_by == "Low Price":
        filtered_df = filtered_df.sort_values('price_float_low', ascending=False)
    elif sort_by == "Mid Price":
        filtered_df = filtered_df.sort_values('price_float', ascending=False)
    elif sort_by == "High Price":
        filtered_df = filtered_df.sort_values('price_float_high', ascending=False)
    else:
        filtered_df = filtered_df.sort_values('name')

    # Display data
    st.dataframe(
        filtered_df[[
            'name', 'type', 'rarity', 'hp', 'card_number', 
            'estimated_price', 'price_float_low', 'price_float', 'price_float_high'
        ]].style.format({
            'price_float_low': '${:.2f}',
            'price_float': '${:.2f}',
            'price_float_high': '${:.2f}'
        })
    )

    # URL viewer
    st.header("Reference URLs")
    selected_card = st.selectbox("Select a card to view its reference URLs", df['name'].unique())
    
    if selected_card:
        card_data = df[df['name'] == selected_card].iloc[0]
        st.subheader(f"URLs for {selected_card}")
        st.text(f"Estimated Price: {card_data['estimated_price']}")
        st.text(f"Price Range: ${card_data['price_float_low']:.2f} - ${card_data['price_float']:.2f} - ${card_data['price_float_high']:.2f}")
        
        urls = json.loads(card_data['search_urls'])
        for url in urls:
            st.markdown(f"- [{url}]({url})")

if __name__ == "__main__":
    main() 