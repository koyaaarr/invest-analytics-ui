import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

if "window_size" not in st.session_state:
    st.session_state.window_size = 365


def calc_portfolio(x, num_holds):
    pf = 0
    for k, v in num_holds.items():
        pf += x[f"Close_{k}"] * v
    return pf


def calc_stock(num_holds, stocks, ratio, portfolio):
    # calc portfolio value
    stocks["Close_Portfolio"] = stocks.apply(
        lambda x: calc_portfolio(x, num_holds), axis=1
    )
    # convert to int
    for k in num_holds.keys():
        stocks[f"Close_{k}"] = np.floor(
            pd.to_numeric(stocks[f"Close_{k}"], errors="coerce")
        ).astype("Int64")
    stocks["Close_Portfolio"] = np.floor(
        pd.to_numeric(stocks["Close_Portfolio"], errors="coerce")
    ).astype("Int64")
    # recent value ratio
    recent_valid_index = (
        stocks.dropna(subset=["Close_Portfolio"]).tail(1).index.values[0]
    )
    # recent value percent
    recent_values = []
    for k in portfolio["ticker"].keys():
        recent = (
            stocks.loc[recent_valid_index, f"Close_{k}"]
            * num_holds[k]
            / stocks.loc[recent_valid_index, "Close_Portfolio"]
            * 100
        )
        recent = round(recent, 2)
        recent_values.append(recent)
    ratio = pd.DataFrame(
        data={"ticker": portfolio["ticker"].keys(), "ratio_percent": recent_values}
    )
    ratio["type"] = ratio.ticker.apply(lambda x: portfolio["ticker"][x]["type"])
    ratio["detail"] = ratio.ticker.apply(lambda x: portfolio["ticker"][x]["detail"])
    ratio["sector"] = ratio.ticker.apply(lambda x: portfolio["ticker"][x]["sector"])
    ratio["num_holds"] = ratio.ticker.apply(lambda x: num_holds[x])

    # calc sharpe ratio
    sharpe = stocks.loc[:, ["Date", "Close_Portfolio"]]
    sharpe = sharpe.dropna(subset=["Close_Portfolio"])
    sharpe["lag_1d"] = sharpe["Close_Portfolio"].shift(1)
    sharpe["rate_change"] = np.log(sharpe.Close_Portfolio / sharpe.lag_1d)
    sharpe["one_year_mean"] = sharpe["rate_change"].rolling(252).mean()
    sharpe["one_year_std"] = sharpe["rate_change"].rolling(252).std()
    sharpe["sharpe_ratio"] = sharpe.one_year_mean / sharpe.one_year_std
    sharpe["sharpe_ratio_annual"] = sharpe["sharpe_ratio"] * 252**0.5
    sharpe = sharpe.dropna(subset=["sharpe_ratio_annual"])
    sharpe = sharpe.loc[:, ["Date", "sharpe_ratio_annual"]]

    return sharpe, stocks, ratio


def layout_input(sharpe, stocks, ratio, portfolio):
    num_holds = {}
    for t in ratio.ticker:
        num_holds[t] = st.number_input(t, 0, key=t)
    num_holds = {k: float(v) for k, v in num_holds.items()}
    if st.button("Calculate"):
        sharpe, stocks, ratio = calc_stock(num_holds, stocks, ratio, portfolio)
    return sharpe, stocks, ratio


@st.cache
def read_stock_data_from_local():
    stocks = pd.read_pickle("data/stocks.pkl")
    ratio = pd.read_pickle("data/ratio.pkl")
    sharpe = pd.read_pickle("data/sharpe.pkl")
    return sharpe, stocks, ratio


def plot_stock_data(df: pd.DataFrame, window: int, x: str, y: str, title: str) -> None:
    """plot line chart"""
    # drop nan values
    df = df.dropna(subset=y)
    df = df.tail(window)
    fig = px.line(df, x=x, y=y)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)


def layout_plots(sharpe, stocks, ratio) -> None:
    """manipulate plots layout"""
    col1, col2 = st.columns(2)

    # plot portfolio values by time series
    # with col1:
    col0, col1, col2, col3, col4 = st.columns([6, 1, 1, 1, 1])
    with col0:
        st.write("Portfolio Overall Performance")
    with col1:
        if st.button("3Year", key="portfolio"):
            st.session_state.window_size = 1080
    with col2:
        if st.button("Year", key="portfolio"):
            st.session_state.window_size = 360
    with col3:
        if st.button("Quarter", key="portfolio"):
            st.session_state.window_size = 90
    with col4:
        if st.button("Month", key="portfolio"):
            st.session_state.window_size = 30
    plot_stock_data(
        df=stocks,
        window=st.session_state.window_size,
        x="Date",
        y="Close_Portfolio",
        title="Portfolio Transition",
    )

    # plot sharpe ratio
    st.write("Portfolio Annual Sharpe Ratio (Baseline = 1)")
    fig = px.line(sharpe, x="Date", y="sharpe_ratio_annual")
    fig.add_hline(1, line_color="red")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

    # plot portfolio share by pie chart
    fig = px.sunburst(
        ratio,
        path=["type", "ticker"],
        values="ratio_percent",
        title="Portfolio Recent Value Ratio",
    )
    st.plotly_chart(fig, use_container_width=True)

    # table for each stock stat
    st.write("Portfolio Detail")
    st.table(
        ratio.loc[
            :, ["ticker", "detail", "type", "sector", "num_holds", "ratio_percent"]
        ]
    )

    for ticker, detail in zip(ratio.ticker.values, ratio.detail.values):
        col0, col1, col2, col3, col4 = st.columns([6, 1, 1, 1, 1])
        with col0:
            st.write(f"{ticker}: {detail}")
        with col1:
            if st.button("3Year", key=f"{ticker}"):
                st.session_state.window_size = 1080
        with col2:
            if st.button("Year", key=f"{ticker}"):
                st.session_state.window_size = 360
        with col3:
            if st.button("Quarter", key=f"{ticker}"):
                st.session_state.window_size = 90
        with col4:
            if st.button("Month", key=f"{ticker}"):
                st.session_state.window_size = 30
        plot_stock_data(
            df=stocks,
            window=st.session_state.window_size,
            x="Date",
            y=f"Close_{ticker}",
            title=f"{ticker}: {detail}",
        )


if __name__ == "__main__":

    # general layout settings
    st.set_page_config(layout="wide")
    st.title("ETF Portfolio Simulator")

    # load sharpe, stocks, and ratio pickles
    sharpe, stocks, ratio = read_stock_data_from_local()

    # load portfolio settings
    with open("portfolio.yaml", "rb") as file:
        portfolio = yaml.safe_load(file)

    # layout sidebar
    with st.sidebar:
        st.write("Input Portfolio To Simulate")
        sharpe, stocks, ratio = layout_input(sharpe, stocks, ratio, portfolio)

    # layout plots
    layout_plots(sharpe, stocks, ratio)
