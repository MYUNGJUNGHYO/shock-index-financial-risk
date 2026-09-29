import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="SDI-SVJ Jump Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 SDI-SVJ Jump Analyzer")

st.write(
    "Shock Index 기반 이상변동 탐지와 "
    "종목 간 동조·역방향 Jump 관계를 분석합니다."
)


# =========================================================
# KOSPI 주요 50종목
# MVP용 고정 목록
# =========================================================
stocks = {

    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "삼성바이오로직스": "207940.KS",
    "현대차": "005380.KS",

    "기아": "000270.KS",
    "셀트리온": "068270.KS",
    "KB금융": "105560.KS",
    "NAVER": "035420.KS",
    "신한지주": "055550.KS",

    "POSCO홀딩스": "005490.KS",
    "삼성물산": "028260.KS",
    "현대모비스": "012330.KS",
    "카카오": "035720.KS",
    "LG화학": "051910.KS",

    "삼성SDI": "006400.KS",
    "하나금융지주": "086790.KS",
    "메리츠금융지주": "138040.KS",
    "삼성생명": "032830.KS",
    "한화에어로스페이스": "012450.KS",

    "두산에너빌리티": "034020.KS",
    "HD현대중공업": "329180.KS",
    "HD한국조선해양": "009540.KS",
    "삼성중공업": "010140.KS",
    "한국전력": "015760.KS",

    "KT&G": "033780.KS",
    "SK텔레콤": "017670.KS",
    "KT": "030200.KS",
    "LG전자": "066570.KS",
    "LG": "003550.KS",

    "SK": "034730.KS",
    "SK이노베이션": "096770.KS",
    "S-Oil": "010950.KS",
    "대한항공": "003490.KS",
    "아모레퍼시픽": "090430.KS",

    "삼성화재": "000810.KS",
    "우리금융지주": "316140.KS",
    "기업은행": "024110.KS",
    "카카오뱅크": "323410.KS",
    "삼성전기": "009150.KS",

    "한미반도체": "042700.KS",
    "현대로템": "064350.KS",
    "포스코퓨처엠": "003670.KS",
    "LG이노텍": "011070.KS",
    "SK스퀘어": "402340.KS",

    "하이브": "352820.KS",
    "크래프톤": "259960.KS",
    "넷마블": "251270.KS",
    "엔씨소프트": "036570.KS",
    "코웨이": "021240.KS"
}


# =========================================================
# 분석 함수
# =========================================================
@st.cache_data(ttl=3600)
def analyze_stock(
    ticker,
    period,
    window,
    sdi_threshold,
    detection_mode
):

    df = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()

    # 로그 수익률
    df["Return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    # 일반 수익률
    df["Return_pct"] = (
        df["Close"].pct_change() * 100
    )

    # Rolling volatility
    df["Volatility"] = (
        df["Return"]
        .rolling(window=window)
        .std()
        .shift(1)
    )

    # Shock Index
    df["Shock_Index"] = (
        df["Return"].abs()
        / df["Volatility"]
    )

    # Visual 기준
    abs_return = df["Return_pct"].abs()

    visual_weak = abs_return >= 4.0
    visual_medium = abs_return >= 6.0
    visual_strong = abs_return >= 8.0


    if detection_mode == "SDI만 사용":

        weak = (
            df["Shock_Index"]
            >= sdi_threshold
        )

        medium = (
            df["Shock_Index"]
            >= max(3.0, sdi_threshold)
        )

        strong = (
            df["Shock_Index"]
            >= max(3.7, sdi_threshold)
        )

    else:

        weak = (
            (
                df["Shock_Index"]
                >= sdi_threshold
            )
            | visual_weak
        )

        medium = (
            (
                df["Shock_Index"]
                >= max(3.0, sdi_threshold)
            )
            | visual_medium
        )

        strong = (
            (
                df["Shock_Index"]
                >= max(3.7, sdi_threshold)
            )
            | visual_strong
        )


    df["Jump_Level"] = "Normal"

    df.loc[
        weak,
        "Jump_Level"
    ] = "약한 Jump"

    df.loc[
        medium,
        "Jump_Level"
    ] = "중간 Jump"

    df.loc[
        strong,
        "Jump_Level"
    ] = "강한 Jump"


    df["Direction"] = np.where(
        df["Return_pct"] >= 0,
        "상승",
        "하락"
    )


    df["SDI_Detected"] = (
        df["Shock_Index"]
        >= sdi_threshold
    )

    df["Visual_Detected"] = (
        df["Return_pct"].abs()
        >= 4.0
    )


    df["Detection_Reason"] = "-"

    df.loc[
        df["SDI_Detected"]
        & df["Visual_Detected"],
        "Detection_Reason"
    ] = "SDI + Visual"

    df.loc[
        df["SDI_Detected"]
        & ~df["Visual_Detected"],
        "Detection_Reason"
    ] = "SDI"

    df.loc[
        ~df["SDI_Detected"]
        & df["Visual_Detected"],
        "Detection_Reason"
    ] = "Visual"

    return df


# =========================================================
# 기본 선택
# =========================================================
c1, c2 = st.columns(2)

with c1:

    stock_name = st.selectbox(
        "📌 기준 종목",
        list(stocks.keys())
    )

with c2:

    period = st.selectbox(
        "📅 분석 기간",
        ["1y", "2y", "5y", "10y"],
        index=2
    )


ticker = stocks[stock_name]


# =========================================================
# 탐지 설정
# =========================================================
st.subheader("⚙️ 탐지 설정")

s1, s2, s3, s4 = st.columns(4)

with s1:

    WINDOW = st.slider(
        "Rolling Window",
        5,
        60,
        20,
        5
    )

with s2:

    SDI_THRESHOLD = st.slider(
        "SDI 최소 기준",
        1.5,
        4.5,
        2.5,
        0.1
    )

with s3:

    detection_mode = st.selectbox(
        "탐지 방식",
        [
            "SDI + Visual 통합",
            "SDI만 사용"
        ]
    )

with s4:

    lag_days = st.selectbox(
        "관계 분석 시차",
        [0, 1, 2, 3],
        format_func=lambda x:
            "같은 날" if x == 0
            else f"{x}거래일 후"
    )


show_weak = st.checkbox(
    "약한 Jump도 메인 차트에 표시",
    value=False
)


# =========================================================
# 기준 종목 분석
# =========================================================
data = analyze_stock(
    ticker,
    period,
    WINDOW,
    SDI_THRESHOLD,
    detection_mode
)


if data is None:

    st.error(
        "주가 데이터를 불러오지 못했습니다."
    )

    st.stop()


# =========================================================
# KPI
# =========================================================
current_price = float(
    data["Close"].iloc[-1]
)

previous_price = float(
    data["Close"].iloc[-2]
)

change_rate = (
    (
        current_price
        - previous_price
    )
    / previous_price
    * 100
)

latest_shock = data[
    "Shock_Index"
].iloc[-1]

if pd.isna(latest_shock):
    latest_shock = 0


recent_jump = int(
    (
        data.tail(30)[
            "Jump_Level"
        ]
        != "Normal"
    ).sum()
)


st.subheader(
    f"📊 {stock_name}"
)

k1, k2, k3, k4 = st.columns(4)


k1.metric(
    "현재 주가",
    f"{current_price:,.0f}원"
)

k2.metric(
    "전일 대비",
    f"{change_rate:.2f}%"
)

k3.metric(
    "현재 Shock Index",
    f"{latest_shock:.2f}"
)

k4.metric(
    "최근 30거래일 Jump",
    f"{recent_jump}회"
)


# =========================================================
# 메인 차트
# =========================================================
st.subheader(
    "📈 주가 및 주요 Jump"
)

fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Close"],
        mode="lines",
        name=stock_name,
        line=dict(
            width=2
        )
    )
)


def add_jump(
    fig,
    df,
    level,
    direction,
    color,
    size,
    symbol,
    name
):

    temp = df[
        (df["Jump_Level"] == level)
        & (df["Direction"] == direction)
    ]

    if temp.empty:
        return


    customdata = np.stack(
        [
            temp["Shock_Index"].fillna(0),
            temp["Return_pct"].fillna(0),
            temp["Detection_Reason"]
        ],
        axis=-1
    )


    fig.add_trace(
        go.Scatter(
            x=temp.index,
            y=temp["Close"],
            mode="markers",
            name=name,

            marker=dict(
                color=color,
                size=size,
                symbol=symbol
            ),

            customdata=customdata,

            hovertemplate=(
                "날짜: %{x}<br>"
                "주가: %{y:,.0f}원<br>"
                "Shock Index: %{customdata[0]:.2f}<br>"
                "등락률: %{customdata[1]:.2f}%<br>"
                "탐지: %{customdata[2]}"
                "<extra></extra>"
            )
        )
    )


# 강한
add_jump(
    fig,
    data,
    "강한 Jump",
    "상승",
    "red",
    14,
    "triangle-up",
    "강한 상승"
)

add_jump(
    fig,
    data,
    "강한 Jump",
    "하락",
    "blue",
    14,
    "triangle-down",
    "강한 하락"
)

# 중간
add_jump(
    fig,
    data,
    "중간 Jump",
    "상승",
    "orange",
    10,
    "triangle-up",
    "중간 상승"
)

add_jump(
    fig,
    data,
    "중간 Jump",
    "하락",
    "deepskyblue",
    10,
    "triangle-down",
    "중간 하락"
)

# 약한
if show_weak:

    add_jump(
        fig,
        data,
        "약한 Jump",
        "상승",
        "gold",
        7,
        "triangle-up",
        "약한 상승"
    )

    add_jump(
        fig,
        data,
        "약한 Jump",
        "하락",
        "lightblue",
        7,
        "triangle-down",
        "약한 하락"
    )


fig.update_layout(
    height=620,
    xaxis_title="날짜",
    yaxis_title="주가",
    hovermode="closest"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 월별 Jump + Shock Index
# =========================================================
monthly = data.copy()

monthly["Month"] = (
    monthly.index
    .to_period("M")
    .astype(str)
)


monthly_jump = (
    monthly[
        monthly["Jump_Level"]
        != "Normal"
    ]
    .groupby("Month")
    .size()
)


left, right = st.columns(2)


with left:

    st.subheader(
        "📅 월별 Jump 발생"
    )

    monthly_fig = go.Figure()

    monthly_fig.add_trace(
        go.Bar(
            x=monthly_jump.index,
            y=monthly_jump.values
        )
    )

    monthly_fig.update_layout(
        height=380,
        xaxis_title="월",
        yaxis_title="Jump 횟수"
    )

    st.plotly_chart(
        monthly_fig,
        use_container_width=True
    )


with right:

    st.subheader(
        "⚡ Shock Index"
    )

    shock_fig = go.Figure()

    shock_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Shock_Index"],
            mode="lines"
        )
    )

    shock_fig.add_hline(
        y=3.0,
        line_dash="dot",
        annotation_text="중간 3.0"
    )

    shock_fig.add_hline(
        y=3.7,
        line_dash="dash",
        annotation_text="강한 3.7"
    )

    shock_fig.update_layout(
        height=380,
        xaxis_title="날짜",
        yaxis_title="Shock Index"
    )

    st.plotly_chart(
        shock_fig,
        use_container_width=True
    )


# =========================================================
# 관계 분석
# =========================================================
st.divider()

st.header(
    "🔗 종목 간 Jump 관계 분석"
)


lag_text = (
    "같은 날"
    if lag_days == 0
    else f"{lag_days}거래일 후"
)


st.write(
    f"""
현재 설정: **{stock_name} Jump 발생 → {lag_text} 다른 종목 반응**

중간·강한 Jump만 관계 분석에 사용합니다.
"""
)


# =========================================================
# 기준 Jump
# =========================================================
major = data[
    data["Jump_Level"]
    .isin(
        [
            "중간 Jump",
            "강한 Jump"
        ]
    )
].copy()


base_up = major[
    major["Direction"]
    == "상승"
]


base_down = major[
    major["Direction"]
    == "하락"
]


base_up_count = len(
    base_up
)

base_down_count = len(
    base_down
)


m1, m2 = st.columns(2)


m1.metric(
    f"{stock_name} 주요 상승 Jump",
    f"{base_up_count}회"
)

m2.metric(
    f"{stock_name} 주요 하락 Jump",
    f"{base_down_count}회"
)


# =========================================================
# 기준 날짜 -> n 거래일 후 날짜 변환 함수
# =========================================================
def get_lagged_dates(
    base_dates,
    target_index,
    lag
):

    target_index = pd.DatetimeIndex(
        target_index
    )

    result_dates = []

    for date in base_dates:

        # 해당 날짜 또는 이후 첫 거래일 위치
        pos = target_index.searchsorted(
            date
        )

        target_pos = (
            pos + lag
        )

        if target_pos < len(
            target_index
        ):

            result_dates.append(
                target_index[
                    target_pos
                ].normalize()
            )

    return set(
        result_dates
    )


# =========================================================
# 다른 49개 종목 분석
# =========================================================
relationship_results = []
comparison_data = {}


progress = st.progress(0)

status = st.empty()


other_stocks = [
    (name, ticker)
    for name, ticker
    in stocks.items()
    if name != stock_name
]


for i, (
    other_name,
    other_ticker
) in enumerate(
    other_stocks
):


    status.write(
        f"분석 중: {other_name}"
    )


    other_df = analyze_stock(
        other_ticker,
        period,
        WINDOW,
        SDI_THRESHOLD,
        detection_mode
    )


    progress.progress(
        (i + 1)
        / len(other_stocks)
    )


    if other_df is None:
        continue


    comparison_data[
        other_name
    ] = other_df


    other_df = other_df.copy()

    other_df["Date_Normal"] = (
        other_df.index.normalize()
    )


    # -----------------------------------------
    # 상승 Jump 날짜 + lag
    # -----------------------------------------
    up_target_dates = (
        get_lagged_dates(
            base_up.index,
            other_df.index,
            lag_days
        )
    )


    # -----------------------------------------
    # 하락 Jump 날짜 + lag
    # -----------------------------------------
    down_target_dates = (
        get_lagged_dates(
            base_down.index,
            other_df.index,
            lag_days
        )
    )


    # 기준 상승 이후
    on_up = other_df[
        other_df[
            "Date_Normal"
        ].isin(
            up_target_dates
        )
    ]


    # 기준 하락 이후
    on_down = other_df[
        other_df[
            "Date_Normal"
        ].isin(
            down_target_dates
        )
    ]


    # -----------------------------------------
    # 삼성 ↑ → 상대 ↑
    # -----------------------------------------
    same_up = on_up[
        (
            on_up["Jump_Level"]
            .isin(
                [
                    "중간 Jump",
                    "강한 Jump"
                ]
            )
        )
        &
        (
            on_up["Direction"]
            == "상승"
        )
    ]


    # -----------------------------------------
    # 삼성 ↑ → 상대 ↓
    # -----------------------------------------
    up_to_down = on_up[
        (
            on_up["Jump_Level"]
            .isin(
                [
                    "중간 Jump",
                    "강한 Jump"
                ]
            )
        )
        &
        (
            on_up["Direction"]
            == "하락"
        )
    ]


    # -----------------------------------------
    # 삼성 ↓ → 상대 ↓
    # -----------------------------------------
    same_down = on_down[
        (
            on_down["Jump_Level"]
            .isin(
                [
                    "중간 Jump",
                    "강한 Jump"
                ]
            )
        )
        &
        (
            on_down["Direction"]
            == "하락"
        )
    ]


    # -----------------------------------------
    # 삼성 ↓ → 상대 ↑
    # -----------------------------------------
    down_to_up = on_down[
        (
            on_down["Jump_Level"]
            .isin(
                [
                    "중간 Jump",
                    "강한 Jump"
                ]
            )
        )
        &
        (
            on_down["Direction"]
            == "상승"
        )
    ]


    same_up_count = len(
        same_up
    )

    same_down_count = len(
        same_down
    )

    up_to_down_count = len(
        up_to_down
    )

    down_to_up_count = len(
        down_to_up
    )


    same_up_rate = (
        same_up_count
        / base_up_count
        * 100
        if base_up_count > 0
        else 0
    )


    same_down_rate = (
        same_down_count
        / base_down_count
        * 100
        if base_down_count > 0
        else 0
    )


    up_to_down_rate = (
        up_to_down_count
        / base_up_count
        * 100
        if base_up_count > 0
        else 0
    )


    down_to_up_rate = (
        down_to_up_count
        / base_down_count
        * 100
        if base_down_count > 0
        else 0
    )


    relationship_results.append(
        {

            "종목":
                other_name,

            "같이급등횟수":
                same_up_count,

            "같이급등률":
                same_up_rate,

            "같이급락횟수":
                same_down_count,

            "같이급락률":
                same_down_rate,

            "급등후급락횟수":
                up_to_down_count,

            "급등후급락률":
                up_to_down_rate,

            "급락후급등횟수":
                down_to_up_count,

            "급락후급등률":
                down_to_up_rate
        }
    )


progress.empty()
status.empty()


relation_df = pd.DataFrame(
    relationship_results
)


# =========================================================
# TOP 5
# =========================================================
if not relation_df.empty:


    top_same_up = (
        relation_df
        .sort_values(
            "같이급등률",
            ascending=False
        )
        .head(5)
    )


    top_same_down = (
        relation_df
        .sort_values(
            "같이급락률",
            ascending=False
        )
        .head(5)
    )


    top_up_down = (
        relation_df
        .sort_values(
            "급등후급락률",
            ascending=False
        )
        .head(5)
    )


    top_down_up = (
        relation_df
        .sort_values(
            "급락후급등률",
            ascending=False
        )
        .head(5)
    )


    st.subheader(
        f"🧭 {lag_text} Jump 관계 TOP 5"
    )


    row1_left, row1_right = (
        st.columns(2)
    )


    # =====================================================
    # 같이 급등
    # =====================================================
    with row1_left:

        st.markdown(
            f"### 🟢 {stock_name} 급등 → 같이 급등"
        )

        f1 = go.Figure()

        f1.add_trace(
            go.Bar(
                x=top_same_up[
                    "같이급등률"
                ],
                y=top_same_up[
                    "종목"
                ],
                orientation="h",

                text=(
                    top_same_up[
                        "같이급등률"
                    ]
                    .round(1)
                    .astype(str)
                    + "%"
                )
            )
        )

        f1.update_layout(
            height=340,
            xaxis_title="비율 (%)"
        )

        st.plotly_chart(
            f1,
            use_container_width=True
        )


    # =====================================================
    # 같이 급락
    # =====================================================
    with row1_right:

        st.markdown(
            f"### 🔵 {stock_name} 급락 → 같이 급락"
        )

        f2 = go.Figure()

        f2.add_trace(
            go.Bar(
                x=top_same_down[
                    "같이급락률"
                ],
                y=top_same_down[
                    "종목"
                ],
                orientation="h",

                text=(
                    top_same_down[
                        "같이급락률"
                    ]
                    .round(1)
                    .astype(str)
                    + "%"
                )
            )
        )

        f2.update_layout(
            height=340,
            xaxis_title="비율 (%)"
        )

        st.plotly_chart(
            f2,
            use_container_width=True
        )


    row2_left, row2_right = (
        st.columns(2)
    )


    # =====================================================
    # 급등 후 급락
    # =====================================================
    with row2_left:

        st.markdown(
            f"### 🔻 {stock_name} 급등 → 반대로 급락"
        )

        f3 = go.Figure()

        f3.add_trace(
            go.Bar(
                x=top_up_down[
                    "급등후급락률"
                ],
                y=top_up_down[
                    "종목"
                ],
                orientation="h",

                text=(
                    top_up_down[
                        "급등후급락률"
                    ]
                    .round(1)
                    .astype(str)
                    + "%"
                )
            )
        )

        f3.update_layout(
            height=340,
            xaxis_title="비율 (%)"
        )

        st.plotly_chart(
            f3,
            use_container_width=True
        )


    # =====================================================
    # 급락 후 급등
    # =====================================================
    with row2_right:

        st.markdown(
            f"### 🔺 {stock_name} 급락 → 반대로 급등"
        )

        f4 = go.Figure()

        f4.add_trace(
            go.Bar(
                x=top_down_up[
                    "급락후급등률"
                ],
                y=top_down_up[
                    "종목"
                ],
                orientation="h",

                text=(
                    top_down_up[
                        "급락후급등률"
                    ]
                    .round(1)
                    .astype(str)
                    + "%"
                )
            )
        )

        f4.update_layout(
            height=340,
            xaxis_title="비율 (%)"
        )

        st.plotly_chart(
            f4,
            use_container_width=True
        )


# =========================================================
# 두 종목 상세 비교
# =========================================================
st.divider()

st.header(
    "📊 두 종목 상세 비교"
)


compare_options = [
    name
    for name
    in stocks.keys()
    if name != stock_name
]


compare_name = st.selectbox(
    "비교할 종목",
    compare_options
)


compare_data = comparison_data.get(
    compare_name
)


if compare_data is not None:


    compare_df = pd.DataFrame(
        {
            stock_name:
                data["Close"],

            compare_name:
                compare_data["Close"]
        }
    ).dropna()


    normalized = (
        compare_df
        / compare_df.iloc[0]
        * 100
    )


    graph_left, graph_right = (
        st.columns(2)
    )


    # =====================================================
    # 정규화 가격
    # =====================================================
    with graph_left:

        st.subheader(
            "📈 정규화 주가"
        )

        pfig = go.Figure()

        pfig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[
                    stock_name
                ],
                mode="lines",
                name=stock_name
            )
        )

        pfig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[
                    compare_name
                ],
                mode="lines",
                name=compare_name
            )
        )

        pfig.update_layout(
            height=430,
            yaxis_title="시작값 = 100",
            hovermode="x unified"
        )

        st.plotly_chart(
            pfig,
            use_container_width=True
        )


    # =====================================================
    # 등락률
    # =====================================================
    with graph_right:

        st.subheader(
            "📊 일일 등락률"
        )


        r_df = pd.DataFrame(
            {
                stock_name:
                    data[
                        "Return_pct"
                    ],

                compare_name:
                    compare_data[
                        "Return_pct"
                    ]
            }
        ).dropna()


        rfig = go.Figure()

        rfig.add_trace(
            go.Scatter(
                x=r_df.index,
                y=r_df[
                    stock_name
                ],
                mode="lines",
                name=stock_name
            )
        )

        rfig.add_trace(
            go.Scatter(
                x=r_df.index,
                y=r_df[
                    compare_name
                ],
                mode="lines",
                name=compare_name
            )
        )

        rfig.update_layout(
            height=430,
            yaxis_title="등락률 (%)",
            hovermode="x unified"
        )

        st.plotly_chart(
            rfig,
            use_container_width=True
        )


    # =====================================================
    # 관계 KPI
    # =====================================================
    selected = relation_df[
        relation_df[
            "종목"
        ]
        == compare_name
    ]


    if not selected.empty:

        result = selected.iloc[0]

        st.subheader(
            f"🔍 {stock_name} → {compare_name} ({lag_text})"
        )


        q1, q2, q3, q4 = (
            st.columns(4)
        )


        q1.metric(
            "같이 급등",
            (
                f"{int(result['같이급등횟수'])}회 "
                f"({result['같이급등률']:.1f}%)"
            )
        )


        q2.metric(
            "같이 급락",
            (
                f"{int(result['같이급락횟수'])}회 "
                f"({result['같이급락률']:.1f}%)"
            )
        )


        q3.metric(
            "급등 → 급락",
            (
                f"{int(result['급등후급락횟수'])}회 "
                f"({result['급등후급락률']:.1f}%)"
            )
        )


        q4.metric(
            "급락 → 급등",
            (
                f"{int(result['급락후급등횟수'])}회 "
                f"({result['급락후급등률']:.1f}%)"
            )
        )


# =========================================================
# 설명
# =========================================================
st.info(
    f"""
현재 관계 분석 시차 : {lag_text}

예를 들어 1거래일 후를 선택하면

삼성전자에서 월요일 상승 Jump 발생
→ SK하이닉스의 화요일 Jump 방향 확인

2거래일 후라면
→ 수요일 반응 확인

3거래일 후라면
→ 목요일 반응 확인


중간 또는 강한 Jump만 관계 분석에 사용합니다.

이 수치는 통계적 동시·후행 발생 관계이며,
인과관계를 의미하지 않습니다.
"""
)