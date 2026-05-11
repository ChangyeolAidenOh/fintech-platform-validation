"""
Fintech Platform Validation — Streamlit Dashboard v4
Customer-centric positioning + Target Users + Platform enhancement framing
"""
from customer_report_tab import tab_customer_report

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

st.set_page_config(
    page_title="Fintech Platform Validation",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data/exports"
FIG_DIR = "figures"


@st.cache_data
def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_figure(filename):
    path = os.path.join(FIG_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


def business_takeaway(text):
    st.markdown(f"""
    <div style="background-color: #f0f2f6; border-left: 4px solid #4682B4;
                padding: 12px 16px; margin-top: 20px; border-radius: 4px;">
        <strong>Business Takeaway:</strong> {text}
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# Sidebar — v4: customer-centric About
# ================================================================
def render_sidebar():
    st.sidebar.title("Fintech Platform Validation")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "금융 빅데이터 플랫폼의 대고객 데이터 상품 고도화를 위해, "
        "공공 외부데이터의 증분가치와 활용 방향을 검증한 프로젝트입니다."
    )

    st.sidebar.markdown(
        '<small>본 프로젝트는 카드사 내부 데이터가 아닌 '
        '서울시 공공 상권 데이터를 사용했으며, '
        '상권×업종 단위에서 공공 외부데이터의 증분효과를 검증하는 데 목적이 있습니다.</small>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Data")
    st.sidebar.markdown(
        "- **Source:** 서울 열린데이터광장, ECOS\n"
        "- **Rows:** 2,906,472 (raw) → 570,089 (mart)\n"
        "- **Period:** 2019Q1 ~ 2025Q3 (27 quarters)\n"
        "- **Districts:** 1,603 | **Industries:** 63"
    )

    st.sidebar.markdown("### Methodology")
    st.sidebar.markdown(
        "- **Model:** XGBoost (+ LR Baseline)\n"
        "- **Split:** Time-based\n"
        "  - Train: 2019Q1 ~ 2024Q3\n"
        "  - Test: 2024Q4 ~ 2025Q3\n"
        "- **Target:** Next-quarter closure rate > threshold\n"
        "- **검증 실험:** 11개\n"
        "- **Interpretation:** SHAP + Feature Ablation"
    )

    st.sidebar.markdown("### Tech Stack")
    st.sidebar.markdown(
        "Python · PostgreSQL · XGBoost · SHAP · Streamlit"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "[GitHub](https://github.com/ChangyeolAidenOh/fintech-platform-validation)"
    )


# ================================================================
# Tab 1: Project Motivation — v4: customer-centric framing
# ================================================================
def tab_motivation():
    st.header("금융 빅데이터 플랫폼 고도화를 위한 외부데이터 증분가치 검증")

    # v4: Key Question + Finding at top
    st.markdown(
        "> **Key Question:** 소상공인 및 금융기관 고객이 상권 리스크를 판단할 때, "
        "공공 외부데이터는 추정매출, 점포 구조 변수 대비 실제로 추가 예측가치를 제공하는가?"
    )
    st.markdown(
        "> **Finding:** 본 공공데이터 환경에서는 증분효과가 제한적이었으며, "
        "외부데이터는 보강재보다 결제 이력이 부족한 경우의 부분적 대체재로 해석됩니다."
    )

    # Executive Summary KPI cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Dataset", "570K rows", help="2.9M raw → 570K mart")
    with k2:
        st.metric("Model AUROC", "0.7829", help="XGB Strict (추정매출 + 점포)")
    with k3:
        st.metric("External Lift", "≈ 0", help="ΔAUROC: -0.002 ~ +0.004")
    with k4:
        st.metric("검증 실험", "11개")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        # v4: customer-centric description
        st.markdown(
            """
            국내 카드사의 금융 빅데이터 플랫폼을 직접 사용하며,
            **소상공인 및 금융기관 고객이 상권 리스크를 판단할 때
            공공 외부데이터가 실제로 추가적인 예측 가치를 제공하는지** 검증했습니다.

            서울시 공공데이터 290만 행을 활용해 상권×업종 단위 폐업 리스크 모델을 구축하고,
            11개 검증 실험을 통해 외부데이터의 증분효과와 데이터 상품화 방향을 분석했습니다.
            """
        )

        # v4: shortened ABP reference
        st.info(
            "플랫폼 AI검색은 카드 매출 외에도 인허가, 사업자 상태, "
            "주변 시장 상황 등 외부 맥락 데이터가 필요할 수 있다고 안내했습니다. "
            "본 프로젝트는 그중 공공 수준에서 확보 가능한 환경 데이터의 증분효과를 검증했습니다."
        )

    with col2:
        st.markdown("#### 직접 사용 과정에서 보완 여지가 보인 지점")
        improvements = {
            "현황 분석 중심, 예측 기능 확장 여지": "고객에게 미래 리스크 등급 제공 가능",
            "폐업 리스크 변수의 직접 제공 제한": "리스크 스코어 상품화 가능",
            "장기 시계열 비교 기능 보완 여지": "추이 기반 조기 경보 기능 확장 가능",
            "입지 의사결정용 추천 기능 확장 가능": "예비 창업자 대상 입지 추천 상품화",
            "경쟁 상권 비교 기능 고도화 여지": "동종업종 밀도 기반 경쟁 분석 강화",
        }
        for title, desc in improvements.items():
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.markdown("---")

    # Target definition box
    st.markdown("#### 타겟 정의")
    st.markdown(
        """
        t분기 피처를 사용해 **t+1분기** high-risk 여부를 예측합니다.
        - **Primary target:** 다음 분기 폐업률이 해당 업종 중앙값의 1.5배 초과
        - **Normalized target:** 다음 분기 폐업률이 같은 업종, 분기 평균을 초과

        모든 피처는 예측 대상 분기 이전 시점의 정보만 사용하며,
        LEAD 윈도우 함수로 시점 분리를 보장합니다.
        """
    )

    st.markdown("---")

    # Target Users section
    st.markdown("#### 플랫폼 고객 유형별 활용 가능성")

    tu1, tu2, tu3 = st.columns(3)
    with tu1:
        st.markdown("**금융기관 / 제휴사**")
        st.markdown(
            "소상공인 리스크 조기 식별\n\n"
            "상권×업종 리스크 스코어와 업종별 차등 모니터링 기준 활용"
        )
    with tu2:
        st.markdown("**소상공인 / 예비 창업자**")
        st.markdown(
            "입지·업종 선택 시 리스크 판단\n\n"
            "단순 유동인구가 아니라 점포 회전율 및 동종업종 밀도 기반 판단"
        )
    with tu3:
        st.markdown("**데이터 구매 / 활용 고객**")
        st.markdown(
            "어떤 외부데이터가 가치 있는지 판단\n\n"
            "공공 외부데이터의 증분효과와 대체재 가능성 구분"
        )

    st.markdown("---")
    st.markdown("#### 핵심 질문 3개")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.metric("Q1", "외부 데이터 유효성")
    with q2:
        st.metric("Q2", "업종별 차이")
    with q3:
        st.metric("Q3", "예측력 개선 정도")


# ================================================================
# Tab 2: Ablation Study — unchanged from v3
# ================================================================
def tab_ablation():
    st.header("Ablation Study: 외부 데이터 증분효과 검증")

    st.markdown("#### 핵심: 추정매출 기반 대비 변화량(ΔAUROC)")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("추정매출+점포 (A)", "0.7830", help="외부 데이터 없이")
    with d2:
        st.metric("+ 전체 외부 (B)", "0.7811", delta="-0.0018", delta_color="inverse")
    with d3:
        st.metric("외부 제거 시", "+0.0018", help="역방향 ablation: 제거 시 개선")

    st.markdown("---")

    st.markdown("#### 추정매출 기반 대비 ΔAUROC (개별 외부 데이터)")

    delta_data = pd.DataFrame([
        {"External Data": "유동인구", "Delta": 0.0007},
        {"External Data": "상권변화지표", "Delta": 0.0005},
        {"External Data": "집객시설", "Delta": 0.0007},
        {"External Data": "CSI (거시경제)", "Delta": -0.0016},
        {"External Data": "상주인구", "Delta": 0.0010},
        {"External Data": "전체 외부", "Delta": -0.0018},
    ])

    fig_delta = px.bar(
        delta_data, x="External Data", y="Delta",
        color=delta_data["Delta"].apply(lambda x: "Positive" if x >= 0 else "Negative"),
        color_discrete_map={"Positive": "#2E8B57", "Negative": "#E8744F"},
        text=delta_data["Delta"].apply(lambda x: f"{x:+.4f}"),
    )
    fig_delta.add_hline(y=0, line_color="black", line_width=1)
    fig_delta.update_layout(
        height=400, showlegend=False,
        yaxis_title="ΔAUROC vs 추정매출 기반",
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    st.caption(
        "개별 외부 데이터의 증분효과는 ΔAUROC ±0.002 이내로, "
        "기존 추정매출 및 점포 변수 대비 추가 정보량이 제한적입니다."
    )

    st.markdown("---")

    st.markdown("#### AUROC 절대값 비교")

    ablation_data = pd.DataFrame([
        {"Model": "LR Baseline", "AUROC": 0.6325, "Type": "Baseline"},
        {"Model": "추정매출+점포 (A)", "AUROC": 0.7830, "Type": "Sales Proxy"},
        {"Model": "A + 전체 외부 (B)", "AUROC": 0.7811, "Type": "Full"},
        {"Model": "A + 유동인구", "AUROC": 0.7836, "Type": "Individual"},
        {"Model": "A + 상권변화", "AUROC": 0.7834, "Type": "Individual"},
        {"Model": "A + 집객시설", "AUROC": 0.7837, "Type": "Individual"},
        {"Model": "A + CSI", "AUROC": 0.7813, "Type": "Individual"},
        {"Model": "A + 상주인구", "AUROC": 0.7840, "Type": "Individual"},
    ])

    color_map = {"Baseline": "#CCCCCC", "Sales Proxy": "#4682B4", "Full": "#2E8B57", "Individual": "#E8744F"}

    fig = px.bar(
        ablation_data, x="Model", y="AUROC", color="Type",
        color_discrete_map=color_map,
        text=ablation_data["AUROC"].apply(lambda x: f"{x:.4f}"),
    )
    fig.add_hline(y=0.7830, line_dash="dash", line_color="#4682B4",
                  annotation_text="추정매출+점포 baseline: 0.7830")
    fig.update_layout(
        height=450, showlegend=True,
        yaxis_range=[0, 0.85],
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("#### Naive Baseline 비교")
    st.markdown("모델의 실질적 기여도를 파악하기 위한 naive baseline과의 비교")

    naive_data = pd.DataFrame([
        {"Model": "Naive: 직전 분기 폐업률", "AUROC": 0.6059, "Category": "Naive"},
        {"Model": "Naive: 업종 평균", "AUROC": 0.6049, "Category": "Naive"},
        {"Model": "Naive: store_count 단독", "AUROC": 0.7445, "Category": "Naive"},
        {"Model": "XGB: 추정매출+점포 (strict)", "AUROC": 0.7829, "Category": "XGBoost"},
    ])

    fig2 = px.bar(
        naive_data, x="Model", y="AUROC", color="Category",
        color_discrete_map={"Naive": "#CCCCCC", "XGBoost": "#4682B4"},
        text=naive_data["AUROC"].apply(lambda x: f"{x:.4f}"),
    )
    fig2.update_layout(height=400, yaxis_range=[0, 0.85], xaxis_tickangle=-20)
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "XGBoost는 최선의 naive baseline(store_count 단독, 0.7445) 대비 "
        "+0.038 AUROC 개선. \n P@100 = 0.99~1.00은 주로 store_count의 규모 효과에 기인합니다."
    )

    business_takeaway(
        "추정매출 및 점포 데이터가 확보된 환경에서는 리스크 스코어 개선을 위해 "
        "공공 외부데이터 확장보다 내부 결제 패턴 심화 분석이 더 효과적일 수 있습니다."
    )


# ================================================================
# Tab 3: Industry Risk — v4: 고회전·고변동
# ================================================================
def tab_industry():
    st.header("업종별 리스크 분석")

    industry_df = load_csv("industry_summary.csv")
    if industry_df is None:
        st.warning("data/exports/industry_summary.csv not found. Run: python run_export.py")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        top20 = industry_df.head(20)
        fig = px.bar(
            top20, x="avg_closure_rate", y="industry",
            orientation="h",
            text=top20["avg_closure_rate"].apply(lambda x: f"{x:.1%}"),
            color="avg_closure_rate",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(
            height=650,
            yaxis=dict(autorange="reversed"),
            xaxis_title="평균 분기 폐업률",
            yaxis_title="",
            title="Top 20 업종별 평균 폐업률",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 업종별 폐업률 차이")
        st.markdown("Kruskal-Wallis H=36,494, **p < 0.001**")
        st.markdown("업종 간 폐업률 차이는 통계적으로 매우 유의")

        st.markdown("---")

        # v4: customer-oriented threshold table
        st.markdown("#### 고객용 업종별 모니터링 기준 제안")
        threshold_df = pd.DataFrame({
            "업종군": ["고회전|고변동", "중간", "안정"],
            "대표 업종": ["치킨, 편의점, 패스트푸드", "커피, 분식, 중식", "의원, 치과, 가전"],
            "추천 경고 기준": ["업종 평균 대비 1.5배", "업종 평균 대비 1.3배", "절대 폐업 발생 여부"],
        })
        st.dataframe(threshold_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 76.3%의 상권-업종에서 분기 내 폐업이 0건")
    st.markdown(
        "폐업은 특정 고회전 업종에 구조적으로 집중됩니다. "
        "이들은 '위험 업종'이라기보다 점포 진입·퇴출 회전이 빠른 구조이며, "
        "고객에게 업종 특성을 반영한 차등 기준을 제공하는 것이 효과적입니다."
    )

    business_takeaway(
        "고객에게 동일 경고 기준이 아니라, 업종별 맥락을 반영한 차등 모니터링 기준 혹은 리포트를 제공한다면 "
        "불필요한 경고를 줄이고 실제 위험 신호의 식별 정확도를 높일 수 있습니다."
    )


# ================================================================
# Tab 4: SHAP & Redundancy — v4: external data role table
# ================================================================
def tab_shap():
    st.header("SHAP 분석 & Feature Redundancy")

    st.markdown("#### SHAP 기여도: 원본 vs 정규화 타겟")

    shap_data = pd.DataFrame({
        "Target": ["원본 타겟 (with store_count)", "원본 타겟 (with store_count)", "원본 타겟 (with store_count)",
                   "정규화 타겟 (no store_count)", "정규화 타겟 (no store_count)", "정규화 타겟 (no store_count)"],
        "Group": ["점포 구조 변수", "Sales Proxy", "Public External"] * 2,
        "SHAP %": [73, 19, 6, 36.6, 32.9, 30.5],
    })

    fig = px.bar(
        shap_data, x="SHAP %", y="Target", color="Group",
        orientation="h",
        color_discrete_map={
            "점포 구조 변수": "#4682B4",
            "Sales Proxy": "#87CEEB",
            "Public External": "#E8744F"
        },
        text=shap_data["SHAP %"].apply(lambda x: f"{x}%"),
        barmode="stack",
    )
    fig.update_layout(
        height=250,
        xaxis_title="SHAP Contribution %",
        yaxis_title="",
        legend_title="Feature Group",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("#### SHAP 30.5% vs AUROC +0.001 — Feature Redundancy")
    st.markdown(
        """
        정규화 타겟에서 외부 데이터의 SHAP 기여도가 **6% → 30.5%**로 5배 상승했습니다.
        그러나 실제 AUROC 개선은 +0.001 수준에 그쳤습니다.

        이 괴리의 원인은 **feature redundancy**입니다:
        - SHAP: 모델이 피처를 **얼마나 사용하는가**
        - AUROC: 피처를 **추가했을 때 판별력이 올라가는가**

        외부 피처가 추정매출 피처와 상관되어 있으면, 모델은 사용하지만(SHAP > 0)
        새로운 정보를 추가하지는 않습니다(AUROC ≈ 0).
        """
    )

    st.markdown("---")

    # v4: External data role classification table
    st.markdown("#### 외부데이터 역할 분류: 고객에게 어떤 데이터를 안내할 것인가")

    role_df = pd.DataFrame({
        "외부데이터": ["유동인구", "상주인구", "CSI (소비자심리)", "집객시설", "상권변화지표"],
        "역할": ["제한적 보조 신호", "부분적 대체재 후보", "중복/노이즈 가능성", "제한적", "해석 주의"],
        "고객 안내 시 해석": [
            "일부 업종(외식, 커피)에서만 참고 가능",
            "결제 이력 부족 시 지역 기반 proxy로 활용 가능",
            "매출 변화에 이미 반영되어 있어 별도 활용 가치 낮음",
            "단독 증분효과 낮으나 상권 인프라 맥락 제공",
            "'다이나믹' = 안전이 아니라 점포 회전율이 높은 상권",
        ],
    })
    st.dataframe(role_df, hide_index=True, use_container_width=True)

    st.info(
        "**고객 관점 시사점:** 결제 데이터가 없는 상황(신규 가맹점, 현금 위주 점포)에서는 "
        "외부 공공데이터가 약한 보조 신호로 활용될 수 있습니다. 그러나 결제 데이터가 "
        "확보된 환경에서는 공공 외부데이터의 추가적인 개선 효과가 제한적이었습니다."
    )

    st.markdown("---")

    # Display SHAP figures if available
    def fit_to_canvas(img, width=700, height=650):
        from PIL import Image as PILImage
        canvas = PILImage.new("RGB", (width, height), "white")
        ratio = min(width / img.width, height / img.height)
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        resized = img.resize((new_w, new_h))
        x = (width - new_w) // 2
        y = (height - new_h) // 2
        canvas.paste(resized, (x, y))
        return canvas

    col1, col2 = st.columns(2)
    with col1:
        img = load_figure("model_02_shap_summary.png")
        if img:
            st.image(fit_to_canvas(img), caption="SHAP Summary (원본 타겟)", use_container_width=True)
    with col2:
        img = load_figure("model_12_normalized_shap.png")
        if img:
            st.image(fit_to_canvas(img), caption="SHAP Summary (정규화, no store_count)", use_container_width=True)

    business_takeaway(
        "고객에게 외부데이터의 역할을 구분해 안내하면 신뢰도가 높아집니다.\n "
        "결제 데이터가 충분한 경우에는 참고 수준이며, 결제 이력이 부족한 경우에는 초기 리스크 판단을 위한 보조 신호로 활용 가능합니다."
    )


# ================================================================
# Tab 5: Temporal Trend — v4: purpose statement
# ================================================================
def tab_trend():
    st.header("시계열 트렌드")

    st.markdown(
        "이 탭은 데이터 기간 내 구조적 변화를 확인하기 위한 검증입니다. "
        "폐업률과 매출, 점포 구조 변화가 함께 움직이는지, "
        "모델 결과가 특정 분기나 COVID-19 구간에만 의존하는지 점검합니다."
    )

    trend_df = load_csv("quarterly_trend.csv")
    if trend_df is None:
        st.warning("data/exports/quarterly_trend.csv not found. Run: python run_export.py")
        return

    trend_df["quarter"] = trend_df["quarter"].astype(str)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=trend_df["quarter"], y=trend_df["avg_closure_rate"],
                   name="평균 폐업률", line=dict(color="#E8744F", width=2),
                   mode="lines+markers"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=trend_df["quarter"], y=trend_df["obs_count"],
               name="관측수", marker_color="#4682B4", opacity=0.3),
        secondary_y=True,
    )
    fig.update_layout(
        height=500,
        title="분기별 평균 폐업률 추이",
        xaxis_title="분기",
        xaxis_type="category",
    )
    fig.update_yaxes(title_text="평균 폐업률", secondary_y=False)
    fig.update_yaxes(title_text="관측수", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    if "avg_sales" in trend_df.columns and "avg_foot_traffic" in trend_df.columns:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(
            go.Scatter(x=trend_df["quarter"], y=trend_df["avg_sales"],
                       name="평균 매출", line=dict(color="#4682B4", width=2),
                       mode="lines+markers"),
            secondary_y=False,
        )
        fig2.add_trace(
            go.Scatter(x=trend_df["quarter"], y=trend_df["avg_foot_traffic"],
                       name="평균 유동인구", line=dict(color="#2E8B57", width=2),
                       mode="lines+markers"),
            secondary_y=True,
        )
        fig2.update_layout(
            height=450,
            title="분기별 매출 vs 유동인구 추이",
            xaxis_type="category",
        )
        fig2.update_yaxes(title_text="평균 매출", secondary_y=False)
        fig2.update_yaxes(title_text="평균 유동인구", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

    business_takeaway(
        "2020~2022 COVID-19 영향 기간의 폐업률 급변이 모델 학습에 포함되어 있으며, "
        "해당 기간을 분리한 별도 검증으로 일반화 성능을 추가 확인할 수 있습니다."
    )

# ================================================================
# Tab 6: Insights — v4: customer-centric titles
# ================================================================
def tab_insights():
    st.header("분석 결과 및 시사점")

    st.markdown("#### 핵심 발견")

    findings = [
        (
            "공공 외부데이터의 증분효과는 제한적",
            "11개 검증 실험에서 일관되게 확인되었으며 역방향 ablation에서는 "
            "외부 데이터 제거 시 오히려 성능이 개선되었습니다. "
            "기존 추정매출 및 점포 변수 대비 추가 정보량이 제한적입니다.",
            "ΔAUROC: -0.002 ~ +0.004"
        ),
        (
            "SHAP 30.5% ≠ AUROC +0.001 (Feature Redundancy)",
            "정규화 타겟에서 외부 데이터의 SHAP 기여도가 30.5%까지 상승하지만, "
            "이는 추정매출 피처와의 중복 신호입니다. "
            "SHAP 기준으로 모델 예측에 기여하지만, "
            "추정매출 변수가 이미 담고 있는 정보와 중복되어 "
            "실제 판별력 개선으로 이어지지 않습니다.",
            "Redundancy"
        ),
        (
            "상권변화지표는 직관과 반대",
            "'다이나믹(LL)' 등급의 폐업률(4.6%)이 '정체(HH)'(3.8%)보다 "
            "높습니다. 개폐업 회전이 빠른 활성 상권을 의미합니다.",
            "역방향"
        ),
        (
            "폐업은 특정 업종에 구조적으로 집중",
            "치킨(13.4%), 편의점(13.3%), 패스트푸드(10.5%) vs "
            "의원(1.1%), 가전(1.6%). 10배 이상 차이. "
            "이들은 점포 진입·퇴출 회전이 빠른 고변동 업종입니다.",
            "업종별 구조 차이"
        ),
        (
            "누수(Leakage) 검증 완료",
            "closure_rate 제거 시 AUROC 차이 0.0004. "
            "시점 분리(LEAD)가 정상 작동합니다.",
            "검증 통과"
        ),
    ]

    for title, desc, metric in findings:
        with st.expander(f"{title}"):
            st.markdown(desc)
            st.caption(f"핵심 수치: {metric}")

    st.markdown("---")

    # v4: customer-centric insight titles
    st.markdown("#### 대고객 데이터 상품 관점의 시사점")

    proposals = [
        ("고객이 신뢰할 수 있는 리스크 스코어를 위해, 내부 결제·점포 구조 신호의 설명력을 강화",
         "공공 외부데이터보다 자체 결제 패턴 심화가 리스크 스코어 정확도에 더 기여"),
        ("고객용 업종별 리스크 모니터링 리포트 제공",
         "폐업률이 10배 이상 다른 업종에 동일 경고 기준은 비효율적. 업종 맥락을 반영한 차등 리포트 필요"),
        ("외부데이터의 역할을 고객에게 구분해 안내: 보강재 vs 대체재",
         "결제 데이터 부족 시(신규 가맹점 등) 부분적 대체재로 활용 가능하다는 맥락 제공"),
        ("상권변화지표 사용 시 고객 오해 방지를 위한 맥락 정보 제공",
         "'다이나믹 = 위험'이 아닌 '개폐업 회전이 빠른 상권'이라는 해석 안내"),
        ("예측형 리스크 인사이트로 데이터 상품 차별화",
         "현황 분석에서 예측으로 확장하면 고객에게 선제적 의사결정 기반 제공"),
    ]

    for title, desc in proposals:
        st.markdown(f"**{title}**")
        st.caption(desc)

    st.markdown("---")
    st.markdown("#### 외부데이터 상품화 가치 평가")

    value_df = pd.DataFrame({
        "데이터": ["유동인구", "상주인구", "CSI (소비자심리)", "집객시설", "상권변화지표"],
        "예측 lift": ["낮음", "소폭", "음수", "낮음", "낮음"],
        "해석 가능성": ["높음", "중간", "낮음", "높음", "중간"],
        "수집 난이도": ["중간", "낮음", "낮음", "낮음", "낮음"],
        "고객 설명력": ["높음", "중간", "낮음", "높음", "주의 필요"],
        "상품화 가치": ["보조 리포트용", "cold-start 보조", "제외 또는 참고", "입지 리포트용", "해석 주석 필요"],
    })
    st.dataframe(value_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 외부데이터 결합 우선순위 제안")

    priority_df = pd.DataFrame({
        "우선순위": ["1", "2", "3", "4", "5"],
        "데이터": ["사업자 등록 상태 / 휴폐업", "인허가 개폐업", "임대료 / 상가 공실률", "배달앱 / 온라인 리뷰 변화", "유동인구 / 상주인구"],
        "이유": [
            "법적 폐업 신호, 타겟과 직접 관련",
            "업종별 실제 영업 상태 반영",
            "폐업 압력의 비용 측면",
            "소비자 수요 변화의 선행 신호",
            "보조적 환경 신호",
        ],
    })
    st.dataframe(priority_df, hide_index=True, use_container_width=True)

    st.caption("상황에 따라 공공 환경 데이터보다 법적 비용, 수요 변화 신호가 더 높은 우선순위를 가질 수 있습니다.")

    st.markdown("---")
    st.markdown("#### 프로젝트 한계")
    st.markdown(
        "- 상권×업종 단위 공공 집계 데이터 (개별 가맹점 단위 아님)\n"
        "- 인허가 정보, 사업자 등록 상태 미확보\n"
        "- COVID-19 팬데믹 영향 기간 포함 (2020~2022)\n"
        "- 서울 지역 한정\n"
        "- 분기 단위 시간 해상도 (실시간 감지 불가)"
    )

    st.markdown("---")
    st.caption(
        "본 프로젝트는 카드사 내부 데이터가 아닌 서울시 공공 상권 데이터를 사용했으며, "
        "개별 가맹점 단위 모델을 대체하기보다 공공 외부데이터의 증분효과를 검증하고 "
        "대고객 데이터 상품화 방향을 탐색하는 데 목적이 있습니다.\n"
        "AUPRC random baseline = test set positive rate (0.228)."
    )


# ================================================================
# Main
# ================================================================
def main():
    render_sidebar()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "프로젝트 동기",
        "Ablation Study",
        "업종별 리스크",
        "SHAP & Redundancy",
        "시계열 트렌드",
        "시사점",
        "고객 리포트",
    ])

    with tab1:
        tab_motivation()
    with tab2:
        tab_ablation()
    with tab3:
        tab_industry()
    with tab4:
        tab_shap()
    with tab5:
        tab_trend()
    with tab6:
        tab_insights()
    with tab7:
        tab_customer_report()


if __name__ == "__main__":
    main()
