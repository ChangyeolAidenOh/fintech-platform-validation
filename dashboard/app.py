"""
Fintech Platform Validation — Streamlit Dashboard

Tabs:
  1. Project Motivation (ABP Gap Analysis)
  2. Ablation Study Results
  3. Industry Risk Analysis
  4. SHAP & Feature Redundancy
  5. Temporal Trend
  6. Insights & Proposals

Usage: streamlit run dashboard/app.py
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

st.set_page_config(
    page_title="Fintech Platform Validation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data/exports"
FIG_DIR = "figures"


# ================================================================
# Data Loading
# ================================================================
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


# ================================================================
# Sidebar
# ================================================================
def render_sidebar():
    st.sidebar.title("📊 Fintech Platform Validation")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "금융 빅데이터 플랫폼의 자체 인정 한계를 "
        "공공데이터로 실증 검증한 프로젝트입니다."
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
        "- **Split:** Time-based (train ~2024Q3, test 2024Q4~)\n"
        "- **Target:** Next-quarter closure rate > threshold\n"
        "- **Experiments:** 11 independent ablation tests\n"
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
# Tab 1: Project Motivation
# ================================================================
def tab_motivation():
    st.header("1. Project Motivation: 플랫폼이 인정한 한계")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
            금융 빅데이터 플랫폼(ABP)에 직접 가입하여 AI검색 기능을 사용했습니다.
            **"카드 결제 데이터만으로 폐업을 예측할 수 있나요?"** 라고 질문했을 때,
            플랫폼은 다음과 같이 안내했습니다:
            """
        )

        st.info(
            "💬 \"카드 결제 데이터(추정매출)만으로 폐업을 완벽하게 예측하는 건 "
            "어렵습니다. 인허가 정보, 사업자 신고 상태, 주변 시장 상황 등 "
            "여러 데이터가 같이 참고되어야 정확도가 올라가요.\""
        )

        st.markdown(
            """
            플랫폼이 지목한 외부 데이터:
            - 인허가 정보
            - 사업자 신고 상태
            - 상권 변화 정보
            - 유동인구 데이터
            - 임대료 및 부동산 정보
            - 지역 경제지표

            이 중 **공공 수준에서 확보 가능한 외부 환경 데이터**의 증분효과를 실증 검증했습니다.
            """
        )

    with col2:
        st.markdown("#### 확인된 플랫폼 한계")
        limitations = {
            "예측 기능 없음": "현황 분석만 제공, 미래 예측 불가",
            "폐업 데이터 미포함": "데이터폴리오에 폐업 변수 없음",
            "시계열 분석 불가": "전월 1개월 스냅샷만 제공",
            "입지 추천 없음": "기존 상위 매출 매장 리스트만 제공",
            "경쟁 분석 제한적": "간접적 프록시 지표만 가능",
        }
        for title, desc in limitations.items():
            st.markdown(f"**{title}**")
            st.caption(desc)

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
# Tab 2: Ablation Study
# ================================================================
def tab_ablation():
    st.header("2. Ablation Study: 외부 데이터 증분효과 검증")

    st.markdown("#### 원본 타겟 (절대 폐업률 예측)")

    # Ablation results - hardcoded from experiments
    ablation_data = pd.DataFrame([
        {"Model": "LR Baseline", "AUROC": 0.6325, "Type": "Baseline"},
        {"Model": "Card Only (A)", "AUROC": 0.7830, "Type": "Card"},
        {"Model": "Card + All External (B)", "AUROC": 0.7811, "Type": "Full"},
        {"Model": "Card + Traffic", "AUROC": 0.7836, "Type": "Individual"},
        {"Model": "Card + Change Index", "AUROC": 0.7834, "Type": "Individual"},
        {"Model": "Card + Facilities", "AUROC": 0.7837, "Type": "Individual"},
        {"Model": "Card + CSI", "AUROC": 0.7813, "Type": "Individual"},
        {"Model": "Card + Residents", "AUROC": 0.7840, "Type": "Individual"},
    ])

    color_map = {"Baseline": "#CCCCCC", "Card": "#4682B4", "Full": "#2E8B57", "Individual": "#E8744F"}

    fig = px.bar(
        ablation_data, x="Model", y="AUROC", color="Type",
        color_discrete_map=color_map,
        text=ablation_data["AUROC"].apply(lambda x: f"{x:.4f}"),
    )
    fig.add_hline(y=0.7830, line_dash="dash", line_color="#4682B4",
                  annotation_text="Card Only baseline: 0.7830")
    fig.update_layout(
        height=400, showlegend=True,
        yaxis_range=[0.6, 0.80],
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Card Only AUROC", "0.7830")
    with col2:
        st.metric("Card + External AUROC", "0.7811", delta="-0.0018")
    with col3:
        st.metric("외부 데이터 제거 시", "0.7830", delta="+0.0018",
                  help="역방향 ablation: 외부 데이터를 전부 빼면 오히려 개선")

    st.markdown("---")

    st.markdown("#### Naive Baseline 비교")
    st.markdown(
        "모델의 실질적 기여를 파악하려면 naive baseline과 비교해야 합니다."
    )

    naive_data = pd.DataFrame([
        {"Model": "Naive: 직전 분기 폐업률", "AUROC": 0.6059, "Category": "Naive"},
        {"Model": "Naive: 업종 평균", "AUROC": 0.6049, "Category": "Naive"},
        {"Model": "Naive: store_count 단독", "AUROC": 0.7445, "Category": "Naive"},
        {"Model": "XGB: Card + Store (strict)", "AUROC": 0.7829, "Category": "XGBoost"},
    ])

    fig2 = px.bar(
        naive_data, x="Model", y="AUROC", color="Category",
        color_discrete_map={"Naive": "#CCCCCC", "XGBoost": "#4682B4"},
        text=naive_data["AUROC"].apply(lambda x: f"{x:.4f}"),
    )
    fig2.update_layout(height=350, yaxis_range=[0.5, 0.82], xaxis_tickangle=-20)
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "XGBoost는 최선의 naive baseline(store_count 단독, 0.7445) 대비 "
        "+0.038 AUROC 개선. P@100 = 99%는 주로 store_count의 크기 효과에 기인합니다."
    )


# ================================================================
# Tab 3: Industry Risk
# ================================================================
def tab_industry():
    st.header("3. 업종별 리스크 분석")

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
            height=600,
            yaxis=dict(autorange="reversed"),
            xaxis_title="평균 분기 폐업률",
            yaxis_title="",
            title="Top 20 업종별 평균 폐업률",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 업종별 폐업률 차이")
        st.markdown(
            "Kruskal-Wallis H=36,494, p≈0\n\n"
            "업종 간 폐업률 차이는 극도로 유의합니다."
        )
        st.markdown("---")
        st.markdown("#### 3-Tier 모니터링 제안")
        st.markdown(
            "**Tier 1 (고회전):** 치킨, 편의점, 패스트푸드\n\n"
            "**Tier 2 (중간):** 커피, 분식, 중식\n\n"
            "**Tier 3 (안정):** 의원, 치과, 가전"
        )

    st.markdown("---")
    st.markdown("#### 76.3%의 상권-업종에서 분기 내 폐업이 0건")
    st.markdown(
        "폐업은 특정 고회전 업종에 구조적으로 집중됩니다. "
        "모든 업종에 동일 리스크 모델을 적용하는 것보다 "
        "업종 특성을 반영한 차등 체계가 효율적입니다."
    )


# ================================================================
# Tab 4: SHAP & Redundancy
# ================================================================
def tab_shap():
    st.header("4. SHAP 분석 & Feature Redundancy")

    st.markdown("#### SHAP 기여도: 원본 vs 정규화 타겟")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**원본 타겟 (with store_count)**")
        original_shap = pd.DataFrame({
            "Group": ["Store Context", "Card Sales", "External"],
            "SHAP %": [73, 19, 6],
        })
        fig1 = px.pie(original_shap, values="SHAP %", names="Group",
                      color="Group",
                      color_discrete_map={
                          "Store Context": "#4682B4",
                          "Card Sales": "#87CEEB",
                          "External": "#E8744F"
                      })
        fig1.update_layout(height=300)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("**정규화 타겟 (no store_count)**")
        norm_shap = pd.DataFrame({
            "Group": ["Store Context", "Card Sales", "External"],
            "SHAP %": [36.6, 32.9, 30.5],
        })
        fig2 = px.pie(norm_shap, values="SHAP %", names="Group",
                      color="Group",
                      color_discrete_map={
                          "Store Context": "#4682B4",
                          "Card Sales": "#87CEEB",
                          "External": "#E8744F"
                      })
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.markdown("#### SHAP 30.5% vs AUROC +0.001 — Feature Redundancy")
    st.markdown(
        """
        정규화 타겟에서 외부 데이터의 SHAP 기여도가 **6% → 30.5%**로 5배 상승했습니다.
        그러나 실제 AUROC 개선은 +0.001 수준에 그쳤습니다.

        이 괴리의 원인은 **feature redundancy**입니다:
        - SHAP: 모델이 피처를 **얼마나 사용하는가**
        - AUROC: 피처를 **추가했을 때 판별력이 올라가는가**

        외부 피처가 카드 피처와 상관되어 있으면, 모델은 사용하지만(SHAP > 0)
        새로운 정보를 추가하지는 않습니다(AUROC ≈ 0).
        """
    )

    st.info(
        "💡 **실무적 시사점:** 카드 데이터가 없는 상황(신규 가맹점, 현금 위주 점포)에서는 "
        "외부 공공데이터가 부분적 대체재로 기능할 수 있습니다. "
        "그러나 카드 데이터가 있는 한, 공공 외부데이터는 이를 보강하지 못합니다."
    )

    # Display SHAP figures if available
    col1, col2 = st.columns(2)
    with col1:
        img = load_figure("model_02_shap_summary.png")
        if img:
            img_resized = img.resize((700, 560))
            st.image(img_resized, caption="SHAP Summary (원본 타겟)", use_container_width=True)
    with col2:
        img = load_figure("model_12_normalized_shap.png")
        if img:
            img_resized = img.resize((700, 560))
            st.image(img_resized, caption="SHAP Summary (정규화, no store_count)", use_container_width=True)

# ================================================================
# Tab 5: Temporal Trend
# ================================================================
def tab_trend():
    st.header("5. 시계열 트렌드")

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
        height=400,
        title="분기별 평균 폐업률 추이",
        xaxis_title="분기",
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
        fig2.update_layout(height=350, title="분기별 매출 vs 유동인구 추이")
        fig2.update_yaxes(title_text="평균 매출", secondary_y=False)
        fig2.update_yaxes(title_text="평균 유동인구", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)


# ================================================================
# Tab 6: Insights
# ================================================================
def tab_insights():
    st.header("6. 분석 결과 및 시사점")

    st.markdown("#### 핵심 발견")

    findings = [
        (
            "공공 외부데이터의 증분효과는 제한적",
            "11개 독립 실험에서 일관되게 확인. 역방향 ablation에서는 "
            "외부 데이터 제거 시 오히려 성능이 개선되었습니다.",
            "-0.0018 ~ +0.004"
        ),
        (
            "SHAP 30.5% ≠ AUROC +0.001 (Feature Redundancy)",
            "외부 데이터는 카드 데이터와 중복 신호를 포착합니다. "
            "모델이 '사용'하지만 '새 정보'는 아닙니다.",
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
            "의원(1.1%), 가전(1.6%). 10배 이상 차이.",
            "업종 집중"
        ),
        (
            "누수(Leakage) 검증 완료",
            "closure_rate 제거 시 AUROC 차이 0.0004. "
            "시점 분리(LEAD)가 정상 작동합니다.",
            "검증 통과"
        ),
    ]

    for title, desc, metric in findings:
        with st.expander(f"📌 {title}"):
            st.markdown(desc)
            st.caption(f"핵심 수치: {metric}")

    st.markdown("---")
    st.markdown("#### BC카드 시사점")

    proposals = [
        ("결제·가맹점 데이터 심화 분석 우선", "외부 공공데이터 확장보다 자체 결제 패턴 심화가 더 높은 우선순위"),
        ("업종별 차등 모니터링", "폐업률이 10배 이상 다른 업종에 동일 임계값은 비효율적"),
        ("외부 데이터의 가치는 '대체재'", "카드 데이터 부족 시(신규 가맹점 등) 부분적 대체재로 활용 가능"),
        ("상권변화지표 맥락 정보 보강", "'다이나믹 = 위험'이 아닌 '개폐업 회전이 빠른 상권'으로 안내"),
        ("예측 기능 상품화 가능성", "현황 분석에서 예측으로 확장하면 데이터 상품 차별화 가능"),
    ]

    for title, desc in proposals:
        st.markdown(f"**{title}:** {desc}")

    st.markdown("---")
    st.markdown("#### 프로젝트 한계")
    st.markdown(
        "- 상권×업종 단위 공공 집계 데이터 (개별 가맹점 단위 아님)\n"
        "- 인허가 정보, 사업자 등록 상태 미확보\n"
        "- COVID-19 기간 포함 (2019~2025)\n"
        "- 서울 지역 한정\n"
        "- 분기 단위 시간 해상도 (실시간 감지 불가)"
    )


# ================================================================
# Main
# ================================================================
def main():
    render_sidebar()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. 프로젝트 동기",
        "2. Ablation Study",
        "3. 업종별 리스크",
        "4. SHAP & Redundancy",
        "5. 시계열 트렌드",
        "6. 시사점",
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


if __name__ == "__main__":
    main()
