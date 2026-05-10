"""
Customer Report Tab — Interactive Risk Assessment Prototype.

Add this as Tab 7 in dashboard/app.py.
Users select a district and industry → get risk grade, reason codes, interpretation.

To integrate: 
  1. Import this function in app.py
  2. Add tab in main()
  3. Call tab_customer_report() inside the tab

Usage in app.py:
  from customer_report_tab import tab_customer_report
  
  # In main(), add to tabs list:
  tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([...existing..., "고객 리포트"])
  with tab7:
      tab_customer_report()
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

DATA_DIR = "data/exports"


def business_takeaway(text):
    st.markdown(f"""
    <div style="background-color: #f0f2f6; border-left: 4px solid #4682B4;
                padding: 12px 16px; margin-top: 20px; border-radius: 4px;">
        <strong>Business Takeaway:</strong> {text}
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_risk_data():
    path = os.path.join(DATA_DIR, "risk_features_sample.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_industry_summary():
    path = os.path.join(DATA_DIR, "industry_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def compute_risk_grade(row, industry_avg_closure):
    """Compute risk grade based on multiple signals."""
    score = 0
    reasons = []

    # R1: High competition density
    if pd.notna(row.get("competition_density")) and row["competition_density"] > 2.0:
        score += 2
        reasons.append(("R1", "동종업종 점포 밀도 높음",
                        f"경쟁밀도 {row['competition_density']:.1f} (평균 대비 높음)"))

    # R2: Sales decline
    if pd.notna(row.get("sales_qoq_change")) and row["sales_qoq_change"] < -0.1:
        score += 2
        reasons.append(("R2", "직전 분기 매출 감소",
                        f"매출 QoQ {row['sales_qoq_change']:.1%} 하락"))

    # R3: Closure rate above industry average
    if pd.notna(row.get("closure_rate")) and pd.notna(industry_avg_closure):
        if row["closure_rate"] > industry_avg_closure * 1.5:
            score += 3
            reasons.append(("R3", "업종 평균 대비 폐업률 상승",
                            f"폐업률 {row['closure_rate']:.1%} vs 업종 평균 {industry_avg_closure:.1%}"))

    # R4: High store turnover
    if pd.notna(row.get("store_count_qoq_change")) and abs(row["store_count_qoq_change"]) > 0.1:
        score += 1
        reasons.append(("R4", "점포 회전율 높은 상권",
                        f"점포 수 QoQ 변화 {row['store_count_qoq_change']:.1%}"))

    # R5: Low transaction conversion relative to foot traffic
    if (pd.notna(row.get("total_foot_traffic")) and
            pd.notna(row.get("total_txn_count")) and
            row["total_foot_traffic"] > 0):
        conversion = row["total_txn_count"] / row["total_foot_traffic"]
        if conversion < 0.001:
            score += 1
            reasons.append(("R5", "유동인구 대비 매출 전환 약함",
                            f"전환율 {conversion:.6f}"))

    # Determine grade
    if score >= 5:
        grade = "High"
        grade_color = "#E8744F"
        grade_label = "주의 필요"
    elif score >= 3:
        grade = "Medium"
        grade_color = "#FFB347"
        grade_label = "모니터링 권장"
    else:
        grade = "Low"
        grade_color = "#2E8B57"
        grade_label = "안정적"

    return grade, grade_color, grade_label, score, reasons


def get_action_recommendation(reasons):
    """Map reason codes to actionable recommendations."""
    actions = []
    reason_codes = [r[0] for r in reasons]

    if "R2" in reason_codes and "R1" in reason_codes:
        actions.append("경쟁 심화 속 매출 하락 — 차별화 전략 또는 업종 전환 검토")
    elif "R2" in reason_codes:
        actions.append("매출 감소 추세 — 원인 분석 및 매출 회복 전략 필요")

    if "R5" in reason_codes:
        actions.append("유동인구 대비 전환율 낮음 — 마케팅·가시성 개선 검토")

    if "R3" in reason_codes:
        actions.append("업종 평균 대비 폐업률 높음 — 해당 상권·업종 조합의 구조적 리스크 점검")

    if "R4" in reason_codes:
        actions.append("점포 회전이 빠른 상권 — 기회와 리스크가 동시 존재, 진입 시 신중한 판단 필요")

    if not actions:
        actions.append("현재 주요 위험 신호 없음 — 정기 모니터링 유지")

    return actions


def tab_customer_report():
    st.header("7. 고객용 리스크 리포트 (Prototype)")

    st.markdown(
        "사용자가 상권과 업종을 선택하면 리스크 등급, 주요 위험 요인, "
        "해석을 제공하는 데이터 상품 프로토타입입니다."
    )

    df = load_risk_data()
    industry_df = load_industry_summary()

    if df is None:
        st.warning("data/exports/risk_features_sample.csv not found. Run: python run_export.py")
        return

    # Latest quarter only
    latest_q = df["stdr_yyqu_cd"].max()
    df_latest = df[df["stdr_yyqu_cd"] == latest_q].copy()

    q_year = str(latest_q)[:4]
    q_num = str(latest_q)[-1]
    st.caption(
        f"데이터 기준: {q_year}년 {q_num}분기 | "
        f"상권 {df_latest['trdar_cd_nm'].nunique()}개 | "
        f"업종 {df_latest['svc_induty_cd_nm'].nunique()}개"
    )
    st.caption(
        "타겟 변수(다음 분기 폐업률)를 구성하기 위해 마지막 분기는 피처 시점으로만 사용되며, "
        "가장 최근 데이터가 기준이 됩니다."
    )

    st.markdown("---")

    # User selection
    col_sel1, col_sel2 = st.columns(2)

    with col_sel1:
        districts = sorted(df_latest["trdar_cd_nm"].unique())
        selected_district = st.selectbox("상권 선택", districts, index=0)

    # Filter industries available in selected district
    available_industries = sorted(
        df_latest[df_latest["trdar_cd_nm"] == selected_district]["svc_induty_cd_nm"].unique()
    )

    with col_sel2:
        selected_industry = st.selectbox("업종 선택", available_industries, index=0)

    # Get the row
    row_df = df_latest[
        (df_latest["trdar_cd_nm"] == selected_district) &
        (df_latest["svc_induty_cd_nm"] == selected_industry)
    ]

    if row_df.empty:
        st.warning("선택한 상권·업종 조합에 대한 데이터가 없습니다.")
        return

    row = row_df.iloc[0]

    # Get industry average closure rate
    industry_avg_closure = None
    if industry_df is not None:
        ind_match = industry_df[industry_df["industry"] == selected_industry]
        if not ind_match.empty:
            industry_avg_closure = ind_match.iloc[0]["avg_closure_rate"]

    # Compute risk
    grade, grade_color, grade_label, score, reasons = compute_risk_grade(row, industry_avg_closure)
    actions = get_action_recommendation(reasons)

    st.markdown("---")

    # Risk Grade Display
    col_grade, col_details = st.columns([1, 2])

    with col_grade:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 30px; border-radius: 12px;
                        background-color: {grade_color}; color: white;">
                <h2 style="margin: 0; color: white;">리스크 등급</h2>
                <h1 style="margin: 10px 0; font-size: 48px; color: white;">{grade}</h1>
                <p style="margin: 0; font-size: 18px; color: white;">{grade_label}</p>
                <p style="margin: 5px 0 0 0; font-size: 14px; color: rgba(255,255,255,0.8);">Risk Score: {score}/9 (높을수록 위험)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_details:
        st.markdown(f"**{selected_district}** — {selected_industry}")

        # Key metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            sales = row.get("total_sales", 0)
            st.metric("분기 매출", f"{sales/1e6:.0f}M" if pd.notna(sales) and sales > 0 else "N/A")
        with m2:
            store_count = row.get("store_count", 0)
            st.metric("점포 수", f"{int(store_count)}" if pd.notna(store_count) else "N/A")
        with m3:
            closure = row.get("closure_rate", 0)
            st.metric("현재 폐업률", f"{closure:.1%}" if pd.notna(closure) else "N/A")
        with m4:
            traffic = row.get("total_foot_traffic", 0)
            st.metric("유동인구", f"{traffic/1e3:.0f}K" if pd.notna(traffic) and traffic > 0 else "N/A")

    st.markdown("---")

    # Reason Codes
    st.markdown("#### 주요 위험 요인 (Reason Codes)")

    if reasons:
        for code, title, detail in reasons:
            st.markdown(
                f"""
                <div style="background-color: #fff3cd; border-left: 4px solid #ffc107;
                            padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                    <strong>{code}:</strong> {title}<br>
                    <small style="color: #666;">{detail}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.success("주요 위험 요인이 감지되지 않았습니다.")

    st.markdown("---")

    # Actionable Recommendations
    st.markdown("#### 해석 및 권장 사항")

    for action in actions:
        st.markdown(f"- {action}")

    # Context: industry comparison
    if industry_avg_closure is not None:
        st.markdown("---")
        st.markdown("#### 업종 맥락")

        current_closure = row.get("closure_rate", 0) if pd.notna(row.get("closure_rate")) else 0

        st.markdown(
            f"- **{selected_industry}** 전체 평균 분기 폐업률: {industry_avg_closure:.1%}\n"
            f"- 현재 상권 폐업률: {current_closure:.1%}\n"
            f"- 업종 평균 대비: {'높음' if current_closure > industry_avg_closure else '낮음 또는 동일'}"
        )

        # Industry tier
        if industry_avg_closure >= 0.08:
            tier = "Tier 1 (고회전·고변동)"
            tier_msg = "이 업종은 점포 진입·퇴출 회전이 빠른 구조입니다. 폐업률 자체가 높다기보다 사업 회전 주기가 짧습니다."
        elif industry_avg_closure >= 0.04:
            tier = "Tier 2 (중간)"
            tier_msg = "업종 평균 수준의 폐업률입니다. 정기 모니터링을 권장합니다."
        else:
            tier = "Tier 3 (안정)"
            tier_msg = "구조적으로 안정적인 업종입니다. 폐업이 발생할 경우 환경적 요인을 점검할 필요가 있습니다."

        st.info(f"**{tier}:** {tier_msg}")

    st.markdown("---")

    st.caption(
        "본 리포트는 서울시 공공 상권 데이터 기반의 프로토타입이며, "
        "개별 가맹점 단위 실시간 리스크 평가를 대체하지 않습니다. "
        "실제 의사결정에는 추가적인 현장 확인과 전문 상담이 필요합니다."
    )

    business_takeaway(
        "이 프로토타입은 리스크 등급 + reason code + 업종 맥락 + 권장 액션을 "
        "하나의 리포트로 통합합니다. 카드사 내부 결제 데이터가 추가되면 "
        "reason code의 정밀도와 액션의 구체성이 높아질 수 있습니다."
    )
