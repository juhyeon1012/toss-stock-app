import os
import re
import unicodedata
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from PIL import Image
import easyocr

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="토스 포트폴리오 분석", layout="wide", page_icon="📊")

FONT_PATH = "NanumGothic-Regular.ttf"
HISTORY_FILE = "portfolio_history.csv"

# ─────────────────────────────────────────────
# 한글 폰트
# ─────────────────────────────────────────────
try:
    font_prop = fm.FontProperties(fname=FONT_PATH)
    matplotlib.rcParams["font.family"] = font_prop.get_name()
except Exception:
    font_prop = None
matplotlib.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────
# OCR 로더
# ─────────────────────────────────────────────
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["ko", "en"], gpu=False)


# ─────────────────────────────────────────────
# 기록 관리
# ─────────────────────────────────────────────
def empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "id", "timestamp", "batch_name",
        "total_amount", "estimated_principal", "estimated_profit", "stock_count"
    ])


def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return empty_history_df()
    try:
        df = pd.read_csv(HISTORY_FILE)
        for col in ["id", "total_amount", "estimated_principal", "estimated_profit", "stock_count"]:
            if col not in df.columns:
                df[col] = 0
        return df
    except Exception:
        return empty_history_df()


def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def append_history(batch_name, total_amount, estimated_principal, estimated_profit, stock_count):
    old_df = load_history()
    next_id = 1 if old_df.empty else int(pd.to_numeric(old_df["id"], errors="coerce").max()) + 1
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([{
        "id": next_id, "timestamp": now_str, "batch_name": batch_name,
        "total_amount": int(total_amount), "estimated_principal": int(estimated_principal),
        "estimated_profit": int(estimated_profit), "stock_count": int(stock_count),
    }])
    save_history(pd.concat([old_df, new_row], ignore_index=True))


def delete_history_row(row_id: int):
    df = load_history()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    save_history(df[df["id"] != row_id].copy())


# ─────────────────────────────────────────────
# 텍스트 유틸
# ─────────────────────────────────────────────
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text).strip())
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def format_krw(x) -> str:
    try:
        return f"{int(round(float(x))):,}원"
    except Exception:
        return str(x)


def format_usd(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


# ─────────────────────────────────────────────
# 파싱 함수
# ─────────────────────────────────────────────
def detect_currency(texts: str) -> str:
    """전체 OCR 텍스트에서 통화 모드 판단"""
    usd = 5 if re.search(r"\$\d", texts) else (2 if "$" in texts else 0)
    krw = 5 if re.search(r"\d[\d,]*원", texts) else (1 if "원" in texts else 0)
    return "USD" if usd > krw else "KRW"


def parse_amount_krw(text: str) -> Optional[int]:
    """'1,308,125원' 형태 → int"""
    m = re.search(r"(\d[\d,]+)원", text)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except Exception:
            return None
    return None


def parse_amount_usd(text: str) -> Optional[float]:
    """'$1,234.56' 형태 → float"""
    m = re.search(r"\$([\d,]+\.?\d*)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except Exception:
            return None
    return None


def parse_profit_krw(text: str) -> Optional[int]:
    """'+610,000원 (108.16%)' 또는 '-203,800원 (16.21%)' → int"""
    m = re.search(r"([+-])(\d[\d,]+)원", text)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        try:
            return sign * int(m.group(2).replace(",", ""))
        except Exception:
            return None
    return None


def parse_profit_usd(text: str) -> Optional[float]:
    """'[+-]$1,234.56' → float"""
    m = re.search(r"([+-])\$([\d,]+\.?\d*)", text)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        try:
            return sign * float(m.group(2).replace(",", ""))
        except Exception:
            return None
    return None


def is_shares_text(text: str) -> bool:
    return bool(re.search(r"\d+(\.\d+)?주", text))


# ─────────────────────────────────────────────
# OCR → 행 그룹핑
# ─────────────────────────────────────────────
def ocr_boxes(reader, pil_img: Image.Image) -> List[Dict]:
    gray = np.array(pil_img.convert("L"))
    results = reader.readtext(gray, detail=1, paragraph=False)
    boxes = []
    for box, text, conf in results:
        if conf < 0.1:
            continue
        text = normalize_text(text)
        if not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        boxes.append({
            "text": text, "conf": conf,
            "x1": x1, "x2": x2, "y1": y1, "y2": y2,
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "w": x2 - x1, "h": y2 - y1,
        })
    return boxes


def group_rows(boxes: List[Dict], row_gap_ratio: float = 0.7) -> List[Dict]:
    """y 좌표 기준으로 같은 줄끼리 묶기"""
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b["cy"])
    rows = []
    for b in sorted_boxes:
        placed = False
        for row in rows:
            threshold = max(10, row["avg_h"] * row_gap_ratio)
            if abs(b["cy"] - row["cy"]) <= threshold:
                row["boxes"].append(b)
                row["cy"] = np.mean([rb["cy"] for rb in row["boxes"]])
                row["avg_h"] = np.mean([rb["h"] for rb in row["boxes"]])
                placed = True
                break
        if not placed:
            rows.append({"boxes": [b], "cy": b["cy"], "avg_h": b["h"]})

    result = []
    for row in rows:
        row_boxes = sorted(row["boxes"], key=lambda b: b["x1"])
        result.append({
            "boxes": row_boxes,
            "text": normalize_text(" ".join(rb["text"] for rb in row_boxes)),
            "cy": row["cy"],
            "x1": min(rb["x1"] for rb in row_boxes),
            "x2": max(rb["x2"] for rb in row_boxes),
            "avg_h": row["avg_h"],
        })
    return sorted(result, key=lambda r: r["cy"])


# ─────────────────────────────────────────────
# 핵심: 토스 레이아웃 파싱
#
# 토스 화면 구조 (한 종목당):
#   [아이콘]  종목명 (1~2줄)          평가금액원
#             N주                    +/-손익원 (X%)
#
# 전략:
#   1. 오른쪽 영역(x1 > 55% width)에서 '숫자원' 패턴 행 = 평가금액
#   2. 바로 아래 행에서 '+/-숫자원' 패턴 = 손익
#   3. 같은 cy 대역 왼쪽 행들 = 종목명 + 주수
# ─────────────────────────────────────────────
def extract_stocks(
    reader,
    pil_img: Image.Image,
    usd_krw_rate: float = 1350.0,
) -> Tuple[List[Dict], str]:

    boxes = ocr_boxes(reader, pil_img)
    all_text = " ".join(b["text"] for b in boxes)
    currency = detect_currency(all_text)
    rows = group_rows(boxes)

    if not rows:
        return [], currency

    img_w = pil_img.size[0]
    # 토스 앱: 평가금액은 오른쪽 절반에 위치
    right_threshold = img_w * 0.50

    # 파싱 함수 선택
    parse_amount = parse_amount_usd if currency == "USD" else parse_amount_krw
    parse_profit = parse_profit_usd if currency == "USD" else parse_profit_krw

    # 각 행을 분류
    classified = []
    for i, row in enumerate(rows):
        txt = row["text"]
        # 오른쪽에 있는 행인지
        is_right = row["x1"] > right_threshold or (
            row["x2"] > img_w * 0.75 and row["x1"] > img_w * 0.40
        )
        amt = parse_amount(txt) if is_right else None
        pft = parse_profit(txt) if is_right else None
        shares = is_shares_text(txt)

        classified.append({
            "idx": i,
            "row": row,
            "text": txt,
            "is_right": is_right,
            "amount": amt,
            "profit": pft,
            "is_shares": shares,
        })

    # 평가금액 행 찾기 → 각 종목 구성
    stocks = []
    used_rows = set()

    for c in classified:
        if c["amount"] is None:
            continue
        if c["idx"] in used_rows:
            continue

        amt_row_idx = c["idx"]
        amt_cy = c["row"]["cy"]
        avg_h = c["row"]["avg_h"]

        # ── 손익 행: 바로 아래 오른쪽 행
        profit_val = 0.0
        for c2 in classified:
            if c2["idx"] <= amt_row_idx:
                continue
            if c2["idx"] in used_rows:
                continue
            dy = c2["row"]["cy"] - amt_cy
            if dy < 0 or dy > avg_h * 2.5:
                continue
            if not c2["is_right"]:
                continue
            if c2["profit"] is not None:
                profit_val = c2["profit"]
                used_rows.add(c2["idx"])
                break

        # ── 왼쪽 행: 같은 cy 대역 (평가금액 기준 위아래 2.5줄)
        name_parts = []
        shares_text = ""
        for c3 in classified:
            if c3["is_right"]:
                continue
            if c3["is_shares"]:
                # 주수는 평가금액 행 아래 가까이
                dy = abs(c3["row"]["cy"] - amt_cy)
                if dy < avg_h * 3:
                    shares_text = c3["text"]
                    used_rows.add(c3["idx"])
                continue
            dy = c3["row"]["cy"] - amt_cy
            if -avg_h * 2.0 <= dy <= avg_h * 1.5:
                txt = c3["text"]
                # 노이즈 필터
                if txt in ("내 투자", "신규 투자", "내투자", "신규투자"):
                    continue
                if len(txt) < 2:
                    continue
                if c3["idx"] not in used_rows:
                    name_parts.append((c3["row"]["cy"], txt))
                    used_rows.add(c3["idx"])

        used_rows.add(amt_row_idx)

        # 종목명: cy 순 정렬 후 합치기
        name_parts.sort(key=lambda x: x[0])
        raw_name = " ".join(p[1] for p in name_parts).strip()
        if not raw_name:
            continue

        # 금액 변환
        amount = c["amount"]
        if currency == "USD":
            amount_krw = round(amount * usd_krw_rate)
            profit_krw = round(profit_val * usd_krw_rate)
        else:
            amount_krw = round(amount)
            profit_krw = round(profit_val)

        principal_krw = amount_krw - profit_krw

        stocks.append({
            "종목명": raw_name,
            "주수": shares_text,
            "통화": currency,
            "평가금액_원본": amount,
            "손익_원본": profit_val,
            "평가금액_원화": amount_krw,
            "손익_원화": profit_krw,
            "원금_원화": principal_krw,
        })

    return stocks, currency


# ─────────────────────────────────────────────
# 차트
# ─────────────────────────────────────────────
def pie_chart(df: pd.DataFrame, label_col: str, value_col: str, title: str):
    if df.empty:
        st.info(f"{title}: 데이터 없음")
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(df))]
    kw = dict(
        x=df[value_col], labels=df[label_col],
        autopct="%1.1f%%", colors=colors,
        startangle=90, wedgeprops={"edgecolor": "black", "linewidth": 0.8},
    )
    if font_prop:
        kw["textprops"] = {"fontproperties": font_prop, "fontsize": 10}
    ax.pie(**kw)
    title_kw = {"fontproperties": font_prop} if font_prop else {}
    ax.set_title(title, fontsize=16, **title_kw)
    st.pyplot(fig)
    plt.close(fig)


def line_chart(history_df: pd.DataFrame, ycols: List[str], title: str):
    if history_df.empty:
        return
    df = history_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in ycols:
        ax.plot(df["timestamp"], df[col], marker="o", label=col)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("원")
    if font_prop:
        ax.set_title(title, fontproperties=font_prop, fontsize=14)
    else:
        ax.set_title(title, fontsize=14)
    ax.legend()
    plt.xticks(rotation=30)
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────
# 미리보기
# ─────────────────────────────────────────────
def show_previews(files):
    cols = st.columns(min(len(files), 4))
    for i, f in enumerate(files):
        with cols[i % 4]:
            img = Image.open(f).convert("RGB")
            st.image(img, caption=f.name, use_container_width=True)
            f.seek(0)  # ← 중요: OCR에서 다시 읽을 수 있게 포인터 리셋


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    st.title("📊 토스 포트폴리오 자동 분석")
    st.caption("토스 앱 캡처 이미지를 올리면 종목·평가금액·손익을 자동으로 추출합니다.")

    # ── 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        usd_krw_rate = st.number_input(
            "달러 → 원 환율",
            min_value=1000.0, max_value=2000.0,
            value=1350.0, step=1.0,
        )
        st.caption("해외 화면이 달러 표시일 때 사용합니다.")
        debug_mode = st.checkbox("디버그 모드 (OCR 원문 확인)", value=False)

    # ── 파일 업로드
    uploaded_files = st.file_uploader(
        "토스 캡처 이미지 업로드 (여러 장 가능)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    batch_name = st.text_input(
        "기록 이름",
        value=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    if not uploaded_files:
        st.info("이미지를 업로드하면 분석이 시작됩니다.")
        st.divider()
        _show_history()
        return

    # ── 미리보기
    st.subheader("📷 업로드된 이미지")
    show_previews(uploaded_files)

    # ── OCR
    try:
        with st.spinner("OCR 엔진 준비 중..."):
            reader = load_ocr_reader()
    except Exception as e:
        st.error(f"OCR 엔진 오류: {e}")
        return

    all_stocks = []
    image_log = []

    with st.spinner("이미지 분석 중..."):
        for f in uploaded_files:
            f.seek(0)
            pil_img = Image.open(f).convert("RGB")

            if debug_mode:
                with st.expander(f"🔍 [{f.name}] OCR 원문"):
                    raw_boxes = ocr_boxes(reader, pil_img)
                    rows = group_rows(raw_boxes)
                    img_w = pil_img.size[0]
                    for r in rows:
                        side = "오른쪽" if r["x1"] > img_w * 0.50 else "왼쪽"
                        st.write(f"  [{side}] x1={r['x1']:.0f} cy={r['cy']:.0f} → `{r['text']}`")

            stocks, currency = extract_stocks(reader, pil_img, usd_krw_rate)
            image_log.append({
                "파일명": f.name,
                "통화": currency,
                "추출 종목 수": len(stocks),
            })
            all_stocks.extend(stocks)

    # ── 이미지별 요약
    st.subheader("📋 이미지별 추출 결과")
    st.dataframe(pd.DataFrame(image_log), use_container_width=True)

    if not all_stocks:
        st.error("종목을 추출하지 못했습니다. 디버그 모드를 켜서 OCR 원문을 확인해주세요.")
        st.divider()
        _show_history()
        return

    # ── 중복 종목 합산
    df = pd.DataFrame(all_stocks)
    result_df = (
        df.groupby("종목명", as_index=False)
        .agg(
            주수=("주수", "first"),
            통화=("통화", "first"),
            평가금액_원본=("평가금액_원본", "max"),
            손익_원본=("손익_원본", "sum"),
            평가금액_원화=("평가금액_원화", "max"),
            손익_원화=("손익_원화", "sum"),
            원금_원화=("원금_원화", "sum"),
        )
        .sort_values("평가금액_원화", ascending=False)
        .reset_index(drop=True)
    )
    result_df["비중(%)"] = (
        result_df["평가금액_원화"] / result_df["평가금액_원화"].sum() * 100
    ).round(2)

    # ── 요약 지표
    total_amount = int(result_df["평가금액_원화"].sum())
    total_profit = int(result_df["손익_원화"].sum())
    total_principal = int(result_df["원금_원화"].sum())
    total_count = len(result_df)
    profit_rate = total_profit / total_principal * 100 if total_principal else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("종목 수", total_count)
    c2.metric("총 평가금액", format_krw(total_amount))
    c3.metric("총 원금", format_krw(total_principal))
    c4.metric("총 손익", format_krw(total_profit))
    c5.metric("수익률", f"{profit_rate:.2f}%")

    # ── 기록 저장
    if st.button("💾 현재 기록 저장"):
        append_history(batch_name, total_amount, total_principal, total_profit, total_count)
        st.success("저장 완료!")
        st.rerun()

    # ── 종목 수정 테이블
    st.subheader("✏️ 종목명 수정 (필요시)")
    edit_df = result_df[["종목명", "주수", "통화", "평가금액_원화", "손익_원화", "원금_원화", "비중(%)"]].copy()
    edit_df["수정종목명"] = edit_df["종목명"]
    edit_df["평가금액_원화_표시"] = edit_df["평가금액_원화"].apply(format_krw)
    edit_df["손익_원화_표시"] = edit_df["손익_원화"].apply(format_krw)
    edit_df["원금_원화_표시"] = edit_df["원금_원화"].apply(format_krw)

    edited = st.data_editor(
        edit_df[["종목명", "수정종목명", "주수", "통화",
                 "평가금액_원화_표시", "손익_원화_표시", "원금_원화_표시", "비중(%)"]],
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "종목명": st.column_config.TextColumn("OCR 종목명", disabled=True),
            "수정종목명": st.column_config.TextColumn("수정종목명"),
            "주수": st.column_config.TextColumn("주수", disabled=True),
            "통화": st.column_config.TextColumn("통화", disabled=True),
            "평가금액_원화_표시": st.column_config.TextColumn("평가금액", disabled=True),
            "손익_원화_표시": st.column_config.TextColumn("손익", disabled=True),
            "원금_원화_표시": st.column_config.TextColumn("원금", disabled=True),
            "비중(%)": st.column_config.NumberColumn("비중(%)", disabled=True),
        },
    )

    # 수정된 종목명 반영
    final_df = result_df.copy()
    final_df["종목명"] = (
        edited["수정종목명"]
        .fillna(final_df["종목명"])
        .replace("", pd.NA)
        .fillna(final_df["종목명"])
    )

    # ── 파이차트
    st.subheader("📈 포트폴리오 비중")
    tab1, tab2 = st.tabs(["종목별 비중", "상위 15개"])

    with tab1:
        pie_chart(final_df, "종목명", "평가금액_원화", "종목별 평가금액 비중")

    with tab2:
        top = final_df.nlargest(15, "평가금액_원화")[["종목명", "평가금액_원화"]].copy()
        rest = final_df.iloc[15:]["평가금액_원화"].sum()
        if rest > 0:
            top = pd.concat(
                [top, pd.DataFrame([{"종목명": "기타", "평가금액_원화": rest}])],
                ignore_index=True,
            )
        pie_chart(top, "종목명", "평가금액_원화", "상위 15종목 비중")

    # ── CSV 다운로드
    csv = final_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ CSV 다운로드", data=csv,
                       file_name="toss_portfolio.csv", mime="text/csv")

    st.divider()
    _show_history()


# ─────────────────────────────────────────────
# 시간별 기록
# ─────────────────────────────────────────────
def _show_history():
    st.subheader("🕐 시간별 기록")
    history_df = load_history()

    if history_df.empty:
        st.info("저장된 기록이 없습니다.")
        return

    display = history_df.copy()
    for col in ["total_amount", "estimated_principal", "estimated_profit"]:
        display[col] = display[col].apply(format_krw)
    st.dataframe(display, use_container_width=True)

    # 삭제
    options = history_df.apply(
        lambda r: f'{int(r["id"])} | {r["timestamp"]} | {r["batch_name"]}', axis=1
    ).tolist()
    sel = st.selectbox("삭제할 기록", ["선택 안 함"] + options)
    if sel != "선택 안 함":
        row_id = int(sel.split("|")[0].strip())
        if st.button("🗑️ 삭제"):
            delete_history_row(row_id)
            st.success("삭제 완료")
            st.rerun()

    # CSV
    hist_csv = history_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 기록 CSV", data=hist_csv,
                       file_name="portfolio_history.csv", mime="text/csv")

    # 차트
    line_chart(history_df, ["total_amount"], "총 자산 변화")
    line_chart(history_df, ["estimated_principal", "estimated_profit"], "원금 / 수익금 변화")


if __name__ == "__main__":
    main()
