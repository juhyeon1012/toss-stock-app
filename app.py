import os
import re
import unicodedata
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from PIL import Image
from rapidfuzz import process, fuzz
import easyocr


st.set_page_config(page_title="토스 포트폴리오 자동 분석", layout="wide")

FONT_PATH = "NanumGothic-Regular.ttf"
HISTORY_FILE = "portfolio_history.csv"

# =========================
# 한글 폰트 설정
# =========================
try:
    font_prop = fm.FontProperties(fname=FONT_PATH)
    matplotlib.rcParams["font.family"] = font_prop.get_name()
except Exception:
    font_prop = None

matplotlib.rcParams["axes.unicode_minus"] = False


# =========================
# 데이터 로드
# =========================
@st.cache_data
def load_rules() -> pd.DataFrame:
    df = pd.read_csv("rules.csv")
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["ko", "en"], gpu=False)


# =========================
# 기록 파일 관련
# =========================
def empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "id",
            "timestamp",
            "batch_name",
            "total_amount",
            "estimated_principal",
            "estimated_profit",
            "stock_count",
        ]
    )


def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return empty_history_df()

    try:
        df = pd.read_csv(HISTORY_FILE)
        for col in [
            "id",
            "total_amount",
            "estimated_principal",
            "estimated_profit",
            "stock_count",
        ]:
            if col not in df.columns:
                df[col] = 0
        return df
    except Exception:
        return empty_history_df()


def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def append_history(
    batch_name: str,
    total_amount: int,
    estimated_principal: int,
    estimated_profit: int,
    stock_count: int,
):
    old_df = load_history()

    if old_df.empty:
        next_id = 1
    else:
        next_id = int(pd.to_numeric(old_df["id"], errors="coerce").max()) + 1

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame(
        [
            {
                "id": next_id,
                "timestamp": now_str,
                "batch_name": batch_name,
                "total_amount": int(total_amount),
                "estimated_principal": int(estimated_principal),
                "estimated_profit": int(estimated_profit),
                "stock_count": int(stock_count),
            }
        ]
    )

    new_df = pd.concat([old_df, new_row], ignore_index=True)
    save_history(new_df)


def delete_history_row(row_id: int):
    history_df = load_history()
    history_df["id"] = pd.to_numeric(history_df["id"], errors="coerce")
    history_df = history_df[history_df["id"] != row_id].copy()
    save_history(history_df)


# =========================
# OCR / 텍스트 유틸
# =========================
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical_key(text: str) -> str:
    text = normalize_text(text).lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", "", text)
    return text


def is_amount_text(text: str) -> bool:
    text = normalize_text(text)
    if "원" in text:
        return True
    if re.fullmatch(r"[\d,]+", text):
        return True
    return False


def parse_amount(text: str) -> Optional[int]:
    text = normalize_text(text)
    nums = re.findall(r"\d[\d,]*", text)
    if not nums:
        return None
    try:
        return int(nums[0].replace(",", ""))
    except Exception:
        return None


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img)
    if img.ndim == 3:
        gray = np.mean(img, axis=2).astype(np.uint8)
    else:
        gray = img.astype(np.uint8)
    return gray


def ocr_boxes_from_image(reader, pil_img: Image.Image) -> List[Dict]:
    gray = preprocess_image(pil_img)
    results = reader.readtext(gray, detail=1, paragraph=False)

    boxes = []
    for item in results:
        box, text, conf = item
        if conf < 0.2:
            continue

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        text = normalize_text(text)
        if not text:
            continue

        boxes.append(
            {
                "text": text,
                "conf": float(conf),
                "x1": float(x1),
                "x2": float(x2),
                "y1": float(y1),
                "y2": float(y2),
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
            }
        )
    return boxes


def prepare_name_lookup(rules_df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    mapping = {}
    official_names = []

    for name in rules_df["종목명"].dropna().tolist():
        name = normalize_text(name)
        official_names.append(name)
        mapping[canonical_key(name)] = name

    return mapping, official_names


def fuzzy_match_name(
    raw_name: str,
    key_to_official: Dict[str, str],
    official_names: List[str],
    score_cutoff: int = 60,
) -> Optional[str]:
    raw_name = normalize_text(raw_name)
    if not raw_name:
        return None

    key = canonical_key(raw_name)

    if key in key_to_official:
        return key_to_official[key]

    keys = list(key_to_official.keys())
    best = process.extractOne(
        key,
        keys,
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff,
    )
    if best is not None:
        return key_to_official[best[0]]

    best2 = process.extractOne(
        raw_name,
        official_names,
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff,
    )
    if best2 is not None:
        return best2[0]

    return None


def find_candidate_name_boxes(boxes: List[Dict]) -> List[Dict]:
    candidates = []
    ignore_words = {
        "보유", "총 자산", "주문 가능", "원", "주", "국내", "해외",
        "ETF", "개별주", "내 투자", "신규 투자"
    }

    for b in boxes:
        t = normalize_text(b["text"])
        if is_amount_text(t):
            continue
        if t in ignore_words:
            continue
        if len(t) <= 1:
            continue
        candidates.append(b)

    return candidates


def find_candidate_amount_boxes(boxes: List[Dict]) -> List[Dict]:
    candidates = []
    for b in boxes:
        t = normalize_text(b["text"])
        amt = parse_amount(t)
        if amt is None:
            continue
        if amt < 1000:
            continue
        candidates.append(b)
    return candidates


def pair_names_and_amounts(
    name_boxes: List[Dict],
    amount_boxes: List[Dict],
    key_to_official: Dict[str, str],
    official_names: List[str],
) -> List[Dict]:
    pairs = []

    for ab in amount_boxes:
        best_name = None
        best_score = -1

        for nb in name_boxes:
            if nb["cx"] >= ab["cx"]:
                continue

            y_diff = abs(nb["cy"] - ab["cy"])
            if y_diff > max(nb["h"], ab["h"]) * 1.3:
                continue

            x_gap = ab["x1"] - nb["x2"]
            if x_gap < -10:
                continue

            score = 1000 - (y_diff * 5 + max(x_gap, 0))
            if score > best_score:
                best_score = score
                best_name = nb

        if best_name is None:
            continue

        raw_name = normalize_text(best_name["text"])
        official_name = fuzzy_match_name(
            raw_name,
            key_to_official,
            official_names,
            score_cutoff=58,
        )
        amount = parse_amount(ab["text"])

        if official_name is None or amount is None:
            continue

        pairs.append(
            {
                "raw_name": raw_name,
                "종목명": official_name,
                "금액": amount,
                "name_y": best_name["cy"],
            }
        )

    if not pairs:
        return []

    temp_df = pd.DataFrame(pairs)
    temp_df = temp_df.sort_values(["종목명", "금액"], ascending=[True, False])
    temp_df = temp_df.drop_duplicates(subset=["종목명"], keep="first")
    temp_df = temp_df.sort_values("name_y").reset_index(drop=True)

    return temp_df.to_dict(orient="records")


def classify_with_rules(extracted_df: pd.DataFrame, rules_df: pd.DataFrame) -> pd.DataFrame:
    return extracted_df.merge(rules_df, on="종목명", how="left")


# =========================
# 표시용 함수
# =========================
def format_currency(x) -> str:
    try:
        return f"{int(float(x)):,}원"
    except Exception:
        return str(x)


def make_pie_chart(df, label_col, value_col, title):
    if df.empty:
        st.info(f"{title}: 표시할 데이터가 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    n = len(df)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n)]

    pie_kwargs = {
        "x": df[value_col],
        "labels": df[label_col],
        "autopct": "%1.1f%%",
        "colors": colors,
        "startangle": 90,
        "wedgeprops": {
            "edgecolor": "black",
            "linewidth": 1
        }
    }

    if font_prop is not None:
        pie_kwargs["textprops"] = {
            "fontproperties": font_prop,
            "fontsize": 11
        }

    ax.pie(**pie_kwargs)

    if font_prop is not None:
        ax.set_title(title, fontproperties=font_prop, fontsize=18)
    else:
        ax.set_title(title, fontsize=18)

    st.pyplot(fig)


def make_total_asset_line_chart(history_df: pd.DataFrame):
    if history_df.empty:
        st.info("저장된 시간별 기록이 없습니다.")
        return

    plot_df = history_df.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], errors="coerce")
    plot_df = plot_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df["timestamp"], plot_df["total_amount"], marker="o", label="총 자산")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("시간")
    ax.set_ylabel("원")

    if font_prop is not None:
        ax.set_title("시간별 총 자산 변화", fontproperties=font_prop, fontsize=16)
    else:
        ax.set_title("시간별 총 자산 변화", fontsize=16)

    ax.legend()
    plt.xticks(rotation=30)
    st.pyplot(fig)


def make_principal_profit_line_chart(history_df: pd.DataFrame):
    if history_df.empty:
        return

    plot_df = history_df.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], errors="coerce")
    plot_df = plot_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df["timestamp"], plot_df["estimated_principal"], marker="o", label="원금")
    ax.plot(plot_df["timestamp"], plot_df["estimated_profit"], marker="o", label="수익금")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("시간")
    ax.set_ylabel("원")

    if font_prop is not None:
        ax.set_title("시간별 원금 / 수익금 변화", fontproperties=font_prop, fontsize=16)
    else:
        ax.set_title("시간별 원금 / 수익금 변화", fontsize=16)

    ax.legend()
    plt.xticks(rotation=30)
    st.pyplot(fig)


# =========================
# 앱 UI
# =========================
st.title("토스 포트폴리오 자동 분석")
st.write("토스 캡처 이미지를 여러 장 올리면 종목명, 금액, 분류를 자동 정리하고 시간별 기록도 관리합니다.")

rules_df = load_rules()
reader = load_ocr_reader()
key_to_official, official_names = prepare_name_lookup(rules_df)

with st.expander("rules.csv 확인", expanded=False):
    st.dataframe(rules_df, use_container_width=True)

uploaded_files = st.file_uploader(
    "토스 캡처 이미지 여러 장 업로드",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

batch_name = st.text_input("이번 기록 이름", value=datetime.now().strftime("%Y-%m-%d %H:%M"))

if uploaded_files:
    all_pairs = []

    preview_count = min(len(uploaded_files), 3)
    st.subheader("업로드 미리보기")
    preview_cols = st.columns(preview_count)

    for i in range(preview_count):
        pil_preview = Image.open(uploaded_files[i]).convert("RGB")
        preview_cols[i].image(
            pil_preview,
            caption=uploaded_files[i].name,
            use_container_width=True
        )

    with st.spinner("여러 이미지 OCR 분석 중..."):
        for uploaded_file in uploaded_files:
            pil_img = Image.open(uploaded_file).convert("RGB")
            boxes = ocr_boxes_from_image(reader, pil_img)
            name_boxes = find_candidate_name_boxes(boxes)
            amount_boxes = find_candidate_amount_boxes(boxes)
            paired = pair_names_and_amounts(
                name_boxes,
                amount_boxes,
                key_to_official,
                official_names,
            )
            all_pairs.extend(paired)

    if all_pairs:
        extracted_df = pd.DataFrame(all_pairs)[["종목명", "금액", "raw_name"]]

        extracted_df = (
            extracted_df.groupby("종목명", as_index=False)
            .agg({
                "금액": "max",
                "raw_name": "first"
            })
        )

        result_df = classify_with_rules(extracted_df, rules_df)
        result_df = result_df.sort_values("금액", ascending=False).reset_index(drop=True)
        result_df["비중(%)"] = (result_df["금액"] / result_df["금액"].sum() * 100).round(2)

        total_amount = int(result_df["금액"].sum())
        total_count = int(result_df["종목명"].nunique())
        unclassified_count = int(result_df["최종분류"].isna().sum())

        st.subheader("원금 / 수익금 입력")
        st.caption("토스의 +손익금을 아직 자동 읽지 않으므로, 이번 기록의 총 수익금을 직접 넣으면 원금이 자동 계산됩니다.")

        col_input1, col_input2 = st.columns(2)

        with col_input1:
            estimated_profit_input = st.number_input(
                "이번 기록의 총 수익금(원)",
                min_value=-10**12,
                max_value=10**12,
                value=0,
                step=1000,
            )

        estimated_principal = total_amount - int(estimated_profit_input)

        with col_input2:
            st.metric("추정 원금", format_currency(estimated_principal))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 종목 수", total_count)
        c2.metric("총 금액", format_currency(total_amount))
        c3.metric("추정 수익금", format_currency(estimated_profit_input))
        c4.metric("미분류 수", unclassified_count)

        save_col1, save_col2 = st.columns([1, 2])
        with save_col1:
            if st.button("현재 기록 저장"):
                append_history(
                    batch_name=batch_name,
                    total_amount=total_amount,
                    estimated_principal=estimated_principal,
                    estimated_profit=int(estimated_profit_input),
                    stock_count=total_count,
                )
                st.success("기록이 저장되었습니다.")
                st.rerun()

        st.subheader("자동 정리 결과")
        show_df = result_df.copy()
        show_df["금액"] = show_df["금액"].apply(format_currency)

        st.dataframe(
            show_df[["종목명", "금액", "자산유형", "지역", "최종분류", "비중(%)", "raw_name"]],
            use_container_width=True
        )

        csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "결과 CSV 다운로드",
            data=csv_bytes,
            file_name="toss_portfolio_result.csv",
            mime="text/csv"
        )

        st.subheader("파이차트")
        tab1, tab2, tab3 = st.tabs(["종목별 비중", "최종분류 비중", "국내/해외 비중"])

        with tab1:
            top_df = result_df.copy()
            if len(top_df) > 15:
                top15 = top_df.nlargest(15, "금액")[["종목명", "금액"]].copy()
                others_sum = top_df.iloc[15:]["금액"].sum()
                pie_df = pd.concat(
                    [top15, pd.DataFrame([{"종목명": "기타", "금액": others_sum}])],
                    ignore_index=True
                )
            else:
                pie_df = top_df[["종목명", "금액"]].copy()

            make_pie_chart(pie_df, "종목명", "금액", "종목별 비중")

        with tab2:
            class_df = (
                result_df.groupby("최종분류", dropna=False)["금액"]
                .sum()
                .reset_index()
                .rename(columns={"최종분류": "분류"})
            )
            class_df["분류"] = class_df["분류"].fillna("미분류")
            make_pie_chart(class_df, "분류", "금액", "최종분류 비중")

        with tab3:
            region_df = (
                result_df.groupby("지역", dropna=False)["금액"]
                .sum()
                .reset_index()
            )
            region_df["지역"] = region_df["지역"].fillna("미분류")
            make_pie_chart(region_df, "지역", "금액", "국내/해외 비중")

        st.subheader("분류별 합계")
        summary_df = (
            result_df.groupby(["최종분류"], dropna=False)["금액"]
            .sum()
            .reset_index()
            .sort_values("금액", ascending=False)
        )
        summary_df["최종분류"] = summary_df["최종분류"].fillna("미분류")
        summary_df["금액"] = summary_df["금액"].apply(format_currency)
        st.dataframe(summary_df, use_container_width=True)

    else:
        st.warning("이미지들에서 종목과 금액을 자동으로 매칭하지 못했습니다. 더 선명한 캡처로 다시 시도해 보세요.")

st.divider()

st.subheader("시간별 기록 관리")
history_df = load_history()

if not history_df.empty:
    display_history = history_df.copy()

    for col in ["total_amount", "estimated_principal", "estimated_profit"]:
        display_history[col] = display_history[col].apply(format_currency)

    st.dataframe(display_history, use_container_width=True)

    delete_options = history_df.apply(
        lambda row: f'{int(row["id"])} | {row["timestamp"]} | {row["batch_name"]}',
        axis=1
    ).tolist()

    selected_delete = st.selectbox("삭제할 기록 선택", ["선택 안 함"] + delete_options)

    if selected_delete != "선택 안 함":
        row_id = int(selected_delete.split("|")[0].strip())
        if st.button("선택 기록 삭제"):
            delete_history_row(row_id)
            st.success("기록이 삭제되었습니다.")
            st.rerun()

    history_csv = history_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "시간별 기록 CSV 다운로드",
        data=history_csv,
        file_name="portfolio_history.csv",
        mime="text/csv"
    )

    st.subheader("시간별 총 자산 변화")
    make_total_asset_line_chart(history_df)

    st.subheader("시간별 원금 / 수익금 변화")
    make_principal_profit_line_chart(history_df)

else:
    st.info("아직 저장된 시간별 기록이 없습니다.")
