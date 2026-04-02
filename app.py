import re
import unicodedata
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

# =========================
# 한글 폰트 설정
# =========================
FONT_PATH = "NanumGothic-Regular.ttf"

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
# 유틸 함수
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
        matched_key = best[0]
        return key_to_official[matched_key]

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
    merged = extracted_df.merge(rules_df, on="종목명", how="left")
    return merged


def format_currency(x: int) -> str:
    return f"{int(x):,}원"


def make_pie_chart(df: pd.DataFrame, label_col: str, value_col: str, title: str):
    if df.empty:
        st.info(f"{title}: 표시할 데이터가 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    pie_kwargs = {
        "x": df[value_col],
        "labels": df[label_col],
        "autopct": "%1.1f%%",
    }

    if font_prop is not None:
        pie_kwargs["textprops"] = {"fontproperties": font_prop, "fontsize": 11}

    ax.pie(**pie_kwargs)

    if font_prop is not None:
        ax.set_title(title, fontproperties=font_prop, fontsize=18)
    else:
        ax.set_title(title, fontsize=18)

    st.pyplot(fig)


# =========================
# UI
# =========================
st.title("토스 포트폴리오 자동 분석")
st.write("토스 보유 종목 캡처 이미지를 올리면 종목명, 금액, 분류를 자동 정리합니다.")

rules_df = load_rules()
reader = load_ocr_reader()
key_to_official, official_names = prepare_name_lookup(rules_df)

with st.expander("rules.csv 확인", expanded=False):
    st.dataframe(rules_df, use_container_width=True)

uploaded_file = st.file_uploader(
    "토스 캡처 이미지 업로드",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.subheader("업로드한 이미지")
        st.image(pil_img, use_container_width=True)

    with col2:
        with st.spinner("OCR 분석 중..."):
            boxes = ocr_boxes_from_image(reader, pil_img)
            name_boxes = find_candidate_name_boxes(boxes)
            amount_boxes = find_candidate_amount_boxes(boxes)
            paired = pair_names_and_amounts(
                name_boxes,
                amount_boxes,
                key_to_official,
                official_names,
            )

        st.subheader("OCR 상태")
        st.write(f"인식된 텍스트 박스 수: {len(boxes)}")
        st.write(f"후보 종목명 수: {len(name_boxes)}")
        st.write(f"후보 금액 수: {len(amount_boxes)}")
        st.write(f"최종 매칭 수: {len(paired)}")

    if paired:
        extracted_df = pd.DataFrame(paired)[["종목명", "금액", "raw_name"]]
        result_df = classify_with_rules(extracted_df, rules_df)

        result_df = result_df.sort_values("금액", ascending=False).reset_index(drop=True)
        result_df["비중(%)"] = (result_df["금액"] / result_df["금액"].sum() * 100).round(2)

        total_amount = int(result_df["금액"].sum())
        total_count = result_df["종목명"].nunique()
        unclassified_count = int(result_df["최종분류"].isna().sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("총 종목 수", total_count)
        c2.metric("총 금액", format_currency(total_amount))
        c3.metric("미분류 수", unclassified_count)

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
            result_df.groupby("최종분류", dropna=False)["금액"]
            .sum()
            .reset_index()
            .sort_values("금액", ascending=False)
        )
        summary_df["최종분류"] = summary_df["최종분류"].fillna("미분류")
        summary_df["금액"] = summary_df["금액"].apply(format_currency)
        st.dataframe(summary_df, use_container_width=True)

    else:
        st.warning("종목과 금액을 자동으로 매칭하지 못했습니다. 더 선명한 캡처로 다시 시도해 보세요.")
