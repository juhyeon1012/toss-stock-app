import os
import re
import base64
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
from rapidfuzz import process, fuzz
import easyocr


st.set_page_config(page_title="토스 포트폴리오 자동 분석", layout="wide")

FONT_PATH = "NanumGothic-Regular.ttf"
HISTORY_FILE = "portfolio_history.csv"


# =========================
# 한글 폰트
# =========================
try:
    font_prop = fm.FontProperties(fname=FONT_PATH)
    matplotlib.rcParams["font.family"] = font_prop.get_name()
except Exception:
    font_prop = None

matplotlib.rcParams["axes.unicode_minus"] = False


# =========================
# 로드
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
# 기록 파일
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
# 문자열 / 포맷
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
    text = text.replace("(소수)", "")
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", "", text)
    return text


def format_currency_krw(x) -> str:
    try:
        return f"{int(round(float(x))):,}원"
    except Exception:
        return str(x)


def format_currency_usd(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


# =========================
# OCR 기초
# =========================
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
        if conf < 0.15:
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


def detect_currency_mode(boxes: List[Dict]) -> str:
    texts = " ".join([normalize_text(b["text"]) for b in boxes])

    usd_score = 0
    krw_score = 0

    if "$" in texts:
        usd_score += 5
    if re.search(r"\$\d", texts):
        usd_score += 5
    if "원화로 보기" in texts:
        usd_score += 3
    if "해외" in texts:
        usd_score += 1

    if "원" in texts:
        krw_score += 2
    if re.search(r"\d[\d,]*원", texts):
        krw_score += 5
    if "국내" in texts:
        krw_score += 1

    return "USD" if usd_score > krw_score else "KRW"


def is_share_text(text: str) -> bool:
    text = normalize_text(text)
    return bool(re.search(r"\d+(\.\d+)?주", text))


def is_profit_text(text: str, currency_mode: str) -> bool:
    text = normalize_text(text)

    if currency_mode == "USD":
        return bool(re.search(r"[+-]\$[\d,]+(\.\d+)?", text))
    return bool(re.search(r"[+-]\d[\d,]*원", text))


def parse_profit(text: str, currency_mode: str) -> Optional[float]:
    text = normalize_text(text)

    if currency_mode == "USD":
        m = re.search(r"([+-])\$(\d[\d,]*\.?\d*)", text)
        if not m:
            return None
        sign = -1 if m.group(1) == "-" else 1
        return sign * float(m.group(2).replace(",", ""))

    m = re.search(r"([+-])(\d[\d,]*)원", text)
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    return sign * int(m.group(2).replace(",", ""))


def is_amount_text(text: str, currency_mode: str) -> bool:
    text = normalize_text(text)

    if currency_mode == "USD":
        return bool(re.fullmatch(r"\$\d[\d,]*\.?\d*", text))
    return bool(re.fullmatch(r"\d[\d,]*원", text))


def parse_amount(text: str, currency_mode: str) -> Optional[float]:
    text = normalize_text(text)

    if currency_mode == "USD":
        m = re.search(r"\$(\d[\d,]*\.?\d*)", text)
        if not m:
            return None
        return float(m.group(1).replace(",", ""))

    m = re.search(r"(\d[\d,]*)원", text)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


# =========================
# 규칙 / 별칭
# =========================
def build_alias_dict() -> Dict[str, str]:
    return {
        "kodex미국나스닥100": "KODEX 미국나스닥100",
        "kodex미국s&p500": "KODEX 미국S&P500",
        "sol미국배당다우존스2호": "SOL 미국배당다우존스2호",
        "sol미국배당다우존스": "SOL 미국배당다우존스",
        "ace미국30년국채액티브": "ACE 미국30년국채액티브",
        "tiger반도체top10": "TIGER 반도체TOP10",
        "tiger미국테크top10indxx": "TIGER 미국테크TOP10 INDXX",
        "acekrx금현물": "ACE KRX금현물",
        "kodex미국ai전력핵심인프라": "KODEX 미국AI전력핵심인프라",
        "hanaro글로벌생성형ai액티브": "HANARO 글로벌생성형AI액티브",
        "tiger코리아top10": "TIGER 코리아TOP10",
        "tiger200": "TIGER 200",
        "kodex미국반도체": "KODEX 미국반도체",
        "koact코스닥액티브": "KoAct 코스닥액티브",
        "ace고배당주": "ACE 고배당주",
        "rise글로벌리얼티인컴": "RISE 글로벌리얼티인컴",
        "sol글로벌ai반도체탑픽액티브": "SOL 글로벌AI반도체탑픽액티브",
        "1q샤오미밸류체인액티브": "1Q 샤오미밸류체인액티브",
        "kodex차이나항셍테크": "KODEX 차이나항셍테크",
        "vanguards&p500": "VANGUARD S&P 500",
        "roundhillmagnificentseven": "ROUNDHILL MAGNIFICENT SEVEN",
        "제타글로벌홀딩스": "제타 글로벌 홀딩스",
        "cme그룹": "CME 그룹",
        "웨이스트매니지먼트": "웨이스트 매니지먼트",
        "웨이스트매니지먼트소수": "웨이스트 매니지먼트",
        "리커전파머슈티컬스": "리커전 파머슈티컬스",
        "usa레어어스": "USA 레어 어스",
        "써클인터넷그룹": "써클 인터넷 그룹",
        "유아이패스": "유아이패스",
        "아마존닷컴": "아마존닷컴",
        "아마존닷컴소수": "아마존닷컴",
        "버크셔해서웨이b": "버크셔 해서웨이 B",
        "마이크론테크놀로지": "마이크론 테크놀로지",
        "뱅크오브아메리카": "뱅크오브아메리카",
        "아메리칸익스프레스": "아메리칸 익스프레스",
        "팔란티어테크": "팔란티어 테크",
    }


def prepare_name_lookup(rules_df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    mapping = {}
    official_names = []

    for name in rules_df["종목명"].dropna().tolist():
        name = normalize_text(name)
        official_names.append(name)
        mapping[canonical_key(name)] = name

    alias_dict = build_alias_dict()
    for k, v in alias_dict.items():
        mapping[canonical_key(k)] = v

    return mapping, official_names


def fuzzy_match_name(
    raw_name: str,
    key_to_official: Dict[str, str],
    official_names: List[str],
    score_cutoff: int = 45,
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


# =========================
# 행(row) 기반 추출
# =========================
def group_boxes_into_rows(boxes: List[Dict]) -> List[Dict]:
    if not boxes:
        return []

    boxes_sorted = sorted(boxes, key=lambda x: x["cy"])
    rows = []

    for b in boxes_sorted:
        placed = False
        for row in rows:
            if abs(b["cy"] - row["cy"]) <= max(12, row["avg_h"] * 0.8):
                row["boxes"].append(b)
                row["cy_values"].append(b["cy"])
                row["avg_h_values"].append(b["h"])
                row["cy"] = float(np.mean(row["cy_values"]))
                row["avg_h"] = float(np.mean(row["avg_h_values"]))
                placed = True
                break

        if not placed:
            rows.append(
                {
                    "boxes": [b],
                    "cy_values": [b["cy"]],
                    "avg_h_values": [b["h"]],
                    "cy": b["cy"],
                    "avg_h": b["h"],
                }
            )

    final_rows = []
    for row in rows:
        row_boxes = sorted(row["boxes"], key=lambda x: x["x1"])
        text = " ".join([rb["text"] for rb in row_boxes])
        final_rows.append(
            {
                "boxes": row_boxes,
                "text": normalize_text(text),
                "cy": row["cy"],
                "x1": min(rb["x1"] for rb in row_boxes),
                "x2": max(rb["x2"] for rb in row_boxes),
                "avg_h": row["avg_h"],
            }
        )

    return sorted(final_rows, key=lambda x: x["cy"])


def build_name_from_left_rows(rows: List[Dict], idx: int, currency_mode: str) -> str:
    current = rows[idx]["text"]
    prev_text = rows[idx - 1]["text"] if idx - 1 >= 0 else ""
    next_text = rows[idx + 1]["text"] if idx + 1 < len(rows) else ""

    parts = []

    if prev_text and not is_share_text(prev_text) and not is_profit_text(prev_text, currency_mode):
        if not is_amount_text(prev_text, currency_mode):
            if abs(rows[idx]["cy"] - rows[idx - 1]["cy"]) < rows[idx]["avg_h"] * 2.5:
                parts.append(prev_text)

    parts.append(current)

    if next_text and not is_share_text(next_text) and not is_profit_text(next_text, currency_mode):
        if not is_amount_text(next_text, currency_mode):
            if abs(rows[idx + 1]["cy"] - rows[idx]["cy"]) < rows[idx]["avg_h"] * 2.5:
                if len(next_text) > 1:
                    parts.append(next_text)

    combined = " ".join(parts)
    combined = normalize_text(combined)
    combined = combined.replace("내 투자", "").replace("신규 투자", "").strip()
    return combined


def extract_stocks_from_image(
    reader,
    pil_img: Image.Image,
    key_to_official: Dict[str, str],
    official_names: List[str],
    usd_krw_rate: float,
) -> Tuple[List[Dict], str]:
    boxes = ocr_boxes_from_image(reader, pil_img)
    currency_mode = detect_currency_mode(boxes)
    rows = group_boxes_into_rows(boxes)

    if not rows:
        return [], currency_mode

    img_width = pil_img.size[0]
    right_threshold = img_width * 0.58

    candidates = []

    for i, row in enumerate(rows):
        text = row["text"]

        if row["x1"] < right_threshold:
            continue

        amount = parse_amount(text, currency_mode)
        if amount is None:
            continue

        nearest_idx = None
        nearest_dist = 1e9

        for j, lr in enumerate(rows):
            if lr["x2"] >= right_threshold:
                continue
            if is_share_text(lr["text"]):
                continue
            if is_profit_text(lr["text"], currency_mode):
                continue
            if parse_amount(lr["text"], currency_mode) is not None:
                continue

            dist = abs(lr["cy"] - row["cy"])
            if dist < nearest_dist and dist < max(40, row["avg_h"] * 4):
                nearest_dist = dist
                nearest_idx = j

        if nearest_idx is None:
            continue

        raw_name = build_name_from_left_rows(rows, nearest_idx, currency_mode)
        official_name = fuzzy_match_name(
            raw_name,
            key_to_official,
            official_names,
            score_cutoff=45,
        )

        if official_name is None:
            official_name = raw_name
            matched_status = "미분류"
        else:
            matched_status = "자동매칭"

        profit_value = 0.0
        for k in range(i + 1, min(i + 3, len(rows))):
            if rows[k]["x1"] >= right_threshold and is_profit_text(rows[k]["text"], currency_mode):
                pv = parse_profit(rows[k]["text"], currency_mode)
                if pv is not None:
                    profit_value = pv
                    break

        if currency_mode == "USD":
            amount_krw = round(amount * usd_krw_rate)
            profit_krw = round(profit_value * usd_krw_rate)
            principal_krw = round((amount - profit_value) * usd_krw_rate)
        else:
            amount_krw = round(amount)
            profit_krw = round(profit_value)
            principal_krw = round(amount - profit_value)

        candidates.append(
            {
                "raw_name": raw_name,
                "종목명": official_name,
                "매칭상태": matched_status,
                "통화": currency_mode,
                "평가금액_원본": amount,
                "손익_원본": profit_value,
                "평가금액_원화": amount_krw,
                "손익_원화": profit_krw,
                "원금_원화": principal_krw,
            }
        )

    return candidates, currency_mode


# =========================
# 표시
# =========================
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
            "linewidth": 1,
        },
    }

    if font_prop is not None:
        pie_kwargs["textprops"] = {
            "fontproperties": font_prop,
            "fontsize": 11,
        }

    ax.pie(**pie_kwargs)

    if font_prop is not None:
        ax.set_title(title, fontproperties=font_prop, fontsize=18)
    else:
        ax.set_title(title, fontsize=18)

    st.pyplot(fig)


def make_line_chart(history_df: pd.DataFrame, ycols: List[str], title: str):
    if history_df.empty:
        return

    plot_df = history_df.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], errors="coerce")
    plot_df = plot_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ycols:
        ax.plot(plot_df["timestamp"], plot_df[col], marker="o", label=col)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("시간")
    ax.set_ylabel("원")

    if font_prop is not None:
        ax.set_title(title, fontproperties=font_prop, fontsize=16)
    else:
        ax.set_title(title, fontsize=16)

    ax.legend()
    plt.xticks(rotation=30)
    st.pyplot(fig)


def render_preview_scroller(files):
    st.subheader("업로드 미리보기")

    preview_items = []
    for uploaded_file in files:
        pil_preview = Image.open(uploaded_file).convert("RGB")
        thumb = pil_preview.copy()
        thumb.thumbnail((260, 520))

        buffer = BytesIO()
        thumb.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        preview_items.append(
            f"""
            <div style="
                min-width: 260px;
                max-width: 260px;
                background: #111827;
                border-radius: 14px;
                padding: 10px;
                margin-right: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.25);
                flex-shrink: 0;
            ">
                <img src="data:image/jpeg;base64,{img_base64}"
                     style="width: 100%; border-radius: 10px; display:block;" />
                <div style="
                    color: white;
                    font-size: 13px;
                    margin-top: 8px;
                    word-break: break-all;
                    text-align: center;
                ">
                    {uploaded_file.name}
                </div>
            </div>
            """
        )

    preview_html = f"""
    <div style="
        display: flex;
        overflow-x: auto;
        gap: 4px;
        padding-bottom: 8px;
        white-space: nowrap;
    ">
        {''.join(preview_items)}
    </div>
    """
    st.markdown(preview_html, unsafe_allow_html=True)


# =========================
# 메인
# =========================
def main():
    st.title("토스 포트폴리오 자동 분석")
    st.write("여러 장 캡처를 올리면 종목, 평가금액, 손익, 원금, 원화 환산까지 자동 정리합니다.")

    with st.sidebar:
        st.header("설정")
        usd_krw_rate = st.number_input(
            "달러 → 원 환율",
            min_value=1000.0,
            max_value=2000.0,
            value=1350.0,
            step=1.0,
        )
        st.caption("해외 화면이 달러 표시일 때 이 환율로 원화 환산합니다.")

    rules_df = load_rules()
    key_to_official, official_names = prepare_name_lookup(rules_df)

    with st.expander("rules.csv 확인", expanded=False):
        st.dataframe(rules_df, use_container_width=True)

    uploaded_files = st.file_uploader(
        "토스 캡처 이미지 여러 장 업로드",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    batch_name = st.text_input(
        "이번 기록 이름",
        value=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    if uploaded_files:
        render_preview_scroller(uploaded_files)

        try:
            with st.spinner("OCR 엔진 준비 중..."):
                reader = load_ocr_reader()
        except Exception as e:
            st.error(f"OCR 엔진 로드 중 오류: {e}")
            st.stop()

        all_candidates = []
        image_summaries = []

        with st.spinner("여러 이미지 OCR 분석 중..."):
            for uploaded_file in uploaded_files:
                pil_img = Image.open(uploaded_file).convert("RGB")

                extracted, currency_mode = extract_stocks_from_image(
                    reader=reader,
                    pil_img=pil_img,
                    key_to_official=key_to_official,
                    official_names=official_names,
                    usd_krw_rate=usd_krw_rate,
                )

                image_summaries.append(
                    {
                        "파일명": uploaded_file.name,
                        "통화": currency_mode,
                        "추출종목수": len(extracted),
                    }
                )
                all_candidates.extend(extracted)

        st.subheader("이미지별 추출 상태")
        summary_df = pd.DataFrame(image_summaries)
        st.dataframe(summary_df, use_container_width=True)

        if not summary_df.empty:
            usd_count = int((summary_df["통화"] == "USD").sum())
            krw_count = int((summary_df["통화"] == "KRW").sum())
            c1, c2 = st.columns(2)
            c1.metric("원화 화면 수", krw_count)
            c2.metric("달러 화면 수", usd_count)

        if all_candidates:
            extracted_df = pd.DataFrame(all_candidates)

            if "매칭상태" not in extracted_df.columns:
                extracted_df["매칭상태"] = "자동매칭"

            result_df = (
                extracted_df.groupby("종목명", as_index=False)
                .agg(
                    {
                        "raw_name": "first",
                        "매칭상태": "first",
                        "통화": "first",
                        "평가금액_원본": "max",
                        "손익_원본": "sum",
                        "평가금액_원화": "max",
                        "손익_원화": "sum",
                        "원금_원화": "sum",
                    }
                )
            )

            result_df = result_df.merge(rules_df, on="종목명", how="left")
            result_df["자산유형"] = result_df["자산유형"].fillna("미분류")
            result_df["지역"] = result_df["지역"].fillna("미분류")
            result_df["최종분류"] = result_df["최종분류"].fillna("미분류")

            result_df = result_df.sort_values("평가금액_원화", ascending=False).reset_index(drop=True)
            result_df["비중(%)"] = (
                result_df["평가금액_원화"] / result_df["평가금액_원화"].sum() * 100
            ).round(2)

            st.subheader("종목명 확인 / 수정")
            st.caption("자동 매칭이 애매한 종목은 여기서 직접 수정할 수 있어요. 비워두면 현재 종목명을 유지합니다.")

            editable_df = result_df.copy()
            editable_df["수정종목명"] = editable_df["종목명"]

            editor_show = editable_df[
                [
                    "raw_name",
                    "종목명",
                    "수정종목명",
                    "매칭상태",
                    "통화",
                    "평가금액_원화",
                    "최종분류",
                ]
            ].copy()
            editor_show["평가금액_원화"] = editor_show["평가금액_원화"].apply(format_currency_krw)

            edited_df = st.data_editor(
                editor_show,
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "raw_name": st.column_config.TextColumn("OCR 원문"),
                    "종목명": st.column_config.TextColumn("현재 종목명", disabled=True),
                    "수정종목명": st.column_config.TextColumn("수정종목명"),
                    "매칭상태": st.column_config.TextColumn("매칭상태", disabled=True),
                    "통화": st.column_config.TextColumn("통화", disabled=True),
                    "평가금액_원화": st.column_config.TextColumn("평가금액(원화)", disabled=True),
                    "최종분류": st.column_config.TextColumn("현재분류", disabled=True),
                },
            )

            apply_df = result_df.copy()
            apply_df["종목명"] = edited_df["수정종목명"].fillna(apply_df["종목명"]).replace("", pd.NA).fillna(apply_df["종목명"])

            apply_df = (
                apply_df.groupby("종목명", as_index=False)
                .agg(
                    {
                        "raw_name": "first",
                        "매칭상태": "first",
                        "통화": "first",
                        "평가금액_원본": "max",
                        "손익_원본": "sum",
                        "평가금액_원화": "max",
                        "손익_원화": "sum",
                        "원금_원화": "sum",
                    }
                )
            )

            apply_df = apply_df.merge(rules_df, on="종목명", how="left")
            apply_df["자산유형"] = apply_df["자산유형"].fillna("미분류")
            apply_df["지역"] = apply_df["지역"].fillna("미분류")
            apply_df["최종분류"] = apply_df["최종분류"].fillna("미분류")

            apply_df = apply_df.sort_values("평가금액_원화", ascending=False).reset_index(drop=True)
            apply_df["비중(%)"] = (
                apply_df["평가금액_원화"] / apply_df["평가금액_원화"].sum() * 100
            ).round(2)

            total_amount = int(apply_df["평가금액_원화"].sum())
            total_profit = int(apply_df["손익_원화"].sum())
            total_principal = int(apply_df["원금_원화"].sum())
            total_count = int(apply_df["종목명"].nunique())
            unclassified_count = int((apply_df["최종분류"] == "미분류").sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("총 종목 수", total_count)
            c2.metric("총 평가금액", format_currency_krw(total_amount))
            c3.metric("총 손익", format_currency_krw(total_profit))
            c4.metric("미분류 수", unclassified_count)

            if st.button("현재 기록 저장"):
                append_history(
                    batch_name=batch_name,
                    total_amount=total_amount,
                    estimated_principal=total_principal,
                    estimated_profit=total_profit,
                    stock_count=total_count,
                )
                st.success("기록이 저장되었습니다.")
                st.rerun()

            st.subheader("자동 정리 결과")
            show_df = apply_df.copy()

            def fmt_original(row):
                if row["통화"] == "USD":
                    return format_currency_usd(row["평가금액_원본"])
                return format_currency_krw(row["평가금액_원본"])

            def fmt_profit_original(row):
                if row["통화"] == "USD":
                    return format_currency_usd(row["손익_원본"])
                return format_currency_krw(row["손익_원본"])

            show_df["평가금액_원본표시"] = show_df.apply(fmt_original, axis=1)
            show_df["손익_원본표시"] = show_df.apply(fmt_profit_original, axis=1)
            show_df["평가금액_원화"] = show_df["평가금액_원화"].apply(format_currency_krw)
            show_df["손익_원화"] = show_df["손익_원화"].apply(format_currency_krw)
            show_df["원금_원화"] = show_df["원금_원화"].apply(format_currency_krw)

            st.dataframe(
                show_df[
                    [
                        "종목명",
                        "매칭상태",
                        "통화",
                        "평가금액_원본표시",
                        "손익_원본표시",
                        "평가금액_원화",
                        "손익_원화",
                        "원금_원화",
                        "자산유형",
                        "지역",
                        "최종분류",
                        "비중(%)",
                        "raw_name",
                    ]
                ],
                use_container_width=True,
            )

            result_csv = apply_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "결과 CSV 다운로드",
                data=result_csv,
                file_name="toss_portfolio_result.csv",
                mime="text/csv",
            )

            st.subheader("파이차트")
            tab1, tab2, tab3 = st.tabs(["종목별 비중", "최종분류 비중", "국내/해외 비중"])

            with tab1:
                pie_source = apply_df.copy()
                if len(pie_source) > 15:
                    top15 = pie_source.nlargest(15, "평가금액_원화")[["종목명", "평가금액_원화"]].copy()
                    others_sum = pie_source.iloc[15:]["평가금액_원화"].sum()
                    pie_df = pd.concat(
                        [top15, pd.DataFrame([{"종목명": "기타", "평가금액_원화": others_sum}])],
                        ignore_index=True,
                    )
                else:
                    pie_df = pie_source[["종목명", "평가금액_원화"]].copy()

                make_pie_chart(pie_df, "종목명", "평가금액_원화", "종목별 비중")

            with tab2:
                class_df = (
                    apply_df.groupby("최종분류", dropna=False)["평가금액_원화"]
                    .sum()
                    .reset_index()
                    .rename(columns={"최종분류": "분류"})
                )
                class_df["분류"] = class_df["분류"].fillna("미분류")
                make_pie_chart(class_df, "분류", "평가금액_원화", "최종분류 비중")

            with tab3:
                region_df = (
                    apply_df.groupby("지역", dropna=False)["평가금액_원화"]
                    .sum()
                    .reset_index()
                )
                region_df["지역"] = region_df["지역"].fillna("미분류")
                make_pie_chart(region_df, "지역", "평가금액_원화", "국내/해외 비중")

            st.subheader("분류별 합계")
            summary_df = (
                apply_df.groupby("최종분류", dropna=False)["평가금액_원화"]
                .sum()
                .reset_index()
                .sort_values("평가금액_원화", ascending=False)
            )
            summary_df["최종분류"] = summary_df["최종분류"].fillna("미분류")
            summary_df["평가금액_원화"] = summary_df["평가금액_원화"].apply(format_currency_krw)
            st.dataframe(summary_df, use_container_width=True)

            st.subheader("미분류 / 확인 필요")
            review_df = apply_df[
                (apply_df["최종분류"] == "미분류") | (apply_df["매칭상태"] != "자동매칭")
            ].copy()

            if review_df.empty:
                st.success("미분류 종목이 없습니다.")
            else:
                review_df["평가금액_원화"] = review_df["평가금액_원화"].apply(format_currency_krw)
                st.dataframe(
                    review_df[["종목명", "raw_name", "매칭상태", "통화", "평가금액_원화"]],
                    use_container_width=True,
                )

        else:
            st.warning("이미지에서 종목을 추출하지 못했습니다.")

    st.divider()
    st.subheader("시간별 기록 관리")
    history_df = load_history()

    if not history_df.empty:
        display_history = history_df.copy()
        for col in ["total_amount", "estimated_principal", "estimated_profit"]:
            display_history[col] = display_history[col].apply(format_currency_krw)

        st.dataframe(display_history, use_container_width=True)

        delete_options = history_df.apply(
            lambda row: f'{int(row["id"])} | {row["timestamp"]} | {row["batch_name"]}',
            axis=1,
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
            mime="text/csv",
        )

        st.subheader("시간별 총 자산 변화")
        make_line_chart(history_df, ["total_amount"], "시간별 총 자산 변화")

        st.subheader("시간별 원금 / 수익금 변화")
        make_line_chart(history_df, ["estimated_principal", "estimated_profit"], "시간별 원금 / 수익금 변화")

    else:
        st.info("아직 저장된 시간별 기록이 없습니다.")


if __name__ == "__main__":
    main()
