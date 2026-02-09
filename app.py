# app.py
import os
import re
import json
import datetime as dt
from typing import Optional, Tuple, Dict, Any, List

import requests
import pandas as pd
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")

st.title("📊 AI 습관 트래커")
st.caption("오늘의 체크인 → 7일 트렌드 → AI 코치 리포트까지 한 번에 🧠")

# -----------------------------
# Sidebar: API keys
# -----------------------------
with st.sidebar:
    st.header("🔑 API 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    owm_api_key = st.text_input("OpenWeatherMap API Key", type="password", placeholder="OWM Key...")
    st.divider()
    st.caption("Tip: 키는 브라우저에만 입력되고 세션 동안만 사용돼요. (배포 시엔 Secrets 권장)")

# -----------------------------
# Helpers: external APIs
# -----------------------------
def get_weather(city: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap 현재 날씨 (한국어, 섭씨).
    실패 시 None 반환. timeout=10
    """
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",
            "lang": "kr",
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()

        weather = (data.get("weather") or [{}])[0]
        main = data.get("main") or {}
        wind = data.get("wind") or {}
        sys_ = data.get("sys") or {}

        icon = weather.get("icon")
        icon_url = f"https://openweathermap.org/img/wn/{icon}@2x.png" if icon else None

        return {
            "city": data.get("name") or city,
            "country": sys_.get("country"),
            "desc": weather.get("description"),
            "temp": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "wind_speed": wind.get("speed"),
            "icon_url": icon_url,
        }
    except Exception:
        return None


def _breed_from_dog_url(url: str) -> Optional[str]:
    """
    Dog CEO 이미지 URL에서 품종 추출:
    예) https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg -> hound (afghan)
    """
    try:
        m = re.search(r"/breeds/([^/]+)/", url)
        if not m:
            return None
        raw = m.group(1)  # e.g. hound-afghan
        parts = raw.split("-")
        if len(parts) == 1:
            return parts[0]
        # Dog CEO는 보통 breed-subbreed 형태
        breed = parts[0]
        sub = " ".join(parts[1:])
        return f"{breed} ({sub})"
    except Exception:
        return None


def get_dog_image() -> Optional[Tuple[str, Optional[str]]]:
    """
    Dog CEO 랜덤 강아지 사진 URL과 품종 반환.
    실패 시 None 반환. timeout=10
    """
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "success":
            return None
        img_url = data.get("message")
        if not img_url:
            return None
        breed = _breed_from_dog_url(img_url)
        return img_url, breed
    except Exception:
        return None


# -----------------------------
# OpenAI: generate report
# -----------------------------
SYSTEM_PROMPTS = {
    "스파르타 코치": (
        "너는 '스파르타 코치'다. 말은 짧고 단호하게. 핑계는 잘라내고, 실행 가능한 지시만 준다. "
        "그래도 인신공격은 금지. 데이터 기반으로 딱딱 정리한다."
    ),
    "따뜻한 멘토": (
        "너는 '따뜻한 멘토'다. 공감은 하되 과장하지 말고, 현실적인 칭찬과 다음 행동을 부드럽게 제안한다. "
        "문장은 너무 길지 않게, 읽기 쉽게."
    ),
    "게임 마스터": (
        "너는 '게임 마스터'다. 사용자의 하루를 RPG 퀘스트 로그처럼 연출한다. "
        "진짜 게임 규칙을 만들 필요는 없고, 톤만 모험/레벨업 느낌으로. 유치하지 않게."
    ),
}

OUTPUT_FORMAT_RULES = """
반드시 아래 출력 형식(섹션 제목 포함)을 지켜서 한국어로 작성해라.

[컨디션 등급] S/A/B/C/D 중 하나 (한 줄)
[습관 분석] 체크된 습관/비어있는 습관을 근거로 3~5줄
[날씨 코멘트] 오늘 날씨를 반영해 1~2줄
[내일 미션] 구체적인 행동 3개(불릿)
[오늘의 한마디] 한 줄 (짧게, 기억에 남게)
""".strip()


def _extract_text_from_responses_api(resp: Any) -> str:
    """
    Responses API 응답에서 텍스트를 최대한 안전하게 추출.
    """
    # 1) 공식 속성 (있는 경우)
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # 2) dict-like
    try:
        if isinstance(resp, dict):
            # output_text
            t = resp.get("output_text")
            if isinstance(t, str) and t.strip():
                return t.strip()
            # output items 탐색
            out = resp.get("output") or []
            chunks = []
            for item in out:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "message":
                    content = item.get("content") or []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                            if isinstance(c.get("text"), str):
                                chunks.append(c["text"])
            if chunks:
                return "\n".join(chunks).strip()
    except Exception:
        pass

    # 3) 객체 탐색
    try:
        out = getattr(resp, "output", None)
        if out:
            chunks = []
            for item in out:
                itype = getattr(item, "type", None)
                if itype == "message":
                    content = getattr(item, "content", None) or []
                    for c in content:
                        ctype = getattr(c, "type", None)
                        if ctype in ("output_text", "text"):
                            t = getattr(c, "text", None)
                            if isinstance(t, str) and t.strip():
                                chunks.append(t)
            if chunks:
                return "\n".join(chunks).strip()
    except Exception:
        pass

    return ""


def generate_report(
    openai_key: str,
    coach_style: str,
    habits: Dict[str, bool],
    mood: int,
    weather: Optional[Dict[str, Any]],
    dog_breed: Optional[str],
) -> Optional[str]:
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달.
    모델: gpt-5-mini
    실패 시 None.
    """
    if not openai_key:
        return None

    checked = [k for k, v in habits.items() if v]
    unchecked = [k for k, v in habits.items() if not v]

    weather_text = "날씨 정보 없음"
    if weather:
        weather_text = (
            f"{weather.get('city')} / {weather.get('desc')} / "
            f"{weather.get('temp')}°C(체감 {weather.get('feels_like')}°C) / 습도 {weather.get('humidity')}%"
        )

    dog_text = dog_breed or "알 수 없음"

    user_payload = {
        "date": str(dt.date.today()),
        "mood_1_to_10": mood,
        "checked_habits": checked,
        "unchecked_habits": unchecked,
        "weather": weather_text,
        "dog_breed": dog_text,
        "instruction": OUTPUT_FORMAT_RULES,
    }

    system = SYSTEM_PROMPTS.get(coach_style, SYSTEM_PROMPTS["따뜻한 멘토"])

    try:
        # OpenAI Python SDK (Responses API)
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=openai_key)

        resp = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        "아래 데이터를 보고 'AI 코치 리포트'를 작성해줘.\n"
                        "데이터(JSON):\n"
                        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
                        "형식은 반드시 지켜."
                    ),
                },
            ],
            text={"verbosity": "medium"},
        )

        text = _extract_text_from_responses_api(resp)
        return text if text else None

    except Exception:
        return None


# -----------------------------
# Session state: history
# -----------------------------
def _init_demo_history() -> List[Dict[str, Any]]:
    """
    데모용 6일 샘플 데이터 (오늘 제외).
    """
    today = dt.date.today()
    demo = []
    # 최근 6일: today-6 ... today-1
    samples = [
        (3, 6),  # (checked_count, mood)
        (4, 7),
        (2, 5),
        (5, 8),
        (3, 6),
        (4, 7),
    ]
    for i, (cc, md) in enumerate(samples, start=6):
        day = today - dt.timedelta(days=i)
        rate = round((cc / 5) * 100, 0)
        demo.append({"date": str(day), "checked": cc, "rate": rate, "mood": md})
    return demo


if "history" not in st.session_state:
    st.session_state.history = _init_demo_history()

# -----------------------------
# Check-in UI
# -----------------------------
HABITS = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏋️", "운동하기"),
    ("😴", "수면"),
]

CITIES = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Ulsan",
    "Suwon",
    "Jeju",
    "Sejong",
]

coach_col, city_col = st.columns([1, 1])
with city_col:
    city = st.selectbox("🌍 도시 선택", CITIES, index=0)
with coach_col:
    coach_style = st.radio("🎭 코치 스타일", ["스파르타 코치", "따뜻한 멘토", "게임 마스터"], horizontal=True)

st.subheader("✅ 오늘 체크인")

c1, c2 = st.columns(2)

habit_state: Dict[str, bool] = {}

with c1:
    for emoji, name in HABITS[:3]:
        habit_state[name] = st.checkbox(f"{emoji} {name}", value=False, key=f"habit_{name}")

with c2:
    for emoji, name in HABITS[3:]:
        habit_state[name] = st.checkbox(f"{emoji} {name}", value=False, key=f"habit_{name}")

mood = st.slider("🙂 오늘 기분(1~10)", min_value=1, max_value=10, value=6, step=1)

# -----------------------------
# Compute today metrics + store in session_state
# -----------------------------
checked_count = sum(1 for v in habit_state.values() if v)
achievement = round((checked_count / 5) * 100, 0)

today_str = str(dt.date.today())
today_row = {"date": today_str, "checked": checked_count, "rate": achievement, "mood": mood}

# history에 오늘 항목을 "항상 최신"으로 1개 유지
history: List[Dict[str, Any]] = st.session_state.history
history = [r for r in history if r.get("date") != today_str]
history.append(today_row)
history = sorted(history, key=lambda x: x["date"])
st.session_state.history = history

# -----------------------------
# Metrics + chart
# -----------------------------
m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{int(achievement)}%")
m2.metric("달성 습관", f"{checked_count}/5")
m3.metric("기분", f"{mood}/10")

st.subheader("📈 7일 달성률 바 차트")

df = pd.DataFrame(st.session_state.history).tail(7)
if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    chart_df = df.set_index("date")[["rate"]]
    st.bar_chart(chart_df)
else:
    st.info("아직 데이터가 없어요. 오늘 체크인을 해보자 ✍️")

# -----------------------------
# Generate report button + results
# -----------------------------
st.subheader("🧾 AI 코치 리포트")

btn = st.button("컨디션 리포트 생성", type="primary")

weather_data = None
dog_data = None
report_text = None

if btn:
    with st.spinner("날씨와 강아지를 데려오는 중... 🐾"):
        weather_data = get_weather(city, owm_api_key)
        dog_data = get_dog_image()

    dog_url, dog_breed = (None, None)
    if dog_data:
        dog_url, dog_breed = dog_data

    with st.spinner("AI 코치가 리포트를 쓰는 중... ✍️"):
        report_text = generate_report(
            openai_key=openai_api_key,
            coach_style=coach_style,
            habits=habit_state,
            mood=mood,
            weather=weather_data,
            dog_breed=dog_breed,
        )

    # Display: weather + dog cards
    wcol, dcol = st.columns(2)

    with wcol:
        st.markdown("#### 🌦️ 오늘의 날씨")
        if weather_data:
            top = st.columns([3, 2])
            with top[0]:
                st.write(f"**도시:** {weather_data.get('city')}")
                st.write(f"**상태:** {weather_data.get('desc')}")
                st.write(f"**기온:** {weather_data.get('temp')}°C (체감 {weather_data.get('feels_like')}°C)")
                st.write(f"**습도:** {weather_data.get('humidity')}%")
                if weather_data.get("wind_speed") is not None:
                    st.write(f"**바람:** {weather_data.get('wind_speed')} m/s")
            with top[1]:
                if weather_data.get("icon_url"):
                    st.image(weather_data["icon_url"], caption="OpenWeatherMap", use_container_width=True)
        else:
            st.warning("날씨를 불러오지 못했어요. (API Key/도시/네트워크 확인)")

    with dcol:
        st.markdown("#### 🐶 오늘의 강아지")
        if dog_url:
            st.image(dog_url, use_container_width=True, caption=f"품종: {dog_breed or '알 수 없음'}")
        else:
            st.warning("강아지 이미지를 불러오지 못했어요. (네트워크 확인)")

    st.markdown("#### 🧠 AI 리포트")
    if report_text:
        st.markdown(report_text)
    else:
        if not openai_api_key:
            st.error("OpenAI API Key를 사이드바에 입력해줘!")
        else:
            st.error("리포트를 생성하지 못했어요. (키/요금/네트워크/모델 접근 권한 확인)")

    # Share text
    st.markdown("#### 📋 공유용 텍스트")
    share_payload = {
        "date": today_str,
        "city": city,
        "coach_style": coach_style,
        "achievement": f"{int(achievement)}%",
        "checked_habits": [k for k, v in habit_state.items() if v],
        "mood": f"{mood}/10",
        "weather": weather_data if weather_data else None,
        "dog_breed": dog_breed,
        "report": report_text,
    }
    st.code(json.dumps(share_payload, ensure_ascii=False, indent=2), language="json")

# -----------------------------
# API 안내
# -----------------------------
with st.expander("🔎 API 안내 / 키 발급 가이드"):
    st.markdown(
        """
- **OpenAI API Key**
  - OpenAI 대시보드에서 발급한 키를 입력해요.
  - 배포(예: Streamlit Cloud)에서는 **Secrets**에 저장하는 걸 권장해요.

- **OpenWeatherMap API Key**
  - OpenWeatherMap에서 발급한 키를 입력해요.
  - 본 앱은 `현재 날씨(Current Weather)`를 `섭씨(units=metric)` + `한국어(lang=kr)`로 요청해요.

- **Dog CEO**
  - 키 없이 무료로 랜덤 강아지 이미지를 가져와요. 네트워크가 불안하면 실패할 수 있어요.

문제가 생기면 체크:
1) API Key 오타/공백 여부  
2) 네트워크 연결  
3) 배포 환경에서 Secrets 설정 여부
        """.strip()
    )
