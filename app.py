# app.py
import os
import json
import sqlite3
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(page_title="AI 농구 코칭 대시보드", page_icon="🏀", layout="wide")
st.title("🏀 AI 농구 코칭 대시보드")
st.caption("코치/선수/부모 모드 · 훈련 로그 · 영상 분석 노트 · AI 피드백 · 리포트/내보내기")

# =========================
# Sidebar: Settings / API
# =========================
with st.sidebar:
    st.header("⚙️ 설정")
    app_mode = st.radio("모드 선택", ["코치", "선수", "부모"], horizontal=True)
    st.divider()
    openai_api_key = st.text_input("OpenAI API Key (선택)", type="password", placeholder="sk-...")
    st.caption("AI 피드백 기능 사용 시 필요. 없으면 앱은 기록 중심으로만 동작해요.")

# =========================
# DB (SQLite)
# =========================
DB_PATH = "coach_app.db"

def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def db_init():
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS players (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        grade TEXT,
        position TEXT,
        notes TEXT,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_date TEXT NOT NULL,
        team TEXT,
        title TEXT,
        duration_min INTEGER,
        focus TEXT,
        plan_json TEXT,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        player_id INTEGER NOT NULL,
        present INTEGER NOT NULL,
        intensity INTEGER,
        mood INTEGER,
        memo TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(session_id, player_id),
        FOREIGN KEY(session_id) REFERENCES sessions(id),
        FOREIGN KEY(player_id) REFERENCES players(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS video_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        note_date TEXT NOT NULL,
        game TEXT,
        team TEXT,
        quarter TEXT,
        timestamp TEXT,
        category TEXT,
        players TEXT,
        note TEXT,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_date TEXT NOT NULL,
        metric_type TEXT NOT NULL,
        player TEXT NOT NULL,
        made INTEGER,
        attempt INTEGER,
        percent REAL,
        grade TEXT,
        memo TEXT,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS parent_msgs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        msg_date TEXT NOT NULL,
        player TEXT NOT NULL,
        from_who TEXT,
        message TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()

db_init()

# =========================
# Utility
# =========================
def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")

def query_df(sql: str, params: Tuple = ()):
    conn = db_conn()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

def exec_sql(sql: str, params: Tuple = ()):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    conn.close()

def exec_sql_return_id(sql: str, params: Tuple = ()):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid

def grade_by_percent(p: float, scheme: str = "rebound_total") -> str:
    # 너가 자주 쓰던 등급 체계(리바운드 토탈용 등) 기반으로 두 가지 제공
    if scheme == "rebound_total":
        # 85%+ A, 75-84 B, 65-74 C, 55-64 D, 54 and down F
        if p >= 85: return "A"
        if p >= 75: return "B"
        if p >= 65: return "C"
        if p >= 55: return "D"
        return "F"
    else:
        # alt: 82%+ A, 72-81 B, 62-71 C, 52-61 D, else F
        if p >= 82: return "A"
        if p >= 72: return "B"
        if p >= 62: return "C"
        if p >= 52: return "D"
        return "F"

# =========================
# AI helper (optional)
# =========================
def ai_feedback(openai_key: str, payload: Dict[str, Any], tone: str = "coach") -> Optional[str]:
    if not openai_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=openai_key)

        if tone == "coach":
            system = "너는 농구 코치 겸 데이터 분석가다. 말은 짧고 명확하게. 실행 가능한 피드백 중심."
        elif tone == "player":
            system = "너는 선수 멘탈/루틴 코치다. 동기부여는 하되 과장하지 말고 구체적으로."
        else:
            system = "너는 학부모 상담 코치다. 공손하고 명확하게. 아이의 성장 포인트와 가정에서 할 과제를 제안."

        format_rule = """
한국어로 아래 형식 고정:
[핵심 요약] 2줄
[잘한 점] 불릿 3개
[보완 포인트] 불릿 3개
[다음 훈련 미션] 불릿 3개
[코치 한마디] 1줄
""".strip()

        resp = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": "아래 데이터를 바탕으로 피드백을 작성해줘.\n"
                                            f"데이터(JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
                                            f"{format_rule}"}
            ],
            text={"verbosity": "medium"},
        )
        text = getattr(resp, "output_text", "") or ""
        return text.strip() if text.strip() else None
    except Exception:
        return None

# =========================
# Layout tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① 선수/팀 관리", "② 훈련 세션(플랜·출석)", "③ 영상 분석 노트", "④ 지표 기록(리바운드·참여율)", "⑤ 학부모/리포트"
])

# =========================
# TAB 1: Players / Team
# =========================
with tab1:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("👥 선수 등록")
        name = st.text_input("선수 이름", placeholder="예: 이원석")
        grade = st.selectbox("학년/레벨(선택)", ["", "초4", "초5", "초6", "중1", "중2", "중3", "고", "성인/동호회"])
        position = st.selectbox("포지션(선택)", ["", "G", "F", "C", "G/F", "F/C"])
        notes = st.text_area("메모(선택)", placeholder="예: 왼손 피니시 약함, 박스아웃 적극적 등")
        if st.button("선수 추가", type="primary", disabled=not name.strip()):
            exec_sql(
                "INSERT INTO players(name, grade, position, notes, created_at) VALUES(?,?,?,?,?)",
                (name.strip(), grade, position, notes, now_iso())
            )
            st.success("선수 추가 완료")

    with c2:
        st.subheader("📋 선수 목록")
        pdf = query_df("SELECT * FROM players ORDER BY id DESC")
        st.dataframe(pdf, use_container_width=True, hide_index=True)

        st.markdown("##### 🧹 선수 삭제(주의)")
        del_id = st.number_input("삭제할 player id", min_value=0, step=1)
        if st.button("삭제 실행", disabled=del_id <= 0):
            exec_sql("DELETE FROM players WHERE id=?", (int(del_id),))
            st.warning("삭제 완료(연관 데이터는 남아 있을 수 있어요)")

# =========================
# TAB 2: Training Session + Attendance + Plan Builder
# =========================
with tab2:
    st.subheader("🗓️ 훈련 세션 생성(플랜 저장)")
    team = st.text_input("팀/클래스(선택)", placeholder="예: 6학년 B팀")
    sdate = st.date_input("훈련 날짜", value=dt.date.today())
    title = st.text_input("세션 제목", placeholder="예: 드리블+피니시+게임")
    duration = st.number_input("총 시간(분)", min_value=30, max_value=240, value=80, step=5)
    focus = st.text_input("오늘 핵심 포커스(한 줄)", placeholder="예: 수비 압박 대응 + 피니시 마무리")

    st.markdown("##### 🧱 플랜 빌더(드릴을 순서대로 추가)")
    if "plan_items" not in st.session_state:
        st.session_state.plan_items = []

    pcol1, pcol2, pcol3 = st.columns([2, 1, 1])
    with pcol1:
        drill = st.text_input("드릴/메뉴", placeholder="예: 스트레칭/워밍업, 풀코트 수비, 원샷, 투볼 드리블 등")
    with pcol2:
        minutes = st.number_input("분", min_value=1, max_value=60, value=8, step=1)
    with pcol3:
        intensity = st.selectbox("강도", ["Low", "Mid", "High"], index=1)

    if st.button("플랜에 추가"):
        if drill.strip():
            st.session_state.plan_items.append({"drill": drill.strip(), "min": int(minutes), "intensity": intensity})
        else:
            st.info("드릴 이름을 입력해줘")

    if st.session_state.plan_items:
        plan_df = pd.DataFrame(st.session_state.plan_items)
        st.dataframe(plan_df, use_container_width=True, hide_index=True)
        total_min = int(plan_df["min"].sum())
        st.caption(f"플랜 합계: {total_min}분 (세션 총 시간 {duration}분과 다르면 조절하면 돼요)")
        if st.button("플랜 초기화"):
            st.session_state.plan_items = []

    if st.button("세션 저장", type="primary"):
        plan_json = json.dumps(st.session_state.plan_items, ensure_ascii=False)
        sid = exec_sql_return_id(
            "INSERT INTO sessions(session_date, team, title, duration_min, focus, plan_json, created_at) VALUES(?,?,?,?,?,?,?)",
            (str(sdate), team, title, int(duration), focus, plan_json, now_iso())
        )
        st.success(f"세션 저장 완료 (session_id={sid})")

    st.divider()
    st.subheader("✅ 출석/컨디션 기록")

    sdf = query_df("SELECT id, session_date, team, title FROM sessions ORDER BY session_date DESC, id DESC")
    if sdf.empty:
        st.info("먼저 세션을 저장해줘.")
    else:
        session_label = sdf.apply(lambda r: f"[{r['id']}] {r['session_date']} | {r['team'] or '-'} | {r['title'] or '-'}", axis=1).tolist()
        session_map = dict(zip(session_label, sdf["id"].tolist()))
        chosen = st.selectbox("세션 선택", session_label)
        session_id = int(session_map[chosen])

        players = query_df("SELECT id, name, grade, position FROM players ORDER BY name ASC")
        if players.empty:
            st.info("선수를 먼저 등록해줘.")
        else:
            st.markdown("##### 선수별 출석/강도/기분/메모")
            rows = []
            for _, r in players.iterrows():
                pid = int(r["id"])
                name = r["name"]
                cols = st.columns([2, 1, 1, 3])
                with cols[0]:
                    present = st.checkbox(f"{name}", value=True, key=f"att_{session_id}_{pid}")
                with cols[1]:
                    inten = st.slider("강도", 1, 10, 6, key=f"inten_{session_id}_{pid}")
                with cols[2]:
                    mood = st.slider("기분", 1, 10, 6, key=f"mood_{session_id}_{pid}")
                with cols[3]:
                    memo = st.text_input("메모", key=f"memo_{session_id}_{pid}", placeholder="예: 왼손 마무리 집중 필요")
                rows.append((session_id, pid, int(present), int(inten), int(mood), memo))

            if st.button("출석 기록 저장", type="primary"):
                for (sid, pid, pres, inten, md, memo) in rows:
                    exec_sql("""
                        INSERT INTO attendance(session_id, player_id, present, intensity, mood, memo, created_at)
                        VALUES(?,?,?,?,?,?,?)
                        ON CONFLICT(session_id, player_id)
                        DO UPDATE SET present=excluded.present, intensity=excluded.intensity, mood=excluded.mood, memo=excluded.memo
                    """, (sid, pid, pres, inten, md, memo, now_iso()))
                st.success("저장 완료")

            adf = query_df("""
                SELECT s.session_date, s.team, s.title, p.name, a.present, a.intensity, a.mood, a.memo
                FROM attendance a
                JOIN players p ON p.id=a.player_id
                JOIN sessions s ON s.id=a.session_id
                WHERE a.session_id=?
                ORDER BY p.name ASC
            """, (session_id,))
            st.markdown("##### 저장된 출석/컨디션")
            st.dataframe(adf, use_container_width=True, hide_index=True)

# =========================
# TAB 3: Video analysis notes
# =========================
with tab3:
    st.subheader("🎥 영상 분석 노트(타임스탬프 기반)")
    ndate = st.date_input("날짜", value=dt.date.today(), key="vn_date")
    game = st.text_input("경기/영상 이름", placeholder="예: 삼성 vs KT (2/1)")
    team = st.text_input("팀(선택)", placeholder="예: 삼성")
    quarter = st.selectbox("쿼터/구간(선택)", ["", "1Q", "2Q", "3Q", "4Q", "연장", "하이라이트", "기타"])
    timestamp = st.text_input("타임스탬프", placeholder="예: 09:50, 02:00, 08:37")
    category = st.selectbox("카테고리", ["리바운드", "박스아웃", "수비", "공격", "트랜지션", "턴오버", "기타"])
    players_text = st.text_input("관련 선수(쉼표로)", placeholder="예: 신동혁, 한호빈")
    note = st.text_area("노트", placeholder="예: 9분50초 시점 4명이 동시에 오펜리바운드 진입 인상적")

    if st.button("영상 노트 저장", type="primary", disabled=not note.strip()):
        exec_sql("""
            INSERT INTO video_notes(note_date, game, team, quarter, timestamp, category, players, note, created_at)
            VALUES(?,?,?,?,?,?,?,?,?)
        """, (str(ndate), game, team, quarter, timestamp, category, players_text, note.strip(), now_iso()))
        st.success("저장 완료")

    st.markdown("##### 🔎 검색/필터")
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        q_category = st.selectbox("카테고리 필터", ["전체", "리바운드", "박스아웃", "수비", "공격", "트랜지션", "턴오버", "기타"])
    with f2:
        q_team = st.text_input("팀 검색", placeholder="예: 삼성", key="q_team")
    with f3:
        q_text = st.text_input("키워드(노트/선수/게임)", placeholder="예: 박스아웃, 9:50, 구탕", key="q_text")

    base = "SELECT * FROM video_notes WHERE 1=1"
    params = []
    if q_category != "전체":
        base += " AND category=?"
        params.append(q_category)
    if q_team.strip():
        base += " AND team LIKE ?"
        params.append(f"%{q_team.strip()}%")
    if q_text.strip():
        base += " AND (note LIKE ? OR players LIKE ? OR game LIKE ? OR timestamp LIKE ?)"
        params += [f"%{q_text.strip()}%"] * 4
    base += " ORDER BY note_date DESC, id DESC"

    vdf = query_df(base, tuple(params))
    st.dataframe(vdf, use_container_width=True, hide_index=True)

# =========================
# TAB 4: Metrics (Rebound/Participation etc.)
# =========================
with tab4:
    st.subheader("📊 지표 기록(예: 리바운드 참가율, 슈팅 성공률, 참여 퍼센트)")
    mdate = st.date_input("날짜", value=dt.date.today(), key="m_date")
    metric_type = st.selectbox("지표 타입", ["리바운드 참가율", "슈팅 성공률", "영상 참여율", "기타"])
    player = st.text_input("선수", placeholder="예: 이원석")
    made = st.number_input("성공/참가(분자)", min_value=0, value=0, step=1)
    attempt = st.number_input("기회(분모)", min_value=0, value=0, step=1)

    scheme = st.selectbox("등급 기준", ["rebound_total (85/75/65/55)", "alt (82/72/62/52)"])
    memo = st.text_input("메모(선택)", placeholder="예: 예전 경기보다 참가/박스아웃 좋아짐")

    percent = None
    grade = None
    if attempt > 0:
        percent = round((made / attempt) * 100, 1)
        grade = grade_by_percent(percent, "rebound_total" if scheme.startswith("rebound_total") else "alt")

    st.write(f"계산: {made}/{attempt} = {percent if percent is not None else '-'}% | 등급: {grade or '-'}")

    if st.button("지표 저장", type="primary", disabled=not player.strip()):
        exec_sql("""
            INSERT INTO metrics(metric_date, metric_type, player, made, attempt, percent, grade, memo, created_at)
            VALUES(?,?,?,?,?,?,?,?,?)
        """, (
            str(mdate), metric_type, player.strip(),
            int(made) if attempt > 0 else None,
            int(attempt) if attempt > 0 else None,
            float(percent) if percent is not None else None,
            grade, memo, now_iso()
        ))
        st.success("저장 완료")

    st.markdown("##### 📈 최근 30개 기록")
    mdf = query_df("SELECT * FROM metrics ORDER BY metric_date DESC, id DESC LIMIT 30")
    st.dataframe(mdf, use_container_width=True, hide_index=True)

    st.markdown("##### 📊 선수별 평균(지표 타입별)")
    if not mdf.empty:
        tmp = mdf.dropna(subset=["percent"])
        if not tmp.empty:
            pivot = tmp.groupby(["metric_type", "player"], as_index=False)["percent"].mean()
            st.dataframe(pivot.sort_values(["metric_type", "percent"], ascending=[True, False]),
                         use_container_width=True, hide_index=True)

# =========================
# TAB 5: Parent msgs + Reports + Export + AI
# =========================
with tab5:
    st.subheader("💬 학부모/상담 메시지 기록")
    msg_date = st.date_input("날짜", value=dt.date.today(), key="pm_date")
    pm_player = st.text_input("선수", placeholder="예: 신동혁", key="pm_player")
    from_who = st.selectbox("발신(선택)", ["", "부모", "선수", "코치", "기타"])
    message = st.text_area("메시지", placeholder="예: 최근 수면이 부족한데 운동 병행해도 될까요?")

    if st.button("메시지 저장", type="primary", disabled=not (pm_player.strip() and message.strip())):
        exec_sql("""
            INSERT INTO parent_msgs(msg_date, player, from_who, message, created_at)
            VALUES(?,?,?,?,?)
        """, (str(msg_date), pm_player.strip(), from_who, message.strip(), now_iso()))
        st.success("저장 완료")

    pmdf = query_df("SELECT * FROM parent_msgs ORDER BY msg_date DESC, id DESC LIMIT 30")
    st.dataframe(pmdf, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🧾 리포트 생성(선택: AI)")

    rcol1, rcol2, rcol3 = st.columns([1, 1, 1])
    with rcol1:
        r_player = st.text_input("리포트 대상 선수(선택)", placeholder="비우면 팀 리포트", key="r_player")
    with rcol2:
        days = st.number_input("최근 N일", min_value=1, max_value=60, value=7, step=1)
    with rcol3:
        tone = st.selectbox("AI 톤", ["coach", "player", "parent"])

    end = dt.date.today()
    start = end - dt.timedelta(days=int(days))

    # Assemble payload from DB
    att = query_df("""
        SELECT s.session_date, s.team, s.title, s.focus, p.name, a.present, a.intensity, a.mood, a.memo
        FROM attendance a
        JOIN players p ON p.id=a.player_id
        JOIN sessions s ON s.id=a.session_id
        WHERE date(s.session_date) BETWEEN date(?) AND date(?)
    """, (str(start), str(end)))

    notes = query_df("""
        SELECT note_date, game, team, quarter, timestamp, category, players, note
        FROM video_notes
        WHERE date(note_date) BETWEEN date(?) AND date(?)
        ORDER BY note_date DESC
    """, (str(start), str(end)))

    metrics = query_df("""
        SELECT metric_date, metric_type, player, made, attempt, percent, grade, memo
        FROM metrics
        WHERE date(metric_date) BETWEEN date(?) AND date(?)
        ORDER BY metric_date DESC
    """, (str(start), str(end)))

    if r_player.strip():
        att_f = att[att["name"] == r_player.strip()] if not att.empty else att
        notes_f = notes[notes["players"].fillna("").str.contains(r_player.strip())] if not notes.empty else notes
        metrics_f = metrics[metrics["player"] == r_player.strip()] if not metrics.empty else metrics
    else:
        att_f, notes_f, metrics_f = att, notes, metrics

    payload = {
        "period": {"start": str(start), "end": str(end)},
        "mode": app_mode,
        "target_player": r_player.strip() if r_player.strip() else None,
        "attendance_summary": att_f.tail(50).to_dict(orient="records") if not att_f.empty else [],
        "video_notes": notes_f.tail(50).to_dict(orient="records") if not notes_f.empty else [],
        "metrics": metrics_f.tail(50).to_dict(orient="records") if not metrics_f.empty else [],
        "request": "최근 기록을 바탕으로 핵심 요약/칭찬/보완/다음 미션을 뽑아줘."
    }

    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### 📌 리포트 원본 데이터(요약)")
        st.write(f"- 출석/컨디션 rows: {0 if att_f.empty else len(att_f)}")
        st.write(f"- 영상 노트 rows: {0 if notes_f.empty else len(notes_f)}")
        st.write(f"- 지표 rows: {0 if metrics_f.empty else len(metrics_f)}")
        st.code(json.dumps(payload, ensure_ascii=False, indent=2)[:4000], language="json")

    with right:
        st.markdown("##### 🤖 AI 피드백(선택)")
        if st.button("AI 피드백 생성", type="primary"):
            with st.spinner("AI가 코치 노트를 쓰는 중..."):
                fb = ai_feedback(openai_api_key, payload, tone=tone)
            if fb:
                st.markdown(fb)
                st.markdown("##### 📋 공유용 텍스트")
                st.code(fb, language="markdown")
            else:
                if not openai_api_key:
                    st.warning("OpenAI API Key가 없어서 AI 기능은 패스했어. (기록/리포트는 계속 사용 가능)")
                else:
                    st.error("AI 피드백 생성 실패(키/네트워크/모델 권한 확인)")

    st.divider()
    st.subheader("⬇️ 내보내기(Export)")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.download_button(
            "출석 데이터 CSV",
            data=att_f.to_csv(index=False).encode("utf-8-sig") if not att_f.empty else "empty".encode(),
            file_name="attendance.csv",
            mime="text/csv"
        )
    with ex2:
        st.download_button(
            "영상 노트 CSV",
            data=notes_f.to_csv(index=False).encode("utf-8-sig") if not notes_f.empty else "empty".encode(),
            file_name="video_notes.csv",
            mime="text/csv"
        )
    with ex3:
        st.download_button(
            "지표 데이터 CSV",
            data=metrics_f.to_csv(index=False).encode("utf-8-sig") if not metrics_f.empty else "empty".encode(),
            file_name="metrics.csv",
            mime="text/csv"
        )

    st.caption("팁: Streamlit Cloud에 올릴 땐 DB 파일이 재시작 시 초기화될 수 있어요. 진짜 운영이면 Postgres/Supabase로 바꾸는 게 좋아요.")
