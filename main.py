import streamlit as st
from datetime import datetime
import os

# 로그 파일 경로
LOG_FILE = "access_log.txt"

# 접속 기록 함수
def log_access():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{now}\n")
    return now

# 최근 접속 일시 읽기 함수
def read_recent_access(n=5):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    # 최근 n개 기록 반환
    return [line.strip() for line in lines[-n:]]

# Streamlit UI
st.title("Welcome to the NetPipe Application")
st.subheader("Main Page")
st.write("""
This is the main page of the NetPipe application. 
Use the navigation menu to explore different features of the app.
""")
st.info("This application is designed to help you manage and analyze network pipelines efficiently.")

# 로그 기록
current_access = log_access()

# 최근 접속 읽기
recent_accesses = read_recent_access()

# Active Pipelines metric 예시로 최근 접속 횟수 활용
st.metric(label="Recent Accesses", value=len(recent_accesses), delta=f"+{len(recent_accesses)-1} from first in list")

# 최근 접속 일시 표시
st.write("### Recent Access Times")
for access in reversed(recent_accesses):
    st.write(f"- {access}")

st.write("More features coming soon!")
