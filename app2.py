from dotenv import load_dotenv
import os
import openai
import streamlit as st
import copy
from datetime import datetime

# .env 파일에서 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI API 키 설정
openai.api_key = OPENAI_API_KEY

# 이미지 파일 경로
image_path = "img/image.png"

# Streamlit 애플리케이션 구성
def main():
    # Streamlit 설정
    st.set_page_config(
        page_title="JobGPT - AI 커리어 도우미",
        page_icon=image_path,
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # 메인 화면 로고 이미지
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=False, caption="당신의 AI 커리어 도우미")
    else:
        st.error("이미지 파일을 찾을 수 없습니다. '{}' 경로를 확인하세요.".format(image_path))

    # 메인 화면 제목
    st.title("JobGPT에 오신 것을 환영합니다.")
    st.markdown("""
    JobGPT는 취업과 경력 개발을 지원하는 AI 기반 챗봇입니다.  
    아래 버튼을 클릭하여 질문을 입력해 보세요!
    """)

    # 초기 세션 상태 설정
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "saved_sessions" not in st.session_state:
        st.session_state["saved_sessions"] = []

    # 사이드바: 이전 채팅 세션을 불러오기 위한 인터페이스
    with st.sidebar:
        st.header("JobGPT 메뉴")

        # 이전 세션 불러오기
        saved_sessions = st.session_state["saved_sessions"]
        if saved_sessions:
            selected_session = st.selectbox("이전 채팅 세션 불러오기", options=list(range(len(saved_sessions))), format_func=lambda x: f"채팅 기록 {x + 1}")
            if st.button("선택된 세션 불러오기"):
                # 선택된 세션 불러오기
                st.session_state["messages"] = saved_sessions[selected_session]

        st.markdown("---")
        st.markdown("📩 **Contact us:** wriml92@knou.ac.kr")

    # 사용자 입력 섹션
    user_input = st.text_input("메세지를 입력해 주십시오.", key="user_input")
    send_button = st.button("Send")

    # 메시지 처리
    if send_button and user_input:
        # 사용자 메시지를 세션 상태에 추가
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # OpenAI GPT-4 모델에 메시지를 보내기
        bot_response = get_openai_response(user_input)

        # JobGPT 응답을 세션 상태에 추가
        st.session_state["messages"].append({"role": "assistant", "content": bot_response})

    # 현재 대화를 저장하기 위한 버튼
    if st.button("현재 대화 저장"):
        # 세션 저장
        st.session_state["saved_sessions"].append(copy.deepcopy(st.session_state["messages"]))
        st.success("현재 대화가 저장되었습니다!")

        # 대화 내용을 텍스트 파일로 저장
        save_chat_to_file(st.session_state["messages"])

    # 사용자와 JobGPT 메시지 스타일
    user_message = """
    <div style="
    background-color: #26a6f0;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    max-width: 60%;
    margin-left: auto;
    ">
        <img src="image.png" style="border-radius: 50%; margin-left: 10px;" alt="User Avatar">
        {message}
    </div>
    """
    assistant_message = """
    <div style="
    background-color: #f1f1f1;
    color: black;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    width: 60%;
    margin-right: auto;
    ">
        <img src="image.png" style="border-radius: 50%; margin-right: 10px;" alt="Bot Avatar">
        {message}
    </div>
    """

    # 채팅 인터페이스
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(user_message.format(message=msg["content"]), unsafe_allow_html=True)
        else:
            st.markdown(assistant_message.format(message=msg["content"]), unsafe_allow_html=True)

# OpenAI GPT-4 API를 호출하여 사용자의 질문에 대한 응답을 생성하는 함수
def get_openai_response(user_input):
    try:
        messages = [{"role": "system", "content": "You are a helpful assistant specialized in job searching and career advice."}]
        messages += st.session_state["messages"]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    except openai.error.OpenAIError as e:
        return f"OpenAI API에서 오류가 발생했습니다: {str(e)}"
    except Exception as e:
        return "오류가 발생했습니다. 인터넷 연결을 확인하고 다시 시도해 주세요."

# 대화 내용을 파일로 저장하는 함수
def save_chat_to_file(messages):
    try:
        # 파일명에 저장 시간을 추가하여 고유하게 만듦
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.txt"

        # 메시지들을 파일에 저장
        with open(filename, "w", encoding="utf-8") as file:
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"]
                file.write(f"{role}: {content}\n")

        st.success(f"채팅 내용이 {filename}에 저장되었습니다.")
    except Exception as e:
        st.error(f"채팅 내용을 저장하는 중 오류가 발생했습니다: {str(e)}")

# Streamlit 앱 실행
if __name__ == "__main__":
    main()