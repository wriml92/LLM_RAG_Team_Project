import os
import openai
import streamlit as st
import copy
from datetime import datetime
from openai import OpenAIError
from elevenlabs.client import ElevenLabs

# OpenAI API 키 초기화
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""

# 이미지 파일 경로
logo_image = "image/logo_image.png"
user_avatar = "image/logo_image.png"
assistant_avatar = "image/assistant_avatar.png"

# Streamlit 애플리케이션 구성
def main():
    # Streamlit 설정
    st.set_page_config(
        page_title="JobGPT - AI 커리어 도우미",
        page_icon=logo_image,
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # CSS 스타일 적용 함수
    def local_css(file_name):
        with open(file_name, encoding='utf-8') as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)

    # 스타일 적용
    local_css("style.css")

    # 메인 화면 로고 이미지
    if os.path.exists(logo_image):
        st.logo(logo_image)
    else:
        st.error(f"이미지 파일을 찾을 수 없습니다: {logo_image}")

    # 메인 화면 중앙 정렬
    with st.container():
        st.markdown("<h1 style='text-align: center;'>JobGPT에 오신 것을 환영합니다.</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p style='text-align: center;'>
        JobGPT는 취업과 경력 개발을 지원하는 <strong>AI 기반 챗봇</strong>입니다.<br>
        아래 입력창에 질문을 입력해 보세요!
        </p>
        """, unsafe_allow_html=True)

    # 초기 세션 상태 설정
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "saved_sessions" not in st.session_state:
        st.session_state["saved_sessions"] = {}
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = ""
    if "ELEVENLABS_API_KEY" not in st.session_state:
        st.session_state["ELEVENLABS_API_KEY"] = ""

    # 사이드바: 이전 채팅 세션을 불러오기 위한 인터페이스
    with st.sidebar:
        st.header("📋 JobGPT 메뉴")

        # 언어 선택
        language_options = ["한국어", "English", "日本語", "中文", "Español"]
        selected_language = st.selectbox("언어를 선택하세요", language_options)
        st.session_state['selected_language'] = selected_language

        st.markdown("---")
        st.subheader("🔑 OpenAI API 키 입력")
        api_key_input = st.text_input("OpenAI API 키를 입력하세요", type="password")
        if api_key_input:
            st.session_state["OPENAI_API_KEY"] = api_key_input
            st.success("API 키가 설정되었습니다.")
        
        st.subheader("🆔 세션 ID 입력")
        session_id_input = st.text_input("세션 ID를 입력하세요")
        if session_id_input:
            st.session_state["session_id"] = session_id_input
            st.success(f"세션 ID가 설정되었습니다: {session_id_input}")
            if session_id_input in st.session_state["saved_sessions"]:
                st.session_state["messages"] = copy.deepcopy(st.session_state["saved_sessions"][session_id_input])
                st.success(f"세션 ID '{session_id_input}'로 저장된 채팅 내용을 불러왔습니다.")
            else:
                # 파일에서 불러오기 시도
                filename = f"chat_history_{session_id_input}.txt"
                if os.path.exists(filename):
                    with open(filename, "r", encoding="utf-8") as file:
                        loaded_messages = load_chat_from_file(file)
                        if loaded_messages:
                            st.session_state["messages"] = loaded_messages
                            st.success(f"세션 ID '{session_id_input}'로 저장된 채팅 내용을 파일에서 불러왔습니다.")
                        else:
                            st.error("채팅 내용을 불러오는 중 오류가 발생했습니다.")
                else:
                    st.info(f"세션 ID '{session_id_input}'로 저장된 채팅 내용이 없습니다.")

        st.subheader("🔊 ElevenLabs API 키 입력")
        elevenlabs_api_key_input = st.text_input("ElevenLabs API 키를 입력하세요", type="password")
        if elevenlabs_api_key_input:
            st.session_state["ELEVENLABS_API_KEY"] = elevenlabs_api_key_input
            st.success("ElevenLabs API 키가 설정되었습니다.")

        st.markdown("---")
        st.subheader("📂 채팅 txt 파일 불러오기")

        # 파일 업로더 추가
        uploaded_file = st.file_uploader("채팅 txt 파일을 선택하세요", type="txt")

        # 파일이 업로드되면 처리
        if uploaded_file is not None:
            loaded_messages = load_chat_from_file(uploaded_file)
            if loaded_messages:
                st.session_state["messages"] = loaded_messages
                st.success("채팅 내용을 성공적으로 불러왔습니다.")
            else:
                st.error("채팅 내용을 불러오는 중 오류가 발생했습니다.")

        st.markdown("---")
        st.markdown("<p style='text-align: center;'>📩 <strong>Contact us:</strong> wriml92@knou.ac.kr</p>", unsafe_allow_html=True)

    # OpenAI API 키 설정
    if st.session_state["OPENAI_API_KEY"]:
        openai.api_key = st.session_state["OPENAI_API_KEY"]
    else:
        st.warning("OpenAI API 키를 입력하세요.")
        st.stop()

    # 사용자 입력 섹션
    user_input = st.chat_input("메세지를 입력해 주십시오.")

    # 메시지 처리
    if user_input:
        # 사용자 메시지를 세션 상태에 추가
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # OpenAI GPT-4o 모델에 메시지를 보내기
        bot_response = get_openai_response(user_input)

        # JobGPT 응답을 세션 상태에 추가
        st.session_state["messages"].append({"role": "assistant", "content": bot_response})

        # # ElevenLabs API 키가 설정되어 있는 경우 음성 생성 및 재생
        # if st.session_state["ELEVENLABS_API_KEY"]:
        #     audio = generate(
        #         text=bot_response,
        #         api_key=st.session_state["ELEVENLABS_API_KEY"],
        #         voice="Rachel"  # 원하는 목소리로 변경 가능
        #     )
        #     # 음성 재생
        #     st.audio(audio, format="audio/mp3")
        # else:
        #     st.info("ElevenLabs API 키가 설정되어 있지 않아 음성을 재생할 수 없습니다.")

    # 현재 대화를 저장하기 위한 버튼
    if st.button("현재 대화 저장"):
        if st.session_state.get("session_id"):
            st.session_state["saved_sessions"][st.session_state["session_id"]] = copy.deepcopy(st.session_state["messages"])
            st.success(f"세션 ID '{st.session_state['session_id']}'로 현재 대화가 저장되었습니다!")
            save_chat_to_file(st.session_state["messages"], st.session_state["session_id"])
        else:
            st.error("세션 ID를 입력하세요.")

    # 채팅 인터페이스
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            # 사용자 말풍선
            st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; margin: 10px 0;">
                <div style="
                    background-color: #E8F4FF;
                    color: #000;
                    padding: 10px 15px;
                    border-radius: 15px;
                    max-width: 70%;
                    text-align: right;
                    font-size: 14px;">
                    {msg['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        else:
            with st.chat_message("assistant", avatar=assistant_avatar):
                st.markdown(msg["content"])

# OpenAI GPT-4o API를 호출하여 사용자의 질문에 대한 응답을 생성하는 함수
def get_openai_response(user_input):
    try:
        # 언어 코드 매핑
        language_code_mapping = {
            "한국어": "ko",
            "English": "en",
            "日本語": "ja",
            "中文": "zh",
            "Español": "es"
        }
        # 선택된 언어 코드 가져오기
        user_language_code = language_code_mapping.get(st.session_state.get('selected_language', '한국어'), 'ko')

        # 시스템 메시지를 선택된 언어로 설정
        system_messages = {
            "ko": "당신은 취업 및 경력 상담에 특화된 유용한 도우미입니다.",
            "en": "You are a helpful assistant specialized in job searching and career advice.",
            "ja": "あなたは就職とキャリアアドバイスに特化した役に立つアシスタントです。",
            "zh": "你是一个专门从事求职和职业建议的有用助手。",
            "es": "Eres un asistente útil especializado en búsqueda de empleo y asesoramiento profesional."
        }
        system_message = system_messages.get(user_language_code, system_messages['en'])

        messages = [{"role": "system", "content": system_message}]

        # 이전 메시지 추가
        for msg in st.session_state["messages"]:
            if msg["role"] != "system":
                messages.append(msg)

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000, # 최대 토큰 길이 300자에서 1000자로 수정
            temperature=0.7
        )
        bot_response = response["choices"][0]["message"]["content"].strip()

        return bot_response
    except openai.OpenAIError as e: # 예외 처리 수정
        return f"OpenAI API에서 오류가 발생했습니다: {str(e)}"
    except Exception as e:
        return "오류가 발생했습니다. 인터넷 연결을 확인하고 다시 시도해 주세요."

# 대화 내용을 파일로 저장하는 함수
def save_chat_to_file(messages, session_id=None):
    try:
        if session_id:
            filename = f"chat_history_{session_id}.txt"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as file:
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"]
                file.write(f"{role}: {content}\n")

        st.success(f"채팅 내용이 {filename}에 저장되었습니다.")
    except Exception as e:
        st.error(f"채팅 내용을 저장하는 중 오류가 발생했습니다: {str(e)}")

def load_chat_from_file(file):
    try:
        messages = []
        content = file.read()
        lines = content.strip().split("\n")
        for line in lines:
            if line.startswith("User: "):
                message_content = line[len("User: "):]
                messages.append({"role": "user", "content": message_content})
            elif line.startswith("Assistant: "):
                message_content = line[len("Assistant: "):]
                messages.append({"role": "assistant", "content": message_content})
            else:
                continue
        return messages
    except Exception as e:
        st.error(f"채팅 내용을 불러오는 중 오류가 발생했습니다: {str(e)}")
        return None

# Streamlit 앱 실행
if __name__ == "__main__":
    main()