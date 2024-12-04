import os
import openai
import streamlit as st
import copy
from datetime import datetime
import llm_model_rag as llm

# OpenAI API 키 초기화
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""

# 이미지 파일 경로
logo_image = "image/logo_image.png"
user_avatar = "image/logo_image.png"
assistant_avatar = "image/assistant_avatar.png"

# 대화 내역 저장 파일
download_folder = './chat_history'

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

    # 채팅 파일 목록 가져오기
    chat_files = []
    if os.path.exists(download_folder):
        chat_files = [f for f in os.listdir(download_folder) if f.endswith('.txt')]
    else:
        st.warning(f"채팅 기록 폴더 '{download_folder}'가 존재하지 않습니다.")

    # 사이드바: 이전 채팅 세션을 불러오기 위한 인터페이스
    with st.sidebar:
        st.header("📋 JobGPT 메뉴")

        # 언어 선택
        language_options = ["Korean", "English", "Japanese", "Chinese", "Spanish", "French"]
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

        st.subheader("🔊 ElevenLabs API 키 입력")
        elevenlabs_api_key_input = st.text_input("ElevenLabs API 키를 입력하세요", type="password")
        if elevenlabs_api_key_input:
            st.session_state["ELEVENLABS_API_KEY"] = elevenlabs_api_key_input
            st.success("ElevenLabs API 키가 설정되었습니다.")

        st.markdown("---")
        st.subheader("📂 채팅 txt 파일 불러오기")

        if chat_files:
            selected_file = st.selectbox("불러올 채팅 파일을 선택하세요", chat_files)
            if st.button("채팅 기록 불러오기"):
                filepath = os.path.join(download_folder, selected_file)
                with open(filepath, "r", encoding="utf-8") as file:
                    loaded_messages = load_chat_from_file(file)
                    if loaded_messages:
                        st.session_state["messages"] = loaded_messages
                        st.success(f"채팅 기록 '{selected_file}'를 불러왔습니다.")
                    else:
                        st.error("채팅 내용을 불러오는 중 오류가 발생했습니다.")
        else:
            st.info("저장된 채팅 기록이 없습니다.")

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
        openai_api_key = st.session_state["OPENAI_API_KEY"]
        bot_response = get_openai_response(user_input, openai_api_key)

        # JobGPT 응답을 세션 상태에 추가
        st.session_state["messages"].append({"role": "assistant", "content": bot_response})

    col1, col2, col3 = st.columns([0.2, 0.2, 0.4])
    
    with col1:
        # 현재 대화를 저장하기 위한 버튼
        if st.button("현재 대화 저장"):
            if st.session_state.get("session_id"):
                st.session_state["saved_sessions"][st.session_state["session_id"]] = copy.deepcopy(st.session_state["messages"])
                st.success(f"세션 ID '{st.session_state['session_id']}'로 현재 대화가 저장되었습니다!")
                save_chat_to_file(st.session_state["messages"], st.session_state["session_id"])
            else:
                st.error("세션 ID를 입력하세요.")
    
    with col2:
        if st.button("음성 채팅"):
            st.info("마이크로 질문을 녹음하세요.")
            # 언어 코드 매핑
            language_code_mapping = {
                "Korean": "ko-KR",
                "English": "en-US",
                "Japanese": "ja-JP",
                "Chinese": "zh-CN",
                "Spanish": "es-ES",
                "French": "fr-FR"
            }
            # api-key
            openai_api_key = st.session_state["OPENAI_API_KEY"]
            elevenlabs_api_key = st.session_state["ELEVENLABS_API_KEY"]

            user_language = st.session_state.get('selected_language', 'Korean')
            user_language_code = language_code_mapping.get(user_language, 'ko-KR')
            text = llm.record_audio(user_language_code)

            try:
                if text:
                    st.session_state["messages"].append({"role": "user", "content": text})
                    bot_response = get_openai_response(text, openai_api_key)
                    st.session_state["messages"].append({"role": "assistant", "content": bot_response})

                    st.info("응답을 음성으로 변환 중...")
                    bot_response_1 = llm.process_response_for_speech(bot_response)
                    audio_response = llm.text_to_speech(bot_response_1, elevenlabs_api_key)
                    if audio_response:
                        st.audio(audio_response, format="audio/wav")
                    else:
                        st.error("음성 변환 중 문제가 발생했습니다.")

            except Exception as e:
                st.error(f"음성 입력 처리 중 오류가 발생했습니다: {str(e)}")


    # 세션 상태에 'refresh' 키가 없으면 초기화
    if 'refresh' not in st.session_state:
        st.session_state['refresh'] = False
    
    with col3:
        if st.button("채팅 내역 초기화"):
            st.session_state["messages"] = []
            st.success("채팅 기록이 초기화되었습니다.")

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
            if msg["role"] == "assistant":
                # 어시스먼트 말풍선
                st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; align-items: center; margin: 10px 0;">
                    <img src="https://raw.githubusercontent.com/wriml92/LLM_RAG_Team_Project/refs/heads/develop_1/image/assistant_avatar.png" alt="Assistant" style=" margin-right: 10px; width: 40px; height: 40px;">
                    <div style="
                        background-color: #f7fcfc;
                        color: #000;
                        padding: 10px 15px;
                        border-radius: 15px;
                        max-width: 70%;
                        text-align: left;
                        font-size: 14px;">
                        {msg['content']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# OpenAI GPT-4o API를 호출하여 사용자의 질문에 대한 응답을 생성하는 함수
def get_openai_response(user_input, api_key):
    try:
        # 선택된 언어 코드 가져오기
        user_language = st.session_state.get('selected_language', 'Korean')
        
        if "rag_initialized" not in st.session_state:
            data = llm.load_job_data('jobdata')
            retriever, embeddings, vectorstore = llm.create_retriever(data)
            prompt = llm.create_prompt(language=user_language)
            chain = llm.create_chain(retriever, prompt, api_key)
            rag_with_chat = llm.create_rag_with_chat(chain)
            st.session_state["rag_with_chat"] = llm.create_rag_with_chat(chain)
            st.session_state["vectorstore"] = vectorstore
            st.session_state["embeddings"] = embeddings
            st.session_state["rag_initialized"] = True
        else:
            rag_with_chat = st.session_state["rag_with_chat"]
            vectorstore = st.session_state["vectorstore"]
            embeddings = st.session_state["embeddings"]

        session_id = st.session_state.get("session_id", "default_session")

        response = llm.realtime_data_update_model(
            rag_with_chat, session_id, user_input, vectorstore, embeddings
        )

        return response
    
    except openai.OpenAIError as e: # 예외 처리 수정
        return f"OpenAI API에서 오류가 발생했습니다: {str(e)}"
    except Exception as e:
        return "오류가 발생했습니다. 인터넷 연결을 확인하고 다시 시도해 주세요."

# 대화 내용을 파일로 저장하는 함수
def save_chat_to_file(messages, session_id=None):
    try:
        # 채팅 기록 저장 폴더
        download_folder = './chat_history'
        os.makedirs(download_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"chat_{session_id}_{timestamp}.txt"
        file_path = os.path.join(download_folder, file_name)

        with open(file_path, "w", encoding="utf-8") as file:
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"]
                file.write(f"{role}: {content}\n")

        st.success(f"채팅 내용이 {file_path}에 저장되었습니다.")
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