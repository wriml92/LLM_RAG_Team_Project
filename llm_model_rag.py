import os
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks import StreamingStdOutCallbackHandler

# 음성 기능 구현 라이브러리.
import requests
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr

'''
api_key가 key.env파일에 존재.
하지만 배포 시 key.env파일은 따로 커밋되지 않음.
따라서 사용자의 개인 key.env파일을 생성해야함.
'''
# api key, env파일에서 로드.
load_dotenv(dotenv_path='key.env')
# openai
api_key = os.getenv("OPENAI_API_KEY")
# elevenlabs
eleven_api_key = os.getenv("Elevenlabs_API_KEY")
voice_url = os.getenv("Eleven_URL")

def load_job_data(folder_path):
    '''
    job이라는 json 파일의 내용을 기본 문서인 document로 변형

    folder_path = collect_data.py에서 수집한 json파일들이 모여있는 폴더.
    '''
    data = []
    # jobdata 폴더에서 존재하는 json파일 모두 불러오기.
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding="utf-8") as file:
                job_data = json.load(file)
                if isinstance(job_data, list):
                    data.extend(job_data)

    print(data)

    data_2 = []

    for info in data:
        # 저장된 data 속에 company_info라는 딕셔너리 형태로 존재하는 데이터가 남아있으므로 텍스트형태로 key, value 값을 join()활용하여 텍스트형태로 만듦.
        company_info = "\n".join([f"{key}: {value}" for key, value in info["content"].items()])
        
        # content를 텍스트 형식으로.
        content = f"""
        Title: {info['title']}
        Company: {info['company']}
        Industry: {info['industry']}
        Location: {info['location']}
        Job Type: {info['job_type']}
        Categories: {', '.join(info['categories'])}
        Experience Level: {info['experience_level']}
        Education Level: {info['education_level']}
        Salary: {info['salary']}
        URL: {info['url']}
        Company URL: {info['company_url']}
        Expiration Date: {info['expiration_date']}
        Company Info: {company_info}
        """
        # 위 내용을 기본 문서단위 Document로 변환.
        doc = Document(page_content=content.strip())
        data_2.append(doc)  # 채용공고 하나씩 추가.
    return data_2

def create_retriever(data):
    '''텍스트 정보 청킹 후에 임베딩 생성, 저장 공간인 faiss vector DB와 이를 활용해 검색기 생성.'''
    # document chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(data)

    # 임베딩 생성 및 벡터 DB 생성, 저장(openaiembeddings, huggingface 중 택 1)
    # embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # BM25, hybrid search을 활용하여 성능 개선 가능.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 리트리버(검색기) 생성. 문서의 정보 검색 및 생성
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever, embeddings, vectorstore

def create_prompt(language="Korean"):
    # 프롬프트 템플릿 작성. (취업 전문 어시스턴트, 답변 언어, 이전 대화 내용, 질문, 참고 자료 등 템플릿 구성)
    prompt = PromptTemplate.from_template(
        f"""You are an expert assistant for job search and recommendation. 
        Always provide well-structured answers in {language}, including relevant data points.
        Avoid unnecessary elaboration.

        # Previous conversation history:
        {{chat_history}}

        # Question:Summation
        {{question}}

        # Information retrieved:
        {{info}}

        # Answer:
        1. Summation: [질문에 대한 간단한 요약]
        2. Related information: [추가로 제공할 수 있는 세부 정보]
        3. URL: [관련 링크 또는 소스]
        """
        )
    return prompt

def create_chain(retriever, prompt, api_key):
    '''
    ragchains 생성. 
    (질문 | 검색기(유사 부분 검색), 질문, 대화 기록) | 프롬프트 템플릿 | llm 모델 | 문자열출력()

    retriever = create_retriever() 생성한 검색기
    prompt = create_prompt()에서 구성한 프롬프트 템플릿
    api_key = openai api key.
    '''
    # llm모델 gpt-4o으로 생성.
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o",
        temperature=0.3,
        streaming=True, # 스트리밍 출력 기능.(파이썬 파일 자체 실행 시 터미널에 구현)
        callbacks=[StreamingStdOutCallbackHandler()]
        )

    # 체인 생성
    chain = (
        {
            "info": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# 대화 세션 기록
chat = {}

def get_session_history(session_ids):
    '''
    세션 ID 기반으로 세션 대화 기록을 가져오는 함수
    
    session_ids = 세션 id.
    '''
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in chat:  # 세션 ID가 chat에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 chat에 저장
        chat[session_ids] = ChatMessageHistory()
    return chat[session_ids]

def create_rag_with_chat(chain):
    '''
    RunnableWithMessageHistory 생성

    chain = create_chain()에서 생성한 ragchain.
    '''
    rag_with_chat = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return rag_with_chat

def save_chat_in_vectorstore(session_id, question, answer, vectorstore, embeddings):
    """
    사용자 대화 기록을 Vector DB에 추가하여 성능을 개선.

    session_id = 세션 id.
    question = 쿼리(유저의 질문)
    answer = 어시스턴트의 답변.
    vectorstore = create_retriever()에서 임베딩 생성한 뒤의 벡터DB
    embeddings = create_retriever()에서 생성된 임베딩
    """
    # 대화 기록을 Document 형태로 변환
    content = f"""
    Session ID: {session_id}
    User Question: {question}
    Assistant Answer: {answer}
    """
    document = Document(page_content=content.strip())

    # 임베딩을 생성하고 Vector DB에 추가
    new_vectorstore = FAISS.from_documents([document], embedding=embeddings)

    # 기존 Vector DB와 병합
    vectorstore.merge_from(new_vectorstore)

    '''
    새 벡터스토어를 만드는 건 비효율적일 수 있음. merge를 활용하지 않고, 
    아래 코드를 활용해 기존 데이터베이스에 추가하는 방법도 사용 가능.
    프로젝트 요구사항이 merge 활용이라고 명시되어 있어서 위처럼 코드 작성.
    
    vectorstore.add_documents([document], embedding=embeddings)
    '''

# 사용자 대화 내역 저장 및 성능 개선
def realtime_data_update_model(rag_with_chat, session_id, question, vectorstore, embeddings):
    """
    RAG 체인과 사용자 대화를 관리하며 실시간으로 Vector DB 업데이트.

    rag_with_chat = create_rag_with_chat()에서 create_chain()의 chain과 이전 대화 내용들(chat_history)과 형성한 chain.
    session_id = 세션 id.
    question = 쿼리(유저의 질문)
    vectorstore = create_retriever()에서 임베딩 생성한 뒤의 벡터DB(이 함수에서는 save_chat_in_vectorstore()에서 새로운 DB와 merge되어 업데이트 됨.)
    embeddings = create_retriever()에서 생성된 임베딩
    """
    try:
        # RAG 모델 실행
        response = rag_with_chat.invoke({"question": question,"session_ids": session_id}, config={"configurable": {"session_id": session_id}})

        if not response:
            raise ValueError("Empty response from RAG model.")
        
        # 대화 내용 저장
        session_history = get_session_history(session_id)
        session_history.add_user_message(question)
        session_history.add_ai_message(response)

        # Vector DB 업데이트
        save_chat_in_vectorstore(session_id, question, response, vectorstore, embeddings)

        return response
    
    except Exception as e:
        print(f"[오류] RAG 모델 업데이트 실패: {e}")
        return "[오류] 대화 업데이트 실패. 다시 시도하세요."
    
def record_audio(language="ko-KR", listen_time=15, energy_threshold=300, pause_threshold=2.0):
    '''
    마이크를 통해 음성 입력을 받고 텍스트로 변환.
    영어: en-US, en-GB
    일본어: ja-JP
    중국어: zh-CN
    불어: fr-FR
    스페인어: es-ES

    language = 설정 언어.
    listen_time = 입력받는 시간 길이.
    energy_threshold = 마이크 민감도.
    pause_threshold = 침묵 설정한 초시간을 넘을 때까지 입력이 멈추지 않음.
    '''
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = energy_threshold
    recognizer.pause_threshold = pause_threshold

    with sr.Microphone() as source:
        print("음성 입력 중...(입력 시간 15초, 미입력시 대기 10초)(중지: 'Ctrl+C')")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=listen_time)
            text = recognizer.recognize_google(audio, language=language)
            print(f"텍스트: {text}")
            return text
        
        except sr.UnknownValueError:
            print("음성 인식 불가.")
            return None
        
        except sr.WaitTimeoutError:
            print("음성 입력이 감지되지 않았음. 다시 시도 요망.")
            return None
        
        except sr.RequestError as e:
            print(f"음성 서비스 오류: {e}")
            return None
        
def process_response_for_speech(response):
    """
    AI 응답에서 음성 출력에 필요한 부분만 추출 (URL 제외).
    
    response = 챗봇에게 받은 답변.
    """
    try:
        # '3. URL:' 이후 텍스트를 제거
        if "3. URL:" in response:
            response = response.split("3. URL:")[0].strip()
        return response
    except Exception as e:
        print(f"[오류] 응답 처리 실패: {e}")
        return response  # 실패 시 원래 응답 반환

def text_to_speech(text, api_key, voice_url="https://api.elevenlabs.io/v1/text-to-speech/eVItLK1UvXctxuaRV2Oq"):
    """
    ElevenLabs TTS를 사용해 텍스트를 음성으로 변환하고 재생.

    text = tts할 문장.
    api_key = elevenlabs api key.
    voice_url = elevenlabs에서 반드시 목소리를 api_key 발급한 계정에서 사용설정 해야함.
    """
    try:
        headers = {
            "Accept": "audio/mpeg",
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 1,
                "style": 1,
                "use_speaker_boost": True
            }
        }
        # elevenlabs에서 tts한 오디오 파일 재생.
        response = requests.post(voice_url, headers=headers, json=data)
        response.raise_for_status()

        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        play(audio)

    except Exception as e:
        print(f"음성 변환 오류: {e}")

# 코드 구현 테스트용.
if __name__ == "__main__":
    session_id = input("session id 입력: ")
    language = input("언어 선택 (ex: Korean): ")
    mode = input("채팅은 0, 음성은 1 선택: ")
    while True:
        if mode == "0":
            question = input("질문 입력하세요(exit 입력 시 종료): ")
            if question == 'exit':
                print("프로그램 종료.")
                break
        elif mode == "1":
            print("질문 입력하세요(exit 입력 시 종료): ")
            question = record_audio()
            if question is None:
                continue
            if question.lower() in ["exit", "종료"]:
                print("프로그램 종료.")
                break

        data = load_job_data('jobdata')
        retriever, embeddings, vectorstore = create_retriever(data)
        prompt = create_prompt(language=language)
        chain = create_chain(retriever, prompt, api_key)
        rag_with_chat = create_rag_with_chat(chain)

        response = realtime_data_update_model(rag_with_chat, session_id, question, vectorstore, embeddings)

        print(f"답: {response}")
        
        if mode == "1":
            speech_response = process_response_for_speech(response)
            text_to_speech(speech_response, eleven_api_key)