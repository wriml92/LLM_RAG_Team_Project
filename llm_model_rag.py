# 환경 설정 및 변수 관리 관련 도구 임포트.
import os                                                                   # 운영 체제 관련 기능 사용을 위한 os 모듈 임포트.
import json                                                                 # JSON 데이터 처리를 위한 라이브러리.
from dotenv import load_dotenv                                              # .env 파일에서 환경 변수 로드.

# LangChain 관련 도구 임포트.
from langchain_text_splitters import RecursiveCharacterTextSplitter         # 긴 텍스트를 문단, 문장 등으로 잘게 나누는 도구.
from langchain.vectorstores import FAISS                                    # FAISS를 활용, 검색과 유사도 계산에 사용하는 벡터 데이터베이스를 가져옴.
from langchain_core.output_parsers import StrOutputParser                   # 출력 데이터를 문자열로 파싱하는 도구를 가져옴.
from langchain_core.prompts import PromptTemplate                           # 프롬프트 템플릿을 생성 및 관리하기 위한 도구.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings                   # OpenAI의 ChatGPT API를 사용하기 위한 인터페이스를 가져옴.
from langchain_community.chat_message_histories import ChatMessageHistory   # 대화 기록을 관리하는 도구. (채팅 메시지 히스토리)
from langchain_core.runnables.history import RunnableWithMessageHistory     # 대화 기록을 기반으로 작업을 처리하는 실행 도구.
from operator import itemgetter                                             # 리스트나 딕셔너리에서, 데이터의 특정 키나 항목을 쉽게 가져오기 위한 유틸리티 함수.
from langchain_core.documents import Document                               # 문서 데이터를 구조화하고 관리하기 위한 클래스를 가져옴.
from langchain.embeddings import HuggingFaceEmbeddings                      # HuggingFace 모델 활용한 임베딩 생성 기능을 가져옴.
from langchain.callbacks import StreamingStdOutCallbackHandler              # AI 모델의 출력 내용을 스트리밍 방식으로 실시간 출력 가능.

# 음성 기능 구현 라이브러리.
import requests                  # HTTP 요청을 보내고, 응답을 처리하기 위한 라이브러리.
from io import BytesIO           # 메모리 내에서 파일처럼 동작하는 바이너리 스트림을 관리하기 위한 모듈.
from pydub import AudioSegment   # 오디오 파일을 다양한 형식으로 변환, 처리 가능한 도구.
from pydub.playback import play  # 오디오 데이터를 재생하기 위한 함수.
import speech_recognition as sr  # 음성 인식을 통해 음성을 텍스트로 변환하는 라이브러리.

'''
api_key가 key.env파일에 존재.
하지만 배포 시 key.env파일은 따로 커밋되지 않음.
따라서 사용자의 개인 key.env파일을 생성해야함.
'''

# api key, env파일에서 로드.
load_dotenv(dotenv_path='key.env')

# openai
api_key = os.getenv("OPENAI_API_KEY")

# ElevenLabs API 설정
eleven_api_key = os.getenv("Elevenlabs_API_KEY")  # 환경 변수에서 ElevenLabs API 키를 가져옴.
voice_url = os.getenv("Eleven_URL")               # 환경 변수에서 ElevenLabs 음성 변환 API의 URL 가져옴.

def load_job_data(folder_path):
    """
    job이라는 json 파일의 내용을 기본 문서인 document로 변형.
    폴더 내의 모든 JSON 파일을 읽고, 채용 데이터를 하나의 리스트로 통합하여 반환하는 함수.
    """
    data = [] # 데이터를 저장할 리스트.
    for file in os.listdir(folder_path):                   #폴더 안의 모든 파일 확인.
        if file.endswith(".json"):                         #json 파일만 선택.
            file_path = os.path.join(folder_path, file)    #파일 경로 생성.
            # json파일 불러오기.
            with open(file_path, "r", encoding="utf-8") as file:
                job_data = json.load(file)
                if isinstance(job_data, list):  # 데이터가 리스트 형태인지 확인.
                    data.extend(job_data)       #리스트 데이터를 통합.

    data_2 = []

    for info in data:
        # 회사 정보를 키-값 쌍 형식의 문자열로 변환.
        company_info = "\n".join([f"{key}: {value}" for key, value in info["content"].items()])
        
        # 직무 정보(content)를 형식화된 텍스트로 구성.
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
        # 텍스트를 LangChain의 Document 객체로 변환.
        doc = Document(page_content=content.strip())
        data_2.append(doc) # 변환된 Document를 리스트에 추가.
    return data_2

def create_retriever(data):
    '''텍스트 정보 청킹 후에 임베딩 생성, 저장 공간인 faiss vector DB와 이를 활용해 검색기 생성.'''
    # document chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(data)

    # 임베딩 생성 및 벡터 DB 생성, 저장.
    # embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # BM25, hybrid search을 활용하여 성능 개선 가능.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 검색기(retriever) 생성. 문서의 정보 검색 및 생성.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever, embeddings, vectorstore

def create_prompt(language="Korean"):
    # 프롬프트 작성.
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
        # 1. Summation: [질문에 대한 간단한 요약]
        2. Related information: [추가로 제공할 수 있는 세부 정보]
        3. URL: [관련 링크 또는 소스]
        """
        )
    return prompt

def create_chain(retriever, prompt, api_key):
    """
    데이터를 검색하고 응답을 생성하는 체인을 생성하는 함수.
    데이터 검색(retriever), 질문 처리(prompt), 모델 응답 생성(llm) 과정을 연결.
    """
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o",
        temperature=0.3,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
        ) 

    # 체인 생성
    chain = (
        {
            "info": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt            #프롬프트를 통해 질문 정교화.
        | llm               #gpt-4o를 사용, 응답 생성.
        | StrOutputParser() #응답을 문자열로 파싱.
    )

    return chain

# 대화 세션 기록
chat = {}

def get_session_history(session_ids):
    '''세션 ID 기반으로 세션 기록을 가져오는 함수'''
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in chat:  # 세션 ID가 chat에 없는 경우.
       
        # 새로운 ChatMessageHistory 객체를 생성하여 chat에 저장.
        chat[session_ids] = ChatMessageHistory()
    return chat[session_ids]

def create_rag_with_chat(chain):
    '''체인과 대화기록을 연결, 사용자의 대화 흐름을 기반으로 동작하는 객체 RunnableWithMessageHistory 생성.'''
    rag_with_chat = RunnableWithMessageHistory(
        chain,  # 실행할 체인. (질문 처리 및 응답 생성)
        get_session_history,  # 세션 기록을 가져오는 함수.
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key.
        history_messages_key="chat_history",  # 기록 메시지의 키.
    )
    return rag_with_chat

def save_chat_in_vectorstore(session_id, question, answer, vectorstore, embeddings):
    """사용자 대화 기록을 Vector DB에 추가하여 성능을 개선."""
    # 대화 기록을 Document 형태로 변환. /1. 사용자를 식별할 ID, 2. 사용자가 입력한 질문, 3. AI의 응답. 
    content = f"""
    Session ID: {session_id}    
    User Question: {question}
    Assistant Answer: {answer}
    """
    document = Document(page_content=content.strip())   # LangChain의 Document 객체 생성

    # 임베딩을 생성, 대화 내용을 벡터로 변환 후 Vector DB에 추가.
    new_vectorstore = FAISS.from_documents([document], embedding=embeddings)

    # 기존 Vector DB와 새롭게 생성한 Vector DB병합.
    vectorstore.merge_from(new_vectorstore)
    
    '''
    새 벡터스토어를 만드는 건 비효율적일 수 있음. merge를 활용하지 않고, 아래 코드를 활용해 기존 데이터베이스에 추가하는 방법도 사용 가능
    vectorstore.add_documents([document], embedding=embeddings)
    '''
    
# 사용자 대화 내역 저장 및 성능 개선
def realtime_data_update_model(rag_with_chat, session_id, question, vectorstore, embeddings):
    """
    RAG 체인과 사용자 대화를 관리하며 실시간으로 Vector DB 업데이트.
    """
    try:
        # RAG 모델 실행 / 사용자 질문과 세션 ID 기반으로, RAG 모델에 요청을 보내 응답을 받음.
        response = rag_with_chat.invoke({"question": question,"session_ids": session_id}, config={"configurable": {"session_id": session_id}})
        
        # 응답이 비어 있을 경우, 예외 처리.
        if not response:
            raise ValueError("Empty response from RAG model.")
        
        # 대화 내용 저장
        session_history = get_session_history(session_id)   # 세션 기록을 가져와, 사용자 메시지와 AI 응답을 추가 저장.
        session_history.add_user_message(question)          # 사용자의 질문 저장.
        session_history.add_ai_message(response)            # AI의 응답 저장.

        # Vector DB 업데이트 / 질문과 응답 데이터를 벡터 데이터베이스에 저장, 향후 검색 및 유사도 계산에 활용.
        save_chat_in_vectorstore(session_id, question, response, vectorstore, embeddings)

        return response
    
    # 예외가 발생할 경우 오류 메시지 출력, 사용자에게 실패 메시지를 표시.
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
    '''
    recognizer = sr.Recognizer()                    # 음성 인식을 위한, 인식기 객체 생성.
    recognizer.energy_threshold = energy_threshold  # 민감도 설정 (배경 소음을 무시)
    recognizer.pause_threshold = pause_threshold    # 음성입력을 안 해도, 인식을 중단하지 않는 시간.

    # 마이크에서 음성 데이터를 수집.
    with sr.Microphone() as source:
        print("음성 입력 중...(입력 시간 15초, 미입력시 대기 10초)(중지: 'Ctrl+C')")
        try:
            # timeout : 음성 입력이 시작되지 않을 경우 대기할 시간.
            # phrase_time_limit : 전체 음성을 입력받을 최대 시간.
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=listen_time)
            text = recognizer.recognize_google(audio, language=language)    # 음성을 텍스트로 변환.
            print(f"텍스트: {text}")
            return text
        
        # 음성 인식 실패, 문제 발생할 경우의 오류 처리들.
        except sr.UnknownValueError:    # 음성을 인식하지 못할 경우.
            print("음성 인식 불가.")
            return None
        
        except sr.WaitTimeoutError:     # 음성 입력 대기 시간 초과.
            print("음성 입력이 감지되지 않았음. 다시 시도 요망.")
            return None
        
        except sr.RequestError as e:    # 음성 인식 서비스 요청 실패.
            print(f"음성 서비스 오류: {e}")
            return None
        
def process_response_for_speech(response):
    """
    AI 응답에서 음성 출력에 필요한 부분만 추출 (URL 제외).
    """
    try:
        # 응답에 '3. URL:'이 포함된다면, 이후 텍스트를 제거.
        if "3. URL:" in response:
            response = response.split("3. URL:")[0].strip() # URL 이전 부분만 남기고 공백 제거.
        return response
    except Exception as e:
        print(f"[오류] 응답 처리 실패: {e}")
        return response  # 실패 시 로그 출력 후, 원래 응답 그대로 반환.

def text_to_speech(text, api_key, voice_url="https://api.elevenlabs.io/v1/text-to-speech/eVItLK1UvXctxuaRV2Oq"):
    """
    ElevenLabs TTS를 사용해 텍스트를 음성으로 변환하고 재생.
    """
    try:
        headers = {
            "Accept": "audio/mpeg",             # 오디오 형식 지정. (audio/mpeg)
            "xi-api-key": api_key,              # ElevenLabs API 키 인증.
            "Content-Type": "application/json", # 요청 데이터 형식(JSON) 지정.
        }
        data = {
            "text": text,                         # 변환할 텍스트.
            "model_id": "eleven_multilingual_v2", # 다국어 지원 TTS 모델 사용.
            "voice_settings": {
                "stability": 0.5,                 # 음성 안정성 설정.
                "similarity_boost": 1,            # 텍스트와 음성의 유사성 강화.
                "style": 1,                       # 음성 스타일 조정. (목소리에 감정을 얼마나 담을 것인지 결정.)
                "use_speaker_boost": True         # 음성 출력 부스트 활성화.
            }
        }
        # ElevenLabs API로 POST 요청.
        response = requests.post(voice_url, headers=headers, json=data) # 텍스트를 음성으로 변환한 결과 반환.
        response.raise_for_status() # HTTP 요청 실패를 감지하고 프로그램에 알림. 요청이 실패하면,에러발생의 이유를 알려줌.

        # 변환된 오디오 데이터를 메모리 스트림으로 읽어 재생.
        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        play(audio)

        # 오류 메시지 출력으로, 예외처리.
    except Exception as e:
        print(f"음성 변환 오류: {e}")

# 코드 테스트용.
if __name__ == "__main__":
    session_id = "jungseok" #세션 ID 설정.
    language = input("언어 선택 (ex: Korean): ")    # 사용자의 언어 선택.
    mode = input("채팅은 0, 음성은 1 선택: ")       # 사용자가 채팅 또는 음성 입력 중 하나를 선택.
    while True:
        if mode == "0":     # 텍스트 기반 채팅 모드.
            question = input("질문 입력하세요(exit 입력 시 종료): ")
            if question == 'exit':
                print("프로그램 종료.")
                break
        elif mode == "1":   # 음성 입력 모드.
            print("질문 입력하세요(exit 입력 시 종료): ")
            question = record_audio()   # 사용자의 음성을 녹음하고 텍스트로 변환.
            if question is None:        # 음성 인식에 실패한 경우 루프를 재시작.
                continue
            if question.lower() in ["exit", "종료"]:
                print("프로그램 종료.")
                break

        data = load_job_data('jobdata') # 'jobdata' 파일에서 직업 관련 데이터를 불러옴.
        retriever, embeddings, vectorstore = create_retriever(data) # 데이터에서 검색기, 임베딩, 벡터 저장소를 생성
        prompt = create_prompt(language=language)   # 사용자가 선택한 언어에 맞는 프롬프트 생성. (지침 역할)
        chain = create_chain(retriever, prompt, api_key) # 검색 및 답변을 위한 RAG 체인 생성.
        rag_with_chat = create_rag_with_chat(chain) # RAG 모델, 대화 기능을 연결한 객체 생성.

        # 사용자 질문을 기반으로, RAG 모델을 실행하여 응답을 얻음. 대화 내용은 벡터DB에 저장.
        response = realtime_data_update_model(rag_with_chat, session_id, question, vectorstore, embeddings)

        print(f"답: {response}")
        
        if mode == "1":
            speech_response = process_response_for_speech(response) # AI의 응답에서 음성 출력에 필요한 부분만 추출.
            text_to_speech(speech_response, eleven_api_key) # 변환된 텍스트를 음성으로 출력.