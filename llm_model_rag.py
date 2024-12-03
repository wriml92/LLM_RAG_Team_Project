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

# 음성 기능 구현 라이브러리.
import requests
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

# api key, env파일에서 로드.
load_dotenv(dotenv_path='key.env')

# openai
api_key = os.getenv("OPENAI_API_KEY")

# elevenlabs
eleven_api_key = os.getenv("Elevenlabs_API_KEY")
voice_url = os.getenv("Eleven_URL")

def load_job_data(folder_path):
    '''job이라는 json 파일의 내용을 기본 문서인 document로 변형'''
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            # json파일 불러오기.
            with open(file_path, "r", encoding="utf-8") as file:
                job_data = json.load(file)
                if isinstance(job_data, list):
                    data.extend(job_data)

    data_2 = []

    for info in data:
        company_info = "\n".join([f"{key}: {value}" for key, value in info["content"].items()])
        
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
        doc = Document(page_content=content.strip())
        data_2.append(doc)
    return data_2

def create_retriever(data):
    '''텍스트 정보 청킹 후에 임베딩 생성, 저장 공간인 faiss vector DB와 이를 활용해 검색기 생성.'''
    # document chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(data)

    # 임베딩 생성 및 벡터 DB 생성, 저장
    # embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # BM25, hybrid search을 활용하여 성능 개선 가능.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 리트리버(검색기) 생성. 문서의 정보 검색 및 생성
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

def create_chain(retriever, prompt):
    # llm모델 gpt-4o으로 생성.
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

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
    '''세션 ID 기반으로 세션 기록을 가져오는 함수'''
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in chat:  # 세션 ID가 chat에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 chat에 저장
        chat[session_ids] = ChatMessageHistory()
    return chat[session_ids]

def create_rag_with_chat(chain):
    '''RunnableWithMessageHistory 생성'''
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
    새 벡터스토어를 만드는 건 비효율적일 수 있음. merge를 활용하지 않고, 아래 코드를 활용해 기존 데이터베이스에 추가하는 방법도 사용 가능
    vectorstore.add_documents([document], embedding=embeddings)
    '''

# 사용자 대화 내역 저장 및 성능 개선
def realtime_data_update_model(rag_with_chat, session_id, question, vectorstore, embeddings):
    """
    RAG 체인과 사용자 대화를 관리하며 실시간으로 Vector DB 업데이트.
    """
    try:
        # RAG 모델 실행
        response = rag_with_chat.invoke({"question": question,"session_ids": session_id},config={"configurable": {"session_id": session_id}})

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

# 코드 테스트용.
if __name__ == "__main__":
    session_id = "jungseok"
    language = input("언어 선택: ")
    while True:
        question = input("질문 입력하세요(exit 입력 시 종료): ")
        if question == "exit":
            break

        data = load_job_data('jobdata')
        retriever, embeddings, vectorstore = create_retriever(data)
        prompt = create_prompt(language=language)
        chain = create_chain(retriever, prompt)
        rag_with_chat = create_rag_with_chat(chain)

        response = realtime_data_update_model(rag_with_chat, session_id, question, vectorstore, embeddings)

        print(f"답: {response}")