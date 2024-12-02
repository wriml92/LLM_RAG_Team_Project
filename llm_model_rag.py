import os                                                                   # 운영 체제 관련 기능을 사용을 위한 os 모듈 임포트.
import json                                                                 # JSON 데이터 처리를 위한 라이브러리.
from dotenv import load_dotenv                                              # .env 파일에서 환경 변수 로드.
from langchain_text_splitters import RecursiveCharacterTextSplitter         # 긴 텍스트를 문단, 문장 등으로 잘게 나누는 도구.
from langchain.vectorstores import FAISS                                    # FAISS를 활용, 검색과 유사도 계산에 사용하는 벡터 데이터베이스를 가져옴.
from langchain_core.output_parsers import StrOutputParser                   # 출력 데이터를 문자열로 파싱하는 도구를 가져옴.
from langchain_core.prompts import PromptTemplate                           # 프롬프트 템플릿을 생성하고 관리하기 위한 도구.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings                   # OpenAI의 ChatGPT API를 사용하기 위한 인터페이스를 가져옴.
from langchain_community.chat_message_histories import ChatMessageHistory   # 대화 기록을 관리하는 도구. (채팅 메시지 히스토리)
from langchain_core.runnables.history import RunnableWithMessageHistory     # 대화 기록을 기반으로 작업을 처리하는 실행 도구.
from operator import itemgetter                                             # 리스트나 딕셔너리에서, 데이터의 특정 키나 항목을 쉽게 가져오기 위한 유틸리티 함수.
from langchain_core.documents import Document                               # 문서 데이터를 구조화하고 관리하기 위한 클래스를 가져옴.
from langchain_huggingface import HuggingFaceEmbeddings                     # HuggingFace 모델 활용한 임베딩 생성 기능을 가져옴.

# api key, env파일에서 로드.
load_dotenv(dotenv_path='key.env')
api_key = os.getenv("OPENAI_API_KEY")

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

data = load_job_data('jobdata') # Document 객체들의 리스트 반환.

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

retriever, embeddings, vectorstore = create_retriever(data)

# 프롬프트 작성.
prompt = PromptTemplate.from_template(
    """You are an expert assistant for job search and recommendation. 
    Always provide well-structured answers in Korean, including relevant data points.
    Avoid unnecessary elaboration.

    # 이전 대화 기록:
    {chat_history}

    # 질문:
    {question}

    # 검색된 정보:
    {info}

    # 답변:
    # 1. 요약: [질문에 대한 간단한 요약]
    2. 관련 정보: [추가로 제공할 수 있는 세부 정보]
    3. URL: [관련 링크 또는 소스]
    """
    )

def create_chain(retriever, prompt):
    # llm모델 gpt-4o으로 생성.
    """
    데이터를 검색하고 응답을 생성하는 체인을 생성하는 함수.
    데이터 검색(retriever), 질문 처리(prompt), 모델 응답 생성(llm) 과정을 연결.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

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

# 검색 도구(retriever)와 질문 생성 도구(prompt)를 활용해 체인 생성.
chain = create_chain(retriever, prompt)

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

# 체인을 기반으로 대화 기록을 포함한 RAG 객체 생성.
rag_with_chat = create_rag_with_chat(chain)

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

# 사용자 대화 내역 저장 및 성능 개선
def realtime_data_update_model(rag_with_chat, session_id, question):
    """
    RAG 체인과 사용자 대화를 관리하며 실시간으로 Vector DB 업데이트.
    AI 모델의 성능과 데이터 활용도를 지속적으로 향상.
    """
    # RAG 모델 실행 / 질문 처리, 응답 생성.
    response = rag_with_chat.invoke(
        {"question": question,"session_ids": session_id},   # 입력 데이터.
        config={"configurable": {"session_id": session_id}} # 세션별 설정.
        )

    # 대화 내용 저장
    session_history = get_session_history(session_id)   # 세션 기록 가져오기.
    session_history.add_user_message(question)          # 사용자의 질문 추가.
    session_history.add_ai_message(response)            # AI 응답 추가.

    # Vector DB 업데이트
    save_chat_in_vectorstore(session_id, question, response, vectorstore, embeddings)

    return response

# 코드 테스트용.
if __name__ == "__main__":
    """
    사용자 입력을 기반으로 RAG 모델을 실시간 테스트하는 루프.
    'exit' 입력 시 프로그램 종료.
    """
    while True:
        session_id = "jungseok"              #테스트용 ID.
        question = input("질문 입력하세요: ") #사용자 질문 입력.  
        if question == "exit":               #루프 종료를 위한 명령.  
            break
        
        # 입력된 질문을 기반으로 RAG 모델 실행 및 응답 생성
        response = realtime_data_update_model(rag_with_chat, session_id, question)

        print(f"답: {response}")