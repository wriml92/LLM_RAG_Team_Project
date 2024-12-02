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
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# api key, env파일에서 로드.
load_dotenv(dotenv_path='key.env')
api_key = os.getenv("OPENAI_API_KEY")

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

data = load_job_data('jobdata')

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

chain = create_chain(retriever, prompt)

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

rag_with_chat = create_rag_with_chat(chain)

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

# 사용자 대화 내역 저장 및 성능 개선
def realtime_data_update_model(rag_with_chat, session_id, question):
    """
    RAG 체인과 사용자 대화를 관리하며 실시간으로 Vector DB 업데이트.
    """
    # RAG 모델 실행
    response = rag_with_chat.invoke({"question": question,"session_ids": session_id},config={"configurable": {"session_id": session_id}})

    # 대화 내용 저장
    session_history = get_session_history(session_id)
    session_history.add_user_message(question)
    session_history.add_ai_message(response)

    # Vector DB 업데이트
    save_chat_in_vectorstore(session_id, question, response, vectorstore, embeddings)

    return response

# 코드 테스트용.
if __name__ == "__main__":
    while True:
        session_id = "jungseok"
        question = input("질문 입력하세요: ")
        if question == "exit":
            break
        
        response = realtime_data_update_model(rag_with_chat, session_id, question)

        print(f"답: {response}")