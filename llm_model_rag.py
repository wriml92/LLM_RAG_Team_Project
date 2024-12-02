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

# api key, env파일에서 로드.
load_dotenv(dotenv_path='key.env')
api_key = os.getenv("OPENAI_API_KEY")

def load_job_data(job):
    '''job이라는 json 파일의 내용을 기본 문서인 document로 변형'''
    data = []
    for info in job:
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
        data.append(doc)
    return data

# json파일 불러오기.
with open("jobdata/data.json", "r", encoding="utf-8") as file:
    job_data = json.load(file)

# json파일 documnet 형태로 변형.
data = load_job_data(job_data)

# document chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(data)

# 임베딩 생성 및 벡터 DB 생성, 저장
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 리트리버(검색기) 생성. 문서의 정보 검색 및 생성
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

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

# 대화 세션 기록
chat = {}

def get_session_history(session_ids):
    '''세션 ID 기반으로 세션 기록을 가져오는 함수'''
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in chat:  # 세션 ID가 chat에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 chat에 저장
        chat[session_ids] = ChatMessageHistory()
    return chat[session_ids]

rag_with_chat = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
)

# 코드 테스트용.
if __name__ == "__main__":
    user_input = input("질문 입력하세요: ")
    print("result: ", rag_with_chat.invoke(
    {"question": user_input},
    config={"configurable": {"session_id": "jungseok"}})
    )
    user_input_2 = input("질문 입력하세요: ")
    print("result: ", rag_with_chat.invoke(
    {"question": user_input_2},
    config={"configurable": {"session_id": "jungseok"}})
    )