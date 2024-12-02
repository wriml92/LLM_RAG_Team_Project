import requests                 # HTTP 요청을 보내고 응답을 받기 위한 라이브러리.
import os                       # 운영 체제 관련 기능을 사용을 위한 os 모듈 임포트.
import json                     # JSON 데이터 처리를 위한 라이브러리.
from dotenv import load_dotenv  # .env 파일에서 환경 변수 로드.
from datetime import datetime   # 날짜 및 시간을 처리를 위한 datetime 모듈 임포트.
from bs4 import BeautifulSoup   # HTML 문서의 데이터 추출을 위한 라이브러리.

load_dotenv(dotenv_path='key.env')

# 사람인 API 정보
api_key = os.getenv("API_KEY") # 발급받은 API 키를 여기에 입력하세요
url = "https://oapi.saramin.co.kr/job-search"  # 채용공고 API 기본 URL

# 'jobdata'라는 이름의 폴더를 현재 디렉토리에 생성. / 이미 해당 폴더가 존재한다면, 오류 없이 그대로 사용.
download_folder = './jobdata'
os.makedirs(download_folder, exist_ok=True)     

# 요청 파라미터 설정
params = {
    "access-key": api_key,
    "keywords": "python",  # 검색 키워드 (예: 'python')
    "count": 10,           # 한 번에 가져올 공고 수
    "start": 1             # 시작 페이지
}

# API 호출
response = requests.get(url, params=params)

# 결과 확인
if response.status_code == 200:
    data = response.json()
    print("API 호출 성공")
else:
    print("실패.")

def refine_job_data(data):
    #리스트 객체를 초기화함으로써, 데이터가 들어갈 리스트를 미리 생성.
    refined_data = []

    jobs = data.get("jobs", {}).get("job", [])  # 'jobs' 키에서 'job' 리스트 추출.

    for job in jobs:
        # 정제시킬 데이터 프레임
        refined_job = {
            "title": job["position"]["title"],                                     # 직책
            "company": job["company"]["detail"]["name"],                           # 회사 이름
            "industry": job["position"]["industry"]["name"],                       # 산업 분야
            "location": job["position"]["location"]["name"].replace("&gt;", ">"),  # HTML 엔티티 처리 / 엔티티를 사용함으로써 문법 오류를 피하고, 웹 브라우저가 문자 그대로 표현 가능.
            "job_type": job["position"]["job-type"]["name"],                       # 직무 유형
            "categories": job["position"]["job-code"]["name"].split(","),          # 카테고리를 리스트로 변환
            "experience_level": job["position"]["experience-level"]["name"],       # 경력
            "education_level": job["position"]["required-education-level"]["name"],# 요구 학력 수준
            "salary": job["salary"]["name"],                                       # 급여
            "url": job["url"],                                                     # 구직 URL
            "company_url": job["company"]["detail"]["href"],                       # 회사 URL
            "expiration_date": datetime.fromtimestamp(int(job["expiration-timestamp"])).strftime('%Y-%m-%d')  # 타임스탬프 변환 (만료 날짜.)
        }
        refined_data.append(refined_job) #위 정보들을 refined_job 리스트에 추가.

    return refined_data

def crawling_job_info(url):
    try:# 웹 요청에 사용할 사용자 에이전트 설정.
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"} 
        
        # 지정된 URL에 GET 요청을 보내고 응답을 받음.
        response = requests.get(url, headers=headers)
        
        # 응답 상태 코드가 200(성공)이 아닐 경우 예외 발생.
        response.raise_for_status()

        # HTML 파싱(Parsing) / HTML 문서를 읽고 구조를 분석하여, 웹 페이지를 브라우저가 이해하고 렌더링할 수 있도록 하는 과정.
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 구직 정보 관련 데이터를 담을 빈 딕셔너리 생성.
        content = {}
        return content
    
    # 요청 중 오류가 발생하면 에러 메시지를 출력, 실패 정보를 담은 딕셔너리 반환.
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return {"description": "크롤링 실패", "address": "크롤링 실패"}

def crawling_company_info(url):
    try:
        # HTTP 요청을 보내기 위한 사용자 에이전트 설정.
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}
        
        # 지정된 URL에 GET 요청을 보내고 응답을 받음.
        response = requests.get(url, headers=headers)
        
        # 응답 상태 코드가 200(성공)이 아닐 경우 예외 발생
        response.raise_for_status()

        # HTML 파싱(Parsing) / HTML 문서를 읽고 구조를 분석하여, 웹 페이지를 브라우저가 이해하고 렌더링할 수 있도록 하는 과정.
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 회사 소개 및 history 정보를 추출, 딕셔너리에 저장.
        content = {
            "company_introduce": soup.select_one(".txt").get_text(strip=True) if soup.select_one(".txt") else "Couldn't find information",
            "company_history": soup.select_one(".history_txt").get_text(strip=True) if soup.select_one(".history_txt") else "Couldn't find information"
        }
        return content
    
    # 요청 중 오류가 발생하면 에러 메시지를 출력, 실패 정보를 담은 딕셔너리 반환.
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return {"description": "크롤링 실패", "address": "크롤링 실패"}
    
def merge_data(data):
    for i in data:
        # 각 항목에서 'company_url' 값을 추출.
        href = i["company_url"]

        if href:
            print(f"Fetching content from: {href}")
            # 크롤링 수행
            content = crawling_company_info(href)
            # 기존 데이터에 "content" 키 추가
            i["content"] = content
        else:
            # 'company_url'이 없으면, 기본값을 가진 'content'를 추가
            print("No href found for this job.")
            i["content"] = {"description": "URL 없음", "address": "URL 없음"}
    
    return data

# 정제된 데이터 출력.
refined_data = refine_job_data(data)
merged_data = merge_data(refined_data)

# 저장할 파일 경로 설정
file_path = './jobdata/data.json'

# 파일을 쓰기 모드('w')로 열고, UTF-8 인코딩을 사용하여 데이터를 저장.
with open(file_path, 'w', encoding="utf-8") as file:
    # ensure_ascii=False: 한글을 포함, 모든 문자가 올바르게 저장되도록 설정.
    # indent=4: JSON 파일을 읽기 쉽게 들여쓰기 사용.
    json.dump(merged_data, file, ensure_ascii=False, indent=4)