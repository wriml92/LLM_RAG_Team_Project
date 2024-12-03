import requests
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from fake_useragent import UserAgent
import random
import time

'''필독
이 데이터 수집 파이썬 파일에는 사람인 api_key를 활용해야함.
하지만 api_key를 담은 key.env은 보안을 위해 커밋되지 않은 상태로 배포됨.
따라서 이 파일로 데이터를 원활하게 수집하려면 사람인 api의 승인 요청을 받아 개인 key.env 파일을 생성해 api-key의 환경변수를 만들어 실행해야함.
'''

load_dotenv(dotenv_path='key.env')

# 사람인 API 정보
api_key = os.getenv("API_KEY") # 발급받은 API 키를 여기에 입력하세요
url = "https://oapi.saramin.co.kr/job-search"  # 채용공고 API 기본 URL

# 데이터 저장 파일
download_folder = './jobdata'
os.makedirs(download_folder, exist_ok=True)

# 요청 파라미터 설정
params = {
    "access-key": api_key,
    "keywords": "python",  # 검색 키워드 (예: 'python')
    "count": 10,           # 한 번에 가져올 공고 수
    "start": 30             # 시작 페이지
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
    '''
    API로 호출받은 채용공고 JSON파일을 리스트에 텍스트 형태로 정제시킴.

    data - API로 받은 채용공고 JSON 데이터들.
    '''
    refined_data = []

    # Json파일에 jobs = {} 안에있는 job = [] 안의 요소들을 jobs로 지정.
    jobs = data.get("jobs", {}).get("job", [])

    for job in jobs:
        # 채용공고 하나당 데이터 프레임
        refined_job = {
            "title": job["position"]["title"],
            "company": job["company"]["detail"]["name"],
            "industry": job["position"]["industry"]["name"],
            "location": job["position"]["location"]["name"].replace("&gt;", ">"),  # HTML 엔티티 처리
            "job_type": job["position"]["job-type"]["name"],
            "categories": job["position"]["job-code"]["name"].split(","),  # 카테고리를 리스트로 변환
            "experience_level": job["position"]["experience-level"]["name"],
            "education_level": job["position"]["required-education-level"]["name"],
            "salary": job["salary"]["name"],
            "url": job["url"],
            "company_url": job["company"]["detail"]["href"],
            "expiration_date": datetime.fromtimestamp(int(job["expiration-timestamp"])).strftime('%Y-%m-%d')  # 타임스탬프 변환
        }
        refined_data.append(refined_job) # 정제한 채용공고 하나씩 리스트에 추가.

    return refined_data

def crawling_company_info(url):
    '''
    셀레니움으로 company_url을 통해 회사정보를 크롤링하는 함수, driver가 실행이 안될 시, 

    chromedriver_path = "chromedriver의 위치 경로"
    service = Service(chromedriver_path)

    driver = webdriver.Chrome(service=service, options=options)

    위와 같이 코드 변형 후 실행 시 오류 해결 가능성 있음.
    '''
    user_agent = UserAgent().random # 랜덤 유저에이전트 설정.
    options = Options()
    options.add_argument(f'user-agent={user_agent}')

    driver = webdriver.Chrome(options=options) # 유저에이전트 랜덤 옵션 적용.

    try:
        driver.get(url) # 웹드라이버로 url 접속

        time.sleep(random.uniform(2, 5))    # 요청 후 랜덤 시간 (2~5초) 대기

        # WebElement를 수집
        info_names = driver.find_elements(By.CLASS_NAME, 'company_summary_desc')
        infos = driver.find_elements(By.CLASS_NAME, 'company_summary_tit')

        content = {}

        '''
        WebElement에서 얻어온 데이터 구조가 (설립일자:연차)인데 다음 인덱스의 정보는 
        (사원수:0명), (매출액:0원)와 같은 형식이라서 (설립일자:0일), (연차:0년) 형태로 바꾸기 위해 아래 코드를 구성.
        '''
        if len(info_names) == len(infos):
            content['설립 일자'] = info_names[0].text if len(info_names) > 0 else "정보 없음"
            content['연차'] = infos[0].text if len(infos) > 0 else "정보 없음"
            for name, info in zip(info_names[1:], infos[1:]):
                key = name.text
                value = info.text
                content[key] = value

        # 얻어온 정보의 내용이 비어있다면.
        if len(info_names) == 0:
            content["회사 정보"] = "정보 없음"

    except Exception as e:
        print(f"Error occurred while fetching URL: {url}, {e}")
        content = {"회사 정보": "크롤링 실패"}
    finally:
        driver.quit()  # ChromeDriver 종료

    return content

def merge_data(data):
    '''
    API에서 얻어온 정보를 정제한 데이터와, 동적 데이터 크롤링으로 얻어온 정보를 merge하는 함수.
    '''
    for i in data:
        url = i["company_url"]  # API에서 얻어온 데이터의 회사 정보 링크

        if url:
            print(f"Fetching content from: {url}")
            # 회사 정보링크에서 회사 정보 동적 크롤링.
            content = crawling_company_info(url)
            # 기존 데이터에 "content" 키 추가하고 크롤링한 데이터 삽입.
            i["content"] = content
        else:
            print("No url found for this job.")
            i["content"] = {"회사 정보": "URL 없음"}
    
    return data

# 정제된 데이터를 json 파일로 jobdata 폴더에 저장..
refined_data = refine_job_data(data)
merged_data = merge_data(refined_data)

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
file_name = f'data_{current_time}.json'
file_path = os.path.join(download_folder, file_name)

with open(file_path, 'w', encoding="utf-8") as file:
    json.dump(merged_data, file, ensure_ascii=False, indent=4)