import requests
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

load_dotenv(dotenv_path='.env')

# 사람인 API 정보
api_key = os.getenv("API_KEY") # 발급받은 API 키를 여기에 입력하세요
url = "https://oapi.saramin.co.kr/job-search"  # 채용공고 API 기본 URL

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
    refined_data = []

    jobs = data.get("jobs", {}).get("job", [])

    for job in jobs:
        # 정제시킬 데이터 프레임
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
        refined_data.append(refined_job)

    return refined_data

def crawling_company_info(url):
    from selenium.webdriver.chrome.service import Service

    chromedriver_path = "/opt/homebrew/bin/chromedriver"  # ChromeDriver 경로
    service = Service(chromedriver_path)

    options = Options()
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(url)
    time.sleep(3)

    # WebElement를 수집
    info_names = driver.find_elements(By.CLASS_NAME, 'company_summary_desc')
    infos = driver.find_elements(By.CLASS_NAME, 'company_summary_tit')

    content = {}

    # WebElement 객체의 text 속성만 추출
    if len(info_names) == len(infos):
        content['설립 일자'] = info_names[0].text if len(info_names) > 0 else "정보 없음"
        content['연차'] = infos[0].text if len(infos) > 0 else "정보 없음"
        for name, info in zip(info_names[1:], infos[1:]):
            key = name.text
            value = info.text
            content[key] = value

    if len(info_names) == 0:
        content["회사 정보"] = "정보 없음"

    driver.quit()  # ChromeDriver 종료
    return content

def merge_data(data):
    for i in data:
        url = i["company_url"]

        if url:
            print(f"Fetching content from: {url}")
            # 크롤링 수행
            content = crawling_company_info(url)
            # 기존 데이터에 "content" 키 추가
            i["content"] = content
        else:
            print("No url found for this job.")
            i["content"] = {"회사 정보": "URL 없음"}
    
    return data

# 정제된 데이터 출력.
refined_data = refine_job_data(data)
merged_data = merge_data(refined_data)

file_path = './jobdata/data.json'
with open(file_path, 'w', encoding="utf-8") as file:
    json.dump(merged_data, file, ensure_ascii=False, indent=4)

print(data)  # API 호출 결과 출력