import requests
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from bs4 import BeautifulSoup

load_dotenv(dotenv_path='key.env')

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

def crawling_job_info(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')

        content = {}
        return content

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return {"description": "크롤링 실패", "address": "크롤링 실패"}

def crawling_company_info(url):
    try:
        # HTTP request
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')

        content = {
            "company_introduce": soup.select_one(".txt").get_text(strip=True) if soup.select_one(".txt") else "Couldn't find information",
            "company_history": soup.select_one(".history_txt").get_text(strip=True) if soup.select_one(".history_txt") else "Couldn't find information"
        }
        return content
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return {"description": "크롤링 실패", "address": "크롤링 실패"}
    
def merge_data(data):
    for i in data:
        href = i["company_url"]

        if href:
            print(f"Fetching content from: {href}")
            # 크롤링 수행
            content = crawling_company_info(href)
            # 기존 데이터에 "content" 키 추가
            i["content"] = content
        else:
            print("No href found for this job.")
            i["content"] = {"description": "URL 없음", "address": "URL 없음"}
    
    return data

# 정제된 데이터 출력.
refined_data = refine_job_data(data)
merged_data = merge_data(refined_data)

file_path = './jobdata/data.json'
with open(file_path, 'w', encoding="utf-8") as file:
    json.dump(merged_data, file, ensure_ascii=False, indent=4)