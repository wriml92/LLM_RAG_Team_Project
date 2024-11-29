from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# ChromeDriver 경로
service = Service("/opt/homebrew/bin/chromedriver")  # 올바른 경로로 수정

# 옵션 설정
options = Options()

# ChromeDriver 실행
driver = webdriver.Chrome(service=service, options=options)

# 테스트용 URL 접속
driver.get("https://www.google.com")
print("ChromeDriver 실행 성공!")
driver.quit()