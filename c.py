from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# Настройки браузера
options = Options()
options.add_argument('--headless')  # Без открытия окна
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Запуск браузера
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Открытие страницы
url = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
driver.get(url)

# Подождать загрузку страницы (важно!)
time.sleep(5)

# Получить текст
text = driver.find_element("tag name", "body").text

# Сохранить
with open("constitution.txt", "w", encoding="utf-8") as f:
    f.write(text)

driver.quit()
print("✅ Конституция успешно сохранена в constitution.txt")
