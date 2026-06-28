from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
        headless=False,
        args=['--no-sandbox', '--disable-dev-shm-usage']
    )
    context = browser.new_context(viewport={'width': 1400, 'height': 900})
    page = context.new_page()

    print("Opening GoCharting...")
    page.goto("https://gocharting.com", timeout=30000)
    time.sleep(3)
    page.screenshot(path="/home/user/Vikram/sc1_home.png")
    print("Home page loaded:", page.title())

    browser.close()
