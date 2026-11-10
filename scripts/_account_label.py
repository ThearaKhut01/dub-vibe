"""Can the profile's Google account be read from Gemini's own header labels?"""
from backend.app import chrome_profiles as cp

manager = cp.ChromeProfileManager(sessions_dir='data/chrome-profile-sessions')
with manager.launch_profile('84e29f179318690d', headless=True) as session:
    page = session.page
    page.goto(cp.GEMINI_URL, wait_until='domcontentloaded', timeout=60000)
    page.wait_for_timeout(6000)
    print('labels:')
    for index in range(page.locator('[aria-label*="Google Account"]').count()):
        try:
            print('  -', page.locator('[aria-label*="Google Account"]').nth(index).get_attribute('aria-label'))
        except Exception as error:
            print('  - ERR', error)
    print('sign-out hrefs:')
    for index in range(min(page.locator('a[href*="SignOutOptions"]').count(), 3)):
        try:
            print('  -', page.locator('a[href*="SignOutOptions"]').nth(index).get_attribute('href')[:120])
        except Exception as error:
            print('  - ERR', error)
