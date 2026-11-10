"""Which Google account is the tool's own Chrome profile signed into?"""
import re

from backend.app import chrome_profiles as cp

manager = cp.ChromeProfileManager(sessions_dir='data/chrome-profile-sessions')
with manager.launch_profile('84e29f179318690d', headless=True) as session:
    page = session.page
    page.goto('https://myaccount.google.com/', wait_until='domcontentloaded', timeout=60000)
    page.wait_for_timeout(4000)
    print('url:', page.url)
    text = page.locator('body').inner_text()
    emails = sorted(set(re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', text)))[:5]
    print('emails on the page:', emails)
    print('body[:200]:', text[:200].replace('\n', ' | '))
