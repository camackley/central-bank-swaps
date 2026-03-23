// Anti-detection: remove the webdriver property that signals browser automation
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
