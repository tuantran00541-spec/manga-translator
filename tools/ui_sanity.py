from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        page.on("console", lambda msg: errors.append(f"[{msg.type}] {msg.text}") if msg.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
        page.goto("http://127.0.0.1:8123/", wait_until="networkidle")
        page.wait_for_timeout(800)
        info = page.evaluate("""() => ({
          renderEditorIsCanonical: (window.renderEditor ? window.renderEditor.toString().includes('translation-workspace') : false),
          bootstrap: !!document.querySelector('#load-btn'),
          chapterInput: !!document.querySelector('#chapter-url'),
          upload: !!document.querySelector('#upload-dropzone'),
          hasToast: !!document.querySelector('#toast-container')
        })""")
        print("INFO:", info)
        print("ERRORS:", errors if errors else "none")
        browser.close()

if __name__ == "__main__":
    main()
