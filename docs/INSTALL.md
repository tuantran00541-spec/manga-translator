# Install

Every way to install and run the app, and what the installer does.


## Requirements

- Python 3.12 (the installer brings its own)
- A CPU-capable machine, with about 5 GB free for the first install
- Chromium for Playwright URL ingestion and browser checks
- More RAM helps OCR and very large pages

The image path accepts up to **100,000,000 decoded pixels** per image.

## Setup and one-command install

**Windows:** download [MangaTranslator-Setup.exe](https://github.com/tuantran00541-spec/manga-translator/releases/latest/download/MangaTranslator-Setup.exe) and double-click it. It needs no admin rights; the first install takes 5–15 minutes and uses about 1.6 GB of disk. Uninstall it from Windows Settings → Apps; it asks whether to keep your translated chapters.

The setup is not code-signed yet, so Windows SmartScreen may show "Windows protected your PC". Click **More info → Run anyway** (Vietnamese Windows: **Thêm thông tin → Vẫn chạy**). The one-command install below does not go through SmartScreen.

Or with one command, nothing else to install first:

~~~powershell
# Windows: paste into PowerShell or cmd
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/tuantran00541-spec/manga-translator/main/install.ps1 | iex"
~~~

~~~bash
# Linux / macOS
curl -LsSf https://raw.githubusercontent.com/tuantran00541-spec/manga-translator/main/install.sh | sh
~~~

On Windows, double-click the **Manga Translator** icon the installer puts on the desktop and in the Start menu: the app opens in its own window, with no console. Elsewhere, or from any **new** cmd or terminal window:

~~~bash
manga           # starts the app and opens it in the browser (or just opens it if it already runs)
manga --window  # the same in its own window (Windows and macOS)
manga update    # gets the latest version; models, chapters and settings stay
manga --version
~~~

It installs the latest release, and `manga update` moves to the next one. To try the newest code instead, set `MANGA_REF=main` before the command (`$env:MANGA_REF='main'` in PowerShell).

The installer works in one folder (`%LOCALAPPDATA%\manga-translator` on Windows, `~/.local/share/manga-translator` elsewhere):

- downloads [uv](https://github.com/astral-sh/uv) (pinned, hash checked), which fetches Python 3.12 and installs the dependencies (no PyTorch);
- on Windows, installs the Microsoft Visual C++ runtime when it is missing or older than 14.40 (onnxruntime and PaddleOCR need it; Windows asks for admin once). `manga` checks it at every start, so a runtime removed later is put back instead of failing with WinError 126;
- downloads the LaMa model and `ctd_seg.onnx` (hash checked, resumes broken downloads); `ctd_seg.onnx` is prebuilt by `scripts/export_ctd_onnx.py` on the [models-v1](https://github.com/tuantran00541-spec/manga-translator/releases/tag/models-v1) release, and is built locally only if that download fails;
- adds a `manga` command to the user PATH and, on Windows, the desktop and Start menu icons.

A Windows user folder with accents (for example `C:\Users\Nguyễn`) breaks PaddleOCR, so the installer then uses `C:\ProgramData\manga-translator`. From a clone, `install.bat` / `./install.sh` install that clone in place.

To uninstall, delete that folder, the icons and the `manga` command (`%LOCALAPPDATA%\manga-translator\bin` on Windows, `~/.local/bin/manga` elsewhere), then remove its line from the user Path or shell start-up file.

Manual install (Python 3.10–3.12):

~~~bash
python -m venv .venv
# Linux / macOS: source .venv/bin/activate    Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
playwright install chromium
~~~

## Models

Place these files in `models/`:

| File | Purpose |
| --- | --- |
| `kiuyha_text_1280.onnx` | Text detection: Kiuyha/Manga-Bubble-YOLO boxes (in the repository) |
| `ctd_seg.onnx` | Letter masks: comic-text-detector, exported fast and mask-only |
| `lama-manga-dynamic.onnx` | Preferred inpainting backend |
| `lama.onnx` | Fixed-resolution fallback |

The LaMa files are too large for Git and must be downloaded. Build `ctd_seg.onnx` from the released [comictextdetector.pt.onnx](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3): save it as `.cache/comictextdetector.pt.onnx` and run `python scripts/export_ctd_onnx.py` (needs `torch`, `onnx` and `onnx2torch` once; the app itself only needs onnxruntime). The **Kiuyha ONNX export** workflow re-exports [Kiuyha/Manga-Bubble-YOLO](https://huggingface.co/Kiuyha/Manga-Bubble-YOLO) and checks the ONNX boxes against the original model.

## Run

~~~bash
manga             # after the installer
python run.py     # from an activated environment
~~~

Open `http://127.0.0.1:8000`.

## Browser extension

Sites behind Cloudflare usually refuse Playwright. The extension in [`extension/`](../extension) takes the images from the chapter you have open in Chrome or Edge instead:

1. Open `chrome://extensions` (or `edge://extensions`) and turn on **Developer mode**.
2. Click **Load unpacked** and pick the `extension` folder: in the app folder for a source checkout, or `%LOCALAPPDATA%\manga-translator\app\extension` after the Windows setup.
3. With the app running, open a chapter, then click the Manga Translator icon and **Gửi chương vào app**.

The same steps, with a button that opens the folder, are in the app under **Nhập nội dung → Dùng tiện ích trình duyệt**. After `manga update`, click **Reload** on the extension card; the extension says so after a send when its copy is older than the app's.

It scrolls the page so lazy images load, takes the wide images and canvases in reading order, converts formats the app does not read (GIF, AVIF) to PNG, and opens the new chapter in the app. It only talks to the app on this computer.

## Docker

~~~bash
docker compose up --build
~~~

The image expects the model files through the mounted `./models` directory.
