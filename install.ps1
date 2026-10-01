# Manga Translator installer for Windows. Paste into PowerShell:
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/tuantran00541-spec/manga-translator/main/install.ps1 | iex"
# From a clone, install.bat runs this file and installs that clone in place.
# Kept ASCII-only: Windows PowerShell 5.1 misreads UTF-8 files without a BOM.

& {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'  # the progress bar makes downloads many times slower in PowerShell 5.1
    $repo = 'tuantran00541-spec/manga-translator'
    $ref = if ($env:MANGA_REF) { $env:MANGA_REF } else { 'main' }
    $uvVersion = '0.12.20'
    $uvSha256 = '95f9bc30fbb3574d276e28ac4a6de932d25153645853d13da8c21eec3bc88d06'

    function Say($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }

    if ($PSVersionTable.PSVersion.Major -lt 5) { throw 'Can PowerShell 5 tro len (Windows 10/11 da co san).' }
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    if ($arch -ne 'AMD64') { throw "Chi ho tro Windows 64-bit (x64); may nay la $arch." }

    # Everything lives in one folder; a user folder with accents breaks PaddleOCR, so fall back to ProgramData.
    $home_ = if ($env:MANGA_HOME) { $env:MANGA_HOME } else { Join-Path $env:LOCALAPPDATA 'manga-translator' }
    if (-not $env:MANGA_HOME -and $home_ -match '[^\x00-\x7F]') { $home_ = Join-Path $env:ProgramData 'manga-translator' }
    New-Item -ItemType Directory -Force -Path $home_ | Out-Null

    Say "Chuan bi uv $uvVersion (trinh cai Python)"
    $uvDir = Join-Path $home_ 'uv'
    $uv = Join-Path $uvDir 'uv.exe'
    $haveUv = (Test-Path $uv) -and ((& $uv --version) -match [regex]::Escape($uvVersion))
    if (-not $haveUv) {
        $zip = Join-Path ([IO.Path]::GetTempPath()) ("uv-" + [guid]::NewGuid() + '.zip')
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
        if ((Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower() -ne $uvSha256) { Remove-Item $zip; throw 'Tai uv bi loi (sai ma kiem tra), hay chay lai.' }
        Expand-Archive -Force -Path $zip -DestinationPath $uvDir
        Remove-Item $zip
    }

    $work = $null
    if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot 'run.py'))) {
        $source = $PSScriptRoot
        $target = $PSScriptRoot
    } else {
        Say "Tai ma nguon ($ref)"
        $work = Join-Path ([IO.Path]::GetTempPath()) ("manga-translator-" + [guid]::NewGuid())
        New-Item -ItemType Directory -Path $work | Out-Null
        $archive = Join-Path $work 'source.zip'
        Invoke-WebRequest -UseBasicParsing -Uri "https://codeload.github.com/$repo/zip/refs/heads/$ref" -OutFile $archive
        Expand-Archive -Path $archive -DestinationPath $work
        $source = (Get-ChildItem -Directory $work | Select-Object -First 1).FullName
        $target = Join-Path $home_ 'app'
    }

    # Settings for this install only; the user's own session gets them back afterwards.
    $settings = @{
        UV_PYTHON_INSTALL_DIR = (Join-Path $home_ 'python')
        UV_PYTHON_PREFERENCE = 'only-managed'
        # The download cache only speeds up a re-run, so it lives here and is removed after a good install.
        UV_CACHE_DIR = (Join-Path $home_ 'uv-cache')
        MANGA_UV = $uv
        MANGA_HOME = $home_
        MANGA_REF = $ref
        PYTHONUTF8 = '1'
    }
    $saved = @{}
    foreach ($name in $settings.Keys) {
        $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $settings[$name], 'Process')
    }
    try {
        & $uv run --no-project --python 3.12 (Join-Path $source 'scripts\install.py') --target $target
        if ($LASTEXITCODE -ne 0) { throw 'Cai dat chua xong, xem loi o tren roi chay lai lenh cai.' }
        Remove-Item -Recurse -Force (Join-Path $home_ 'uv-cache') -ErrorAction SilentlyContinue
    } finally {
        foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
        if ($work -and $work.StartsWith([IO.Path]::GetTempPath())) { Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue }
    }

    # This window can use the command at once; new windows read it from the user Path.
    $bin = Join-Path $home_ 'bin'
    if (-not (($env:Path -split ';') -contains $bin)) { $env:Path = "$bin;$env:Path" }
    Say 'Xong. Go lenh:  manga'
}
