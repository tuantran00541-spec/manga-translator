; Manga Translator setup for Windows: installs with install.ps1, then the app opens from its desktop icon.
; Built by .github/workflows/desktop-setup.yml, which passes the version from run.py.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6C7F0E52-3B8A-4E59-9D1B-7A3C2F1E8B40}
AppName=Manga Translator
AppVersion={#AppVersion}
AppPublisher=Manga Translator
AppPublisherURL=https://github.com/tuantran00541-spec/manga-translator
DefaultDirName={localappdata}\manga-translator\setup
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename=MangaTranslator-Setup
OutputDir=..\dist
SetupIconFile=..\app\static\favicon.ico
UninstallDisplayIcon={app}\favicon.ico
UninstallDisplayName=Manga Translator
WizardStyle=modern
Compression=lzma2
SolidCompression=yes

[Messages]
WelcomeLabel2=Cài Manga Translator vào máy này.%n%nLần đầu cài sẽ tải Python, thư viện và model, mất 5–15 phút tùy mạng; cài xong chiếm khoảng 1,6 GB. Một cửa sổ sẽ hiện từng bước đang làm.
FinishedLabel=Đã cài xong. Bấm đúp biểu tượng Manga Translator trên desktop để mở app.

[Files]
Source: "..\install.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\app\static\favicon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Run]
Filename: "{userdesktop}\Manga Translator.lnk"; Description: "Mở Manga Translator"; Flags: postinstall shellexec nowait skipifsilent skipifdoesntexist

[Code]
var
  InstallFailed: Boolean;

function InstallHome: String;
begin
  Result := ExpandConstant('{localappdata}\manga-translator');
  if not DirExists(Result + '\app') then
    Result := ExpandConstant('{commonappdata}\manga-translator');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Code: Integer;
  Pause, Command: String;
begin
  if CurStep <> ssPostInstall then
    Exit;
  Pause := '';
  if not WizardSilent then
    Pause := ' Read-Host ''Cai dat chua xong. Nhan Enter de dong cua so nay'';';
  // The installer prints each step; a failure keeps the window open so its reason can be read.
  Command := '-NoProfile -ExecutionPolicy Bypass -Command "try { & ''' + ExpandConstant('{app}\install.ps1') +
    ''' } catch { Write-Host $_ -ForegroundColor Red;' + Pause + ' exit 1 }"';
  if (not Exec('powershell.exe', Command, '', SW_SHOW, ewWaitUntilTerminated, Code)) or (Code <> 0) then
  begin
    InstallFailed := True;
    SuppressibleMsgBox('Cài đặt chưa xong (mã lỗi ' + IntToStr(Code) + '). Kiểm tra mạng rồi chạy lại file cài.',
      mbError, MB_OK, IDOK);
  end;
end;

function GetCustomSetupExitCode: Integer;
begin
  Result := 0;
  if InstallFailed then
    Result := 1;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Home, App: String;
  KeepChapters: Boolean;
begin
  if CurUninstallStep <> usUninstall then
    Exit;
  Home := InstallHome;
  App := Home + '\app';
  DeleteFile(ExpandConstant('{userdesktop}\Manga Translator.lnk'));
  DeleteFile(ExpandConstant('{userprograms}\Manga Translator.lnk'));
  KeepChapters := DirExists(App + '\data') and (SuppressibleMsgBox(
    'Giữ lại các chương đã dịch (thư mục ' + App + '\data)?', mbConfirmation, MB_YESNO, IDYES) = IDYES);
  if KeepChapters then
    RenameFile(App + '\data', Home + '\chapters');
  DelTree(App, True, True, True);
  DelTree(Home + '\python', True, True, True);
  DelTree(Home + '\uv', True, True, True);
  DelTree(Home + '\uv-cache', True, True, True);
  DelTree(Home + '\bin', True, True, True);
  DelTree(Home + '\paddlex', True, True, True);
end;
