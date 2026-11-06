# Inno Setup script for the Llama RPC Farm app (Windows).
#define MyAppVersion GetEnv('APP_VERSION')
#define MyBuildVersion GetEnv('BUILD_VERSION')
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef MyBuildVersion
  #define MyBuildVersion MyAppVersion
#endif

[Setup]
AppId={{A3E1B7C2-5F4D-4E8A-9B6C-2D1E0F7A8B91}
AppName=Llama RPC Farm
AppVersion={#MyAppVersion}
AppPublisher=Llama RPC Farm contributors
DefaultDirName={autopf}\LlamaRPCFarm
DefaultGroupName=Llama RPC Farm
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=llama-rpc-farm_{#MyBuildVersion}_windows-x64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=Llama RPC Farm

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "..\dist\llama-rpc-farm.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Llama RPC Farm"; Filename: "{app}\llama-rpc-farm.exe"
Name: "{autodesktop}\Llama RPC Farm"; Filename: "{app}\llama-rpc-farm.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\llama-rpc-farm.exe"; Parameters: "--install-service"; Flags: runhidden waituntilterminated; StatusMsg: "Registering login auto-start and firewall rules..."
Filename: "{sys}\schtasks.exe"; Parameters: "/Run /TN LlamaRPCFarm"; Flags: runhidden

[UninstallRun]
Filename: "{app}\llama-rpc-farm.exe"; Parameters: "--uninstall-service"; Flags: runhidden waituntilterminated
Filename: "{sys}\schtasks.exe"; Parameters: "/Delete /TN LlamaRPCFarm /F"; Flags: runhidden
