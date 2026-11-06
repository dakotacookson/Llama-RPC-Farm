# Inno Setup script for the Llama RPC Farm peer agent (Windows).
#define MyAppVersion GetEnv('APP_VERSION')
#define MyBuildVersion GetEnv('BUILD_VERSION')
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef MyBuildVersion
  #define MyBuildVersion MyAppVersion
#endif

[Setup]
AppId={{C7F2D9A4-8E3B-4A15-8D6E-6B4F1C2A9E73}
AppName=Llama RPC Farm Agent
AppVersion={#MyAppVersion}
AppPublisher=Llama RPC Farm contributors
DefaultDirName={autopf}\LlamaRPCFarmAgent
DisableProgramGroupPage=yes
DisableDirPage=auto
OutputDir=..\dist
OutputBaseFilename=llama-peer-agent_{#MyBuildVersion}_windows-x64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=Llama RPC Farm Agent

[Files]
Source: "..\dist\llama-peer-agent.exe"; DestDir: "{app}"; Flags: ignoreversion

[Run]
Filename: "{app}\llama-peer-agent.exe"; Parameters: "--install-service"; Flags: runhidden waituntilterminated; StatusMsg: "Registering login auto-start and firewall rules..."
Filename: "{sys}\schtasks.exe"; Parameters: "/Run /TN LlamaPeerAgent"; Flags: runhidden

[UninstallRun]
Filename: "{app}\llama-peer-agent.exe"; Parameters: "--uninstall-service"; Flags: runhidden waituntilterminated
Filename: "{sys}\schtasks.exe"; Parameters: "/Delete /TN LlamaPeerAgent /F"; Flags: runhidden
