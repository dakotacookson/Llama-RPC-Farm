# Windows SmartScreen & code signing

If Windows shows:

> Windows protected your PC — Microsoft Defender SmartScreen prevented an
> unrecognized app from starting.

that is SmartScreen reacting to an **unsigned** installer, not to a virus
detection. The rolling build logs say exactly what happened:

```
No Windows signing certificate configured - unsigned build.
Installer left unsigned (no cert).
```

There is no code change that removes that warning. Windows only trusts a
binary after it is Authenticode-signed by a certificate that chains to a
trusted CA, and (for newer OV certificates) after that publisher builds
reputation.

## Run it today (unsigned build)

Only do this if you downloaded the file from this project's GitHub
Releases and its SHA-256 matches the published `.sha256` file.

1. In the SmartScreen dialog click **More info** → **Run anyway**.
2. Or remove the download mark first, then run:

   ```powershell
   Unblock-File .\llama-rpc-farm__windows-x64-setup.exe
   ```

3. Verify the hash before running:

   ```powershell
   Get-FileHash .\llama-rpc-farm__windows-x64-setup.exe -Algorithm SHA256
   ```

   Compare it with the `.sha256` asset in the same release.

Do **not** disable SmartScreen or Defender to work around this.

## Permanent fix: sign the build

The release workflow already supports signing. It signs both inner
executables and the Inno Setup installers with SHA-256 + a timestamp.

### 1. Get a code-signing certificate

| Option | SmartScreen result | Notes |
|---|---|---|
| **EV code-signing cert** | Trusted immediately | Hardware/HSM token or cloud signing; ~$300+/yr |
| **Microsoft Trusted Signing** (Azure) | Trusted after validation | ~$10/month; eligibility rules apply (business/individual history) |
| **OV code-signing cert** | Warns until reputation accrues | Exportable `.pfx` usually; ~$100–300/yr |
| **SignPath Foundation** | Trusted once approved | Free for qualifying open-source projects |
| Self-signed certificate | **Still warns** | Only useful for machines where you install the cert yourself |

This repository is Apache-2.0, so SignPath Foundation is often the
cheapest route. EV / Trusted Signing is the only option that removes the
warning for every user immediately.

### 2. Add GitHub secrets (`.pfx` path, already wired)

Export the certificate as a password-protected `.pfx` that includes the
private key, then add two repository secrets:

- `WINDOWS_CERT_PFX_BASE64` — base64 of the `.pfx`
- `WINDOWS_CERT_PASSWORD`

```powershell
# base64 the PFX for the GitHub secret
[Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\codesign.pfx"))
```

The next push to `main` (or a manual `workflow_dispatch`) signs:

- `dist\llama-rpc-farm.exe`
- `dist\llama-peer-agent.exe`
- `dist\*setup.exe`

### 3. Verify locally

```powershell
Get-AuthenticodeSignature .\llama-rpc-farm__windows-x64-setup.exe |
  Format-List Status, StatusMessage, SignerCertificate
```

`Status` must be `Valid`.

### 4. Hardware-token / cloud certificates

EV certificates on a USB token cannot be exported to a `.pfx`. Use the
vendor's cloud-signing integration instead (Azure Trusted Signing,
DigiCert KeyLocker, SSL.com eSigner, SignPath, …) and replace/extend the
`signtool sign` step in `.github/workflows/rolling-main.yml` with that
provider's `/dlib` + `/dmdf` (or equivalent) invocation.

## Notes

- Every release must be signed with a **timestamp** (`/tr … /td sha256`);
  otherwise the signature expires with the certificate.
- A new OV certificate starts with no SmartScreen reputation. Signing the
  same publisher name consistently builds it over time; EV and Trusted
  Signing skip that wait.
- The installer and the exe inside it must both be signed. Signing only
  the installer is not enough for every antivirus/browser flow.
