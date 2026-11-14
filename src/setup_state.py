"""First-run setup walkthrough: test override.

Normal installs use the persisted `setup_wizard_done` marker. For manual
testing you can force the walkthrough even when an admin and a completed
setup already exist:

  * edit FORCE_SETUP_WIZARD to True in this file, or
  * launch the app with FARM_SETUP_WIZARD_FORCE=1

Leave the constant False in committed code. The override only affects the
walkthrough prompt; it never changes the stored marker or user accounts.
"""
import os


# Test-only source switch. Toggle to True, restart the UI, and every admin
# boot opens the storage/engines walkthrough again.
FORCE_SETUP_WIZARD = False


def force_enabled() -> bool:
    if FORCE_SETUP_WIZARD:
        return True
    return str(os.environ.get("FARM_SETUP_WIZARD_FORCE") or "").strip().lower() in (
        "1", "true", "yes", "on")
