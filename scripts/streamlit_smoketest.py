"""Headless smoke test for the Streamlit app (``streamlit.testing.v1.AppTest``)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    os.environ.pop("PARAMISP_WEIGHTS_DIR", None)
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "streamlit_app" / "streamlit_app.py"), default_timeout=180)
    at.run()

    errors = [repr(e.value) for e in at.exception]
    assert not errors, f"script raised: {errors}"
    assert not at.error, [e.value for e in at.error]
    assert at.title and at.title[0].value.startswith("ParamISP")

    print("tabs:", [t.label for t in at.tabs])
    print("buttons:", [b.label for b in at.button])
    print("uploaders:", len(at.get("file_uploader")))
    print("STREAMLIT SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
