"""
SSL / network utility — apply at the top of every module that calls HuggingFace or external HTTPS.
Handles corporate proxies that intercept TLS with their own CA.
"""

import os
import ssl
import urllib3


def patch_ssl():
    """Disable SSL verification globally for all HTTPS calls."""
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""

    # huggingface_hub >= 0.22 custom session
    try:
        import requests
        from huggingface_hub import configure_http_backend

        def _no_verify() -> requests.Session:
            s = requests.Session()
            s.verify = False
            return s

        configure_http_backend(backend_factory=_no_verify)
    except (ImportError, AttributeError):
        pass


def use_hf_cache_only():
    """
    Tell HuggingFace libraries to never call home — use whatever is cached.
    Call this AFTER the model has been downloaded at least once.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
