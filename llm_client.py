

import subprocess
import shutil
import time
import requests

OLLAMA_BASE_URL = "http://localhost:11434"


def is_ollama_running() -> bool:
    """Checks whether Ollama is running locally."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _ollama_installed() -> bool:
    """Checks whether Ollama is installed."""
    return shutil.which("ollama") is not None


def start_ollama_background() -> bool:
    """
    Starts Ollama in the background.
    """
    if not _ollama_installed():
        return False

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        return True

    except (FileNotFoundError, OSError):
        return False


def ensure_ollama_running(max_wait_seconds: int = 15) -> str:
    """
    Makes sure Ollama is running.
    Returns:
        - running
        - started
        - not_installed
        - timeout
    """

    if is_ollama_running():
        return "running"

    if not _ollama_installed():
        return "not_installed"

    launched = start_ollama_background()

    if not launched:
        return "not_installed"

    waited = 0
    interval = 0.5

    while waited < max_wait_seconds:

        if is_ollama_running():
            return "started"

        time.sleep(interval)
        waited += interval

    return "timeout"


def list_local_models() -> list:
    """
    Returns all locally available Ollama models.
    """

    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()

        data = r.json()

        return [m["name"] for m in data.get("models", [])]

    except requests.exceptions.RequestException:
        return []


def ask_ollama(
    prompt: str,
    model: str = "phi3",
    system: str = None,
    temperature: float = 0.1,
    timeout: int = 120,
) -> str:
    """
    Sends a prompt to Ollama and returns the response.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        },
    }

    if system:
        payload["system"] = system

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return None

    except requests.exceptions.Timeout:
        return None

    except requests.exceptions.RequestException:
        return None


def chat_ollama(
    messages: list,
    model: str = "phi3",
    temperature: float = 0.1,
    timeout: int = 120,
) -> str:
    """
    Chat API for Ollama.
    """

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

        data = response.json()

        return (
            data.get("message", {})
            .get("content", "")
            .strip()
        )

    except requests.exceptions.ConnectionError:
        return None

    except requests.exceptions.Timeout:
        return None

    except requests.exceptions.RequestException:
        return None
