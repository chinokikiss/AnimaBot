import json
import time
import base64
import urllib.request
from utils import log

COMFY_HOST = "http://127.0.0.1:8188"


def load_workflow(path, overrides: dict | None = None):
    with open(path, encoding="utf-8") as f:
        workflow = json.load(f)
    if overrides:
        for node_id, inputs in overrides.items():
            if node_id in workflow:
                workflow[node_id]["inputs"].update(inputs)
    return workflow


def post_prompt(workflow, client_id=None):
    body = {"prompt": workflow}
    if client_id:
        body["client_id"] = client_id
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFY_HOST}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_history(prompt_id):
    with urllib.request.urlopen(f"{COMFY_HOST}/history/{prompt_id}") as resp:
        return json.loads(resp.read())


def run_workflow(workflow, wait=True, poll_interval=1.0, timeout=300):
    result = post_prompt(workflow)
    prompt_id = result["prompt_id"]
    log(f"Submitted prompt: {prompt_id}")

    if not wait:
        return prompt_id

    start = time.time()
    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"Prompt {prompt_id} timed out")
        history_data = get_history(prompt_id)
        if history_data and prompt_id in history_data:
            history = history_data[prompt_id]
            status = history.get("status", {})
            if status.get("completed"):
                results = []
                for node_id, node_output in history.get("outputs", {}).items():
                    if "images_data" in node_output:
                        for img in node_output["images_data"]:
                            results.append(base64.b64decode(img["data"]))
                return results
            elif status.get("status_str") == "error":
                raise RuntimeError(f"Execution error: {history}")
        time.sleep(poll_interval)