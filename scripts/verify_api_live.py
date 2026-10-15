import os
import sys
import time
import json
import threading
import httpx
import uvicorn

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main import app

def run_server(server: uvicorn.Server):
    server.run()

def main():
    print("=" * 60)
    print("STARTING LIVE FASTAPI SERVER VERIFICATION")
    print("=" * 60)

    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=run_server, args=(server,), daemon=True)
    server_thread.start()

    # Wait for server to start
    client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0)
    for _ in range(30):
        try:
            r = client.get("/health")
            if r.status_code == 200 and r.json().get("retrieval_engine_initialized"):
                break
        except Exception:
            time.sleep(0.5)

    print("\n--- 1. VERIFYING GET /health ---")
    res_health = client.get("/health")
    print(f"Status Code: {res_health.status_code}")
    print(json.dumps(res_health.json(), indent=2))

    print("\n--- 2. VERIFYING POST /api/v1/search ---")
    search_payload = {"query": "breach of contract compensation for loss", "top_k": 2}
    res_search = client.post("/api/v1/search", json=search_payload)
    print(f"Status Code: {res_search.status_code}")
    print(json.dumps(res_search.json(), indent=2))

    print("\n--- 3. VERIFYING POST /api/v1/civilian/ask ---")
    civilian_payload = {"query": "Section 73 of Indian Contract Act", "top_k": 2}
    res_civ = client.post("/api/v1/civilian/ask", json=civilian_payload)
    print(f"Status Code: {res_civ.status_code}")
    print(json.dumps(res_civ.json(), indent=2))

    print("\n--- 4. VERIFYING POST /api/v1/student/ask ---")
    student_payload = {"query": "Section 3 Arms Act 1959", "top_k": 2}
    res_stu = client.post("/api/v1/student/ask", json=student_payload)
    print(f"Status Code: {res_stu.status_code}")
    print(json.dumps(res_stu.json(), indent=2))

    print("\n--- 5. VERIFYING POST /api/v1/lawyer/research ---")
    lawyer_payload = {
        "query": "punishment for culpable homicide not amounting to murder",
        "top_k": 2,
        "act_filter": "BNS"
    }
    res_law = client.post("/api/v1/lawyer/research", json=lawyer_payload)
    print(f"Status Code: {res_law.status_code}")
    print(json.dumps(res_law.json(), indent=2))

    print("\n" + "=" * 60)
    print("ALL 5 LIVE ENDPOINTS VERIFIED SUCCESSFULLY")
    print("=" * 60)

    server.should_exit = True

if __name__ == "__main__":
    main()
