import requests
import json
import time

def run_test():
    session_id = "test_5turn_sequence_session"
    url_chat = "http://127.0.0.1:8000/api/chat"
    url_reset = "http://127.0.0.1:8000/api/reset"

    # Reset session to guarantee fresh baseline
    r_reset = requests.post(url_reset, json={"session_id": session_id})
    print(f"Reset status: {r_reset.status_code}")

    turns = [
        (1, "What is the exact chemical formula for glucose?"),
        (2, "explain glucose in detail"),
        (3, "now for sucrose say"),
        (4, "in short explain"),
        (5, "where sucrose is used")
    ]

    for turn_num, msg in turns:
        print(f"\n==========================================")
        print(f"=== TURN {turn_num}: \"{msg}\"")
        print(f"==========================================")
        resp = requests.post(url_chat, json={"session_id": session_id, "message": msg})
        if resp.status_code != 200:
            print(f"ERROR: HTTP {resp.status_code}: {resp.text}")
            continue
        data = resp.json()
        t = data.get("telemetry", {})
        applied = t.get("applied")
        proba = t.get("probability")
        tier = t.get("tier")
        reason = t.get("reason")
        pref = t.get("preference_description")
        answer = data.get("response", "")
        
        print(f"Status: {resp.status_code}")
        print(f"Applied: {applied} | Probability: {proba}")
        print(f"Tier: {tier}")
        print(f"Reason: {reason}")
        print(f"Active Preference: {pref}")
        preview = answer[:300].replace('\n', ' ')
        print(f"Answer Preview: {preview}...")
        time.sleep(1.0)

if __name__ == "__main__":
    run_test()
