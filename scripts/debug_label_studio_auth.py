"""
debug_label_studio_auth.py — diagnose Label Studio authentication

Usage:
    python scripts/debug_label_studio_auth.py --token <your_token>
    python scripts/debug_label_studio_auth.py  # reads LS_TOKEN env var

The script tries every known auth strategy and reports what works.
"""
import argparse
import base64
import json
import os
import sys

import requests


def decode_jwt_payload(token: str) -> dict:
    try:
        payload_b64 = token.split(".")[1]
        # JWT uses URL-safe base64 without padding — add it back
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception as e:
        return {"error": str(e)}


def check(label: str, session: requests.Session, url: str) -> bool:
    try:
        r = session.get(f"{url}/api/projects")
        if r.ok:
            n = len(r.json().get("results", []))
            print(f"  [OK]  {label}  →  HTTP {r.status_code}, {n} project(s)")
            return True
        else:
            print(f"  [--]  {label}  →  HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"  [ERR] {label}  →  {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Diagnose Label Studio auth")
    parser.add_argument("--token", default=os.environ.get("LS_TOKEN", ""), help="API token")
    parser.add_argument("--url", default="http://localhost:8080", help="Label Studio URL")
    args = parser.parse_args()

    token = args.token.strip()
    url   = args.url.rstrip("/")

    if not token:
        sys.exit("No token — pass --token or set LS_TOKEN env var.")

    print(f"\nLabel Studio URL : {url}")
    print(f"Token length     : {len(token)}")
    print(f"Token preview    : {token[:16]}...{token[-8:]}")
    print(f"Starts with eyJ  : {token.startswith('eyJ')}")

    # ── Decode JWT payload if applicable ──────────────────────────────────────
    if token.startswith("eyJ"):
        payload = decode_jwt_payload(token)
        print(f"\nJWT payload:")
        for k, v in payload.items():
            if k == "exp":
                import datetime
                dt = datetime.datetime.fromtimestamp(v)
                print(f"  {k:12s}: {v}  ({dt})")
            else:
                print(f"  {k:12s}: {v}")

    # ── Connectivity check ────────────────────────────────────────────────────
    print(f"\nChecking connectivity (no auth)...")
    try:
        r = requests.get(f"{url}/health", timeout=5)
        print(f"  /health → HTTP {r.status_code}")
    except requests.ConnectionError:
        sys.exit(f"Cannot reach {url} — is Label Studio running?")

    # ── Auth strategies ───────────────────────────────────────────────────────
    print(f"\nTrying auth strategies against GET /api/projects:")

    def session_with(header_value):
        s = requests.Session()
        s.headers["Authorization"] = header_value
        return s

    strategies = [
        ("Token <token>",  session_with(f"Token {token}")),
        ("Bearer <token>", session_with(f"Bearer {token}")),
    ]

    # If the token looks like a JWT refresh token, try exchanging it
    if token.startswith("eyJ"):
        payload = decode_jwt_payload(token)
        if payload.get("token_type") == "refresh":
            print(f"\n  Token is a JWT refresh token — attempting exchange at /api/token/refresh ...")
            try:
                r = requests.post(f"{url}/api/token/refresh", json={"refresh": token})
                if r.ok:
                    access = r.json().get("access")
                    if access:
                        print(f"  Exchange OK — access token: {access[:16]}...{access[-8:]}")
                        strategies.append(
                            ("Bearer <access> (exchanged)", session_with(f"Bearer {access}"))
                        )
                    else:
                        print(f"  Exchange returned no 'access' field: {r.text[:100]}")
                else:
                    print(f"  Exchange failed: HTTP {r.status_code}  {r.text[:100]}")
            except Exception as e:
                print(f"  Exchange error: {e}")

    print()
    working = None
    for label, s in strategies:
        if check(label, s, url):
            working = (label, s)
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    if working:
        print(f"Working strategy: {working[0]}")
        print(f"\nAdd this to make_session() in label_studio.py to fix the notebook.")
    else:
        print("No strategy worked.")
        print("\nThings to check:")
        print("  - Is Label Studio running?  →  label-studio start")
        print("  - Is the token expired?     →  check 'exp' in JWT payload above")
        print("  - Is the token complete?    →  copy the full token, including both dots")


if __name__ == "__main__":
    main()
