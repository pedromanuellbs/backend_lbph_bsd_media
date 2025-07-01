import requests

BASE = "https://backendlbphbsdmedia-production.up.railway.app"

# 1) Register
with open("faces/contoh1.jpg", "rb") as f:
    data  = {"user_id": "user123"}
    files = {"image": f}
    r = requests.post(f"{BASE}/register_face", data=data, files=files)

print("=== REGISTER ===")
print("Status code:", r.status_code)
print("Content-Type:", r.headers.get("Content-Type"))
print("Response body:", repr(r.text))  # cetak mentah, termasuk whitespace
try:
    print("JSON:", r.json())
except Exception as e:
    print("Failed to parse JSON:", e)

# 2) Verify
with open("faces/contoh2_test.jpg", "rb") as f:
    files = {"image": f}
    r = requests.post(f"{BASE}/verify_face", files=files)

print("\n=== VERIFY ===")
print("Status code:", r.status_code)
print("Content-Type:", r.headers.get("Content-Type"))
print("Response body:", repr(r.text))
try:
    print("JSON:", r.json())
except Exception as e:
    print("Failed to parse JSON:", e)
