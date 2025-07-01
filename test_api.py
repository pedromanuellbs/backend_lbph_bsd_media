import requests
import glob
import os

BASE    = "https://backendlbphbsdmedia-production.up.railway.app"
USER_ID = "user123"

# 1) Batch Register
faces_dir = os.path.join("faces", USER_ID)
register_paths = glob.glob(os.path.join(faces_dir, "*.jpg"))

print("=== REGISTER BATCH ===")
for path in register_paths:
    with open(path, "rb") as f:
        # r = requests.post(
        #     f"{BASE}/register_face",
        #     data={"user_id": USER_ID},
        #     files={"image": f}
        # )
        r = requests.post(
    "https://backendlbphbsdmedia-production.up.railway.app/register_face",
    data={"user_id": "user123"},
    files={"image": open("faces/contoh1.jpg", "rb")},
)

    print(f"{os.path.basename(path):20} → {r.status_code}  {r.json()}")

# 2) Batch Verify
# Let’s assume kamu punya folder `faces/user123_test/` dengan 5 foto untuk verifikasi
verify_dir   = os.path.join("faces", f"{USER_ID}_test")
verify_paths = glob.glob(os.path.join(verify_dir, "*.jpg"))

print("\n=== VERIFY BATCH ===")
for path in verify_paths:
    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE}/verify_face",
            files={"image": f}
        )
    print(f"{os.path.basename(path):20} → {r.status_code}  {r.json()}")
