
import random
import time

import requests
import os
import requests

API_URL = "http://127.0.0.1:8000/events/analyze"

FEATURES = [
    "L1_norm", "L2_norm", "L3_norm", "L4_norm",
    "L5_norm", "L6_norm", "L7_norm", "L8_norm",
    "L9_norm", "E1_norm", "E2_norm", "E3_norm",
    "E4_norm", "E5_norm", "F1_norm", "H1_norm",
    "H2_norm", "H3_norm", "D1_norm", "D2_norm",
]

LOGIN_URL = "http://127.0.0.1:8000/auth/login"

def get_access_token():
    response = requests.post(
        LOGIN_URL,
        json={
            "username": os.getenv("ANALYST_USERNAME", "admin"),
            "password": os.getenv("ANALYST_PASSWORD", ""),
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["access_token"]

def generate_activity():
    scenario = random.choices(
        ["routine", "unusual", "mixed"],
        weights=[60, 25, 15],
        k=1,
    )[0]

    if scenario == "routine":
        values = [
            random.uniform(0.0, 0.35)
            for _ in FEATURES
        ]
        event_type = "simulated_normal_activity"
        description = "Synthetic routine employee activity"
        risk_level = "low"

    elif scenario == "unusual":
        values = [
            random.uniform(0.75, 1.0)
            for _ in FEATURES
        ]
        event_type = "simulated_unusual_activity"
        description = "Synthetic unusual employee activity"
        risk_level = "high"

    else:
        values = [
            random.uniform(0.0, 1.0)
            for _ in FEATURES
        ]
        event_type = "simulated_mixed_activity"
        description = "Synthetic mixed employee activity"
        risk_level = "medium"

    return {
        "user_id": f"EMP{random.randint(1, 10):03d}",
        "event_type": event_type,
        "description": description,
        "risk_level": risk_level,
        "features": dict(zip(FEATURES, values)),
    }

def main():
    print("Synthetic activity simulator started.")
    print("Press Ctrl+C to stop.")

    # Log in once and obtain the JWT token
    try:
        login_response = requests.post(
            "http://127.0.0.1:8000/auth/login",
            json={
                "username": os.getenv("ANALYST_USERNAME", "prem"),
                "password": os.getenv("ANALYST_PASSWORD", "Prem@7337"),
            },
            timeout=15,
        )
        login_response.raise_for_status()
        token = login_response.json()["access_token"]

        print("JWT authentication successful.")

    except (requests.RequestException, KeyError, ValueError) as exc:
        print(f"Authentication failed: {exc}")
        return

    headers = {"Authorization": f"Bearer {token}"}

    try:
        while True:
            event = generate_activity()

            try:
                response = requests.post(
                    API_URL,
                    json=event,
                    headers=headers,
                    timeout=15,
                )

                if response.status_code == 201:
                    result = response.json()
                    print(
                        f"User={result['user_id']} | "
                        f"Prediction={result['prediction']} | "
                        f"Score={result['risk_score']} | "
                        f"Event={result['event_id']}"
                    )

                elif response.status_code == 401:
                    print("JWT expired or invalid. Restart the simulator to log in again.")

                else:
                    print(
                        f"API error {response.status_code}: "
                        f"{response.text}"
                    )

            except requests.RequestException as exc:
                print(f"Connection error: {exc}")

            time.sleep(5)

    except KeyboardInterrupt:
        print("\nSimulator stopped.")

if __name__ == "__main__":
    main()
