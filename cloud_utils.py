import os
import json
import requests
import time
import hashlib

from google.oauth2 import service_account
from google.auth.transport.requests import Request

# ==== Config ====
SERVICE_ACCOUNT_FILE = os.getenv("GCP_SERVICE_ACCOUNT_FILE")
ENDPOINTS = os.getenv("GCP_ENDPOINTS", "").split(",")  # Comma-separated base URLs
MAX_WAIT_TIME = 300  # Max seconds to wait for a user's slot to free
WAIT_INTERVAL = 10  # Polling interval
# ================

def get_access_token(service_account_file):
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_request = Request()
    credentials.refresh(auth_request)
    return credentials.token


def get_server_status(endpoint, access_token):
    try:
        response = requests.get(f"{endpoint}/status", headers={
            "Authorization": f"Bearer {access_token}"
        })
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Warning: Could not get status from {endpoint}: {e}")
        return {"load": float("inf"), "active_users": []}


def select_endpoint(submission, endpoints, access_token):
    # Rule 1: Sticky user routing
    for endpoint in endpoints:
        status = get_server_status(endpoint, access_token)
        if submission["user_id"] in status.get("active_users", []):
            return endpoint, "sticky"

    # Rule 2: Choose lowest load
    best_endpoint = None
    lowest_load = float("inf")
    for endpoint in endpoints:
        status = get_server_status(endpoint, access_token)
        if status["load"] < lowest_load:
            lowest_load = status["load"]
            best_endpoint = endpoint

    return best_endpoint, "load"

def wait_for_user_slot(user_id, endpoint, access_token):
    waited = 0
    while waited < MAX_WAIT_TIME:
        status = get_server_status(endpoint, access_token)
        if user_id not in status.get("active_users", []):
            return True
        print(f"User {user_id} is still active on {endpoint}. Waiting...")
        time.sleep(WAIT_INTERVAL)
        waited += WAIT_INTERVAL
    return False

def trigger_google_cloud_job(submission, endpoint, access_token):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "submission_id": submission["id"],
        "user_id": submission["user_id"],
        "submission_data": submission["data"]
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        print(f"[SUCCESS] {submission['id']} sent to {endpoint}")
    except requests.RequestException as e:
        print(f"[ERROR] {submission['id']} failed to send: {e}")


def main():
    access_token = get_access_token(SERVICE_ACCOUNT_FILE)
    print(access_token)

    sys.exit()
    submissions = find_new_submissions()

    if not submissions:
        print("No new submissions found.")
        return

    for submission in submissions:
        endpoint, reason = select_endpoint(submission, ENDPOINTS, access_token)

        if reason == "sticky":
            print(f"{submission['id']} routing to sticky server for user {submission['user_id']}")
        else:
            print(f"{submission['id']} routing to lowest-load server: {endpoint}")

        if not wait_for_user_slot(submission["user_id"], endpoint, access_token):
            print(f"Timeout waiting for user {submission['user_id']} slot to free. Skipping submission.")
            continue

        trigger_google_cloud_job(submission, endpoint, access_token)

if __name__ == "__main__":
    main()
