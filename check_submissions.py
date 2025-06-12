import os, sys, json, tempfile
import subprocess
import urllib.request as req
import pickle
import logging
from datetime import datetime

from pprint import pprint

from urllib.parse import urlparse
from huggingface_hub import snapshot_download, login, HfApi

from cloud_utils import trigger_google_cloud_job, select_endpoint, get_access_token, wait_for_user_slot, ENDPOINTS, SERVICE_ACCOUNT_FILE
from util import update_request, fetch_submissions, load_submissions, filter_status, data_to_saveable_name
from const import *

def trigger_local_eval():
  try:
    result = subprocess.run(["sbatch", LOCAL_EVAL_SCRIPT_PATH], check=True, capture_output=True, text=True)
    logging.info("Job submitted successfully!")
    logging.info("SLURM response:", result.stdout)
  except subprocess.CalledProcessError as e:
      logging.error("Error submitting job:", e.stderr)

def main():
  os.makedirs("logs/", exist_ok=True)
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

  logging.basicConfig(
          filename=f"logs/submission_check_{timestamp}.log",
          # filemode='a',
          level=logging.INFO,
          format='%(asctime)s - %(levelname)s - %(message)s'
        )


  API = HfApi()
  login(os.environ['HF_API_KEY'])

  fetch_submissions()
  submissions = load_submissions()
  logging.info(f"Submissions pre-filtering: {len(submissions)}")

  # This is just for debugging purposes; only one test submission is up currently
  # for sub in submissions:
  #     filepath, data = sub
  #     if sub[1]['status'].strip() == STATUS_QUEUED:
  #       update_request(API, data, STATUS_PENDING, filepath)


  submissions = filter_status(submissions, status=STATUS_PENDING)
  if len(submissions) == 0:
    logging.info("No submissions to evaluate. Returning.")
    return

  # Update statuses
  for sub in submissions:
      filepath, data = sub

      method_name_saveable = data_to_saveable_name(data)
      
      logging.info(f"Updating {method_name_saveable} request status to {STATUS_QUEUED}; uploading update to requests repo.")
      try:
        pass
        # update_request(API, data, STATUS_QUEUED, filepath)
      except Exception as e:
        logging.error(f"Error updating status of request: {e}. Should retry.")

  # Hardcode to run locally now
  endpoint = 'local' if True else 'gcloud'
  logging.info(f"Running {len(submissions)} pending jobs on {endpoint}.")

  if endpoint == 'local':
    # Execute local job
    logging.info("Triggering local eval.")
    trigger_local_eval()

  elif endpoint == 'gcloud':
    logging.info("Triggering google cloud eval.")
    # Not setup yet, skeleton code
    sys.exit(-1)

    access_token = get_access_token(SERVICE_ACCOUNT_FILE)

    gc_endpoint, reason = select_endpoint(sub, ENDPOINTS, access_token)

    if reason == "sticky":
        print(f"{sub['id']} routing to sticky server for user {sub['user_id']}")
    else:
        print(f"{sub['id']} routing to lowest-load server: {gc_endpoint}")

    if not wait_for_user_slot(sub["user_id"], gc_endpoint, access_token):
        print(f"Timeout waiting for user {sub['user_id']} slot to free. Skipping submission.")

    trigger_google_cloud_job(sub, gc_endpoint, access_token)

if __name__ == '__main__':
  main()