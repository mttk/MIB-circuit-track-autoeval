import os, json
from urllib.parse import urlparse
import urllib.request as req

from huggingface_hub import snapshot_download

from const import *

def update_request(API, data: dict, new_status: str, local_path: str):
    """Updates a given eval request with its new status on the hub (running, completed, failed,)"""
    with open(local_path) as fp:
        data = json.load(fp)

    data["status"] = new_status

    with open(local_path, "w") as f:
        f.write(json.dumps(data))

    API.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=os.path.relpath(local_path, start=SUBMISSION_DIR),
        repo_id=SUBMISSIONS_REPO,
        repo_type='dataset',
    )

def data_to_saveable_name(data):
    method_name = data['method_name']
    user_name = data['user_name']
    submission_id = data['_id']
    submit_time = data['submit_time']

    method_name_saveable = f"{user_name}_{method_name}_{submission_id}_{submit_time}"
    return method_name_saveable


def fetch_submissions():
  # TODO: Error handling
  snapshot_download(repo_id=SUBMISSIONS_REPO, repo_type='dataset', allow_patterns="*.json", local_dir=SUBMISSION_DIR)

def fetch_submission(hf_repo, circuit_level="edge"):
  # TODO: Error handling
  obj = {"hf_repo": hf_repo, "method_name": "test", "circuit_level": circuit_level,
         "description": None, "status": "PENDING"}
  return obj

def load_submissions(root_path=SUBMISSION_DIR):
  submissions = []
  for file in os.listdir(root_path):
    if file.endswith('.json'):
      fp = root_path + file
      with open(fp, 'r') as infile:
        submissions.append((fp, json.load(infile)))
  return submissions

def filter_status(submissions, status=STATUS_PENDING):
  return [s for s in submissions if s[1]['status'].strip() == status]

def upload_results(API, results_path):
    basename = os.path.basename(results_path)
    API.upload_file(
       path_or_fileobj=results_path,
       path_in_repo=os.path.join("submissions", basename),
       repo_id=RESULTS_REPO,
       repo_type='dataset'
    )


def download_circuit(url, rootdir=''):
  fname = os.path.basename(urlparse(url).path)

  # TODO: Error handling
  if rootdir:
    filepath = os.path.join(rootdir, fname)
  else:
    filepath = fname

  req.urlretrieve(url, filepath)
  return filepath


def parse_huggingface_url(url: str):
    """
    Extracts repo_id and subfolder path from a Hugging Face URL.
    Returns (repo_id, folder_path).
    """
    # Handle cases where the input is already a repo_id (no URL)
    if not url.startswith(("http://", "https://")):
        return url, None
    
    parsed = urlparse(url)
    path_parts = parsed.path.strip("/").split("/")
    revision = "main"
    
    # Extract repo_id (username/repo_name)
    if len(path_parts) < 2:
        return None, None, None     # Can't extract repo_id
    else:
        repo_id = f"{path_parts[0]}/{path_parts[1]}"
    
    # Extract folder path (if in /tree/ or /blob/)
    if "tree" in path_parts or "blob" in path_parts:
        try:
            branch_idx = path_parts.index("tree") if "tree" in path_parts else path_parts.index("blob")
            folder_path = "/".join(path_parts[branch_idx + 2:])  # Skip "tree/main" or "blob/main"
            revision = path_parts[branch_idx + 1]
        except (ValueError, IndexError):
            folder_path = None
    else:
        folder_path = None
    
    return repo_id, folder_path, revision
