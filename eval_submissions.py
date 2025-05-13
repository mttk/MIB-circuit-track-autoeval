import os, sys, json, tempfile
import urllib.request as req
import pickle

from pprint import pprint

from urllib.parse import urlparse
from huggingface_hub import snapshot_download, login
from messaging import send_email

from run_evaluation import run_evaluation
from MIB_circuit_track.utils import TASKS_TO_HF_NAMES, MODEL_NAME_TO_FULLNAME, COL_MAPPING

STATUS_PENDING = 'PENDING'
STATUS_FINISHED = 'FINISHED'

SUBMISSION_DIR = 'submissions/'
CIRCUITS_DIR = 'eval_circuits/'
SUBMISSIONS_REPO = 'mib-bench/requests-subgraph'

SHARED_TASK_EMAIL = '' # TODO: update

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


def fetch_submissions():
  # TODO: Error handling
  snapshot_download(repo_id=SUBMISSIONS_REPO, repo_type='dataset', allow_patterns="*.json", local_dir=SUBMISSION_DIR)


def load_submissions(root_path=SUBMISSION_DIR):
  submissions = []
  for file in os.listdir(root_path):
    if file.endswith('.json'):
      fp = root_path + file
      with open(fp, 'r') as infile:
        submissions.append(json.load(infile))
  return submissions


def filter_status(submissions, status=STATUS_PENDING):
  return [s for s in submissions if s['status'].strip() == status]


def download_circuit(url, rootdir=''):
  fname = os.path.basename(urlparse(url).path)

  # TODO: Error handling
  if rootdir:
    filepath = os.path.join(rootdir, fname)
  else:
    filepath = fname

  req.urlretrieve(url, filepath)
  return filepath


def email_on_complete(author_email, new_status):
  send_email(sender=SHARED_TASK_EMAIL, receiver=author_email,
             subject="[MIB] Status change", content="")


def main():
  # 1. Fetch submissions
  # 2. Check request status, filter PENDING
  login(os.environ['HF_API_KEY'])
  fetch_submissions()

  submissions = filter_status(load_submissions())
  pprint(submissions[0])

  output_dir = "eval_results"

  with tempfile.TemporaryDirectory() as temp_dir:
    for sub in submissions:
      # 3. Download circuit (all submissions are PENDING)
      circuit_path = sub['hf_repo']
      level = sub['circuit_level']
      method_name = sub['method_name']
      user_name = sub['user_name']
      submission_id = sub['_id']
      print(circuit_path)

      # TODO: Parse these parts from path
      submission_repo, submission_prefix, revision = parse_huggingface_url(circuit_path)
      suffixes = ('.pt', '.json')
      revision = sub['revision']
      snapshot_download(repo_id=submission_repo, allow_patterns=[f"{submission_prefix}*{suffix}" for suffix in suffixes],
                        revision=revision, local_dir=CIRCUITS_DIR)

      local_circuit_path = os.path.join(CIRCUITS_DIR, submission_prefix)

      circuit_dirs, datasets, model_names = [], [], []
      # Generate info required for circuit eval
      for dir in os.listdir(local_circuit_path):
        task, model = None, None
        # Look for task names in filename
        for task in list(TASKS_TO_HF_NAMES.keys()):
            if dir.startswith(task) or f"_{task}" in dir:
                curr_task = task
        # Look for model names in filename
        for model in list(MODEL_NAME_TO_FULLNAME.keys()):
            if dir.startswith(model) or f"_{model}" in dir:
                curr_model = model
        if task is None or model is None:
          print(f"Skipping {dir}: could not find valid model or task name")
          continue
        circuit_dirs.append(dir)
        datasets.add(task)
        model_names.add(model)
        # Note model files as well? Just the root circuit dir seems necessary

      print(datasets, model_names)

      # 4. Evaluate circuit using https://github.com/hannamw/MIB-circuit-track/blob/main/run_evaluation.py
      # requires: 
      # parser.add_argument("--models", type=str, nargs='+', required=True)
      # parser.add_argument("--tasks", type=str, nargs='+', required=True)

      # TODO: does this need to go into a separate shell script? Not yet sure how a cronjob will hook into this part
      # TODO: if task.startswith('arc') and model_name.startswith('llama'): use 80G GPU
      #       else: use 40G GPU

      for idx in range(circuit_dirs):
        circuit_dir = circuit_dirs[idx]
        dataset = datasets[idx]
        model_name = model_names[idx]
        split = "test"
        # We need to run evals with and without absolute to match our paper's eval setting
        for absolute in (True, False):
          results = run_evaluation(circuit_path, model_name, task, split, method_name, level, batch_size=20, head=None,
                        absolute=absolute)
          method_name_saveable = f"{user_name}_{method_name}_{submission_id}"
          output_path = os.path.join(output_dir, method_name_saveable)
          os.makedirs(output_path, exist_ok=True)
          with open(f"{output_path}/{task}_{model_name}_{split}_abs-{absolute}.pkl", 'wb') as f:
            pickle.dump(results, f)


      # 5. Upload json to results repo
      # mib-bench/subgraph-results
    

      # 6. Update status
      new_status = ''

      # 7. Send an email to the submission contact informing of status change
      # TODO: login to gmail with whichever acct will be used
      # email_on_complete(author_email=sub['contact_email'], new_status=new_status)

if __name__ == '__main__':
  main()
