import os, sys, json, tempfile
import urllib.request as req

from pprint import pprint

from urllib.parse import urlparse
from huggingface_hub import snapshot_download, login
from messaging import send_email

STATUS_PENDING = 'PENDING'
STATUS_FINISHED = 'FINISHED'

SUBMISSION_DIR = 'submissions/'
CIRCUITS_DIR = 'eval_circuits/'
SUBMISSIONS_REPO = 'mib-bench/requests-subgraph'

SHARED_TASK_EMAIL = '' # TODO: update

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

  with tempfile.TemporaryDirectory() as temp_dir:
    for sub in submissions:
      # 3. Download circuit (all submissions are PENDING)
      circuit_path = sub['hf_repo']
      print(circuit_path)

      # TODO: Parse these parts from path
      result_repo = 'mib-bench/mib-circuits-example'
      result_prefix = 'importances/pt'
      result_suffix = '.pt'
      revision = sub['revision']
      snapshot_download(repo_id=result_repo, allow_patterns=f"{result_prefix}*{result_suffix}", revision=revision, local_dir=CIRCUITS_DIR)


      local_circuit_path = os.path.join(CIRCUITS_DIR, result_prefix)

      datasets, model_names = set(), set()
      # Generate info required for circuit eval
      for dir in os.listdir(local_circuit_path):
        dataset, short_model_name = dir.split("_", 1)
        datasets.add(dataset)
        model_names.add(short_model_name)
        # Note model files as well? Just the root circuit dir seems necessary

      print(datasets, model_names)
      method_name = sub['method_name']

      # 4. Evaluate circuit using https://github.com/hannamw/MIB-circuit-track/blob/main/run_evaluation.py
      # requires: 
      # parser.add_argument("--models", type=str, nargs='+', required=True)
      # parser.add_argument("--tasks", type=str, nargs='+', required=True)


      # 5. Upload json to results repo
      # mib-bench/subgraph-results


      # 6. Update status
      new_status = ''

      # 7. Send an email to the submission contact informing of status change
      # TODO: login to gmail with whichever acct will be used
      # email_on_complete(author_email=sub['contact_email'], new_status=new_status)

if __name__ == '__main__':
  main()
