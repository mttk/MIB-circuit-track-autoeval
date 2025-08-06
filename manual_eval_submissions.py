import os, sys, json, tempfile
import pickle
import logging

from pprint import pprint


from huggingface_hub import snapshot_download, login, HfApi

from run_evaluation import run_evaluation
from MIB_circuit_track.utils import TASKS_TO_HF_NAMES, MODEL_NAME_TO_FULLNAME, COL_MAPPING
from util import update_request, fetch_submissions, load_submissions, filter_status, upload_results, parse_huggingface_url, data_to_saveable_name
from const import *

def main():
  # 1. Fetch submissions
  # 2. Check request status, filter PENDING
  login(os.environ['MIB_HF_KEY'])
  os.makedirs("logs/", exist_ok=True)
  API = HfApi()

  fetch_submissions()
  submissions = load_submissions()
  print(f"Found {len(submissions)} submissions.")
  print(submissions)

  # Run only submissions which are set as QUEUED by the check_submissions script
  submissions = filter_status(submissions, status=STATUS_PENDING)
  if len(submissions) == 0:
    print("No submissions set to be queued. Returning.")
    return

  print(f"Executing eval for {len(submissions)} submissions.")
  output_dir = "eval_results"

  with tempfile.TemporaryDirectory() as temp_dir:
    for sub in submissions:
      filepath, data = sub
      # 3. Download circuit (all submissions are QUEUED)
      circuit_path = data['hf_repo']
      level = data['circuit_level']
      method_name = data['method_name']
      user_name = data['user_name']
      submission_id = data['_id']
      submit_time = data['submit_time']

      method_name_saveable = f"{user_name}_{method_name}_{submission_id}_{submit_time}"
      logging.basicConfig(
        filename=f"logs/{method_name_saveable}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
      )
      logging.info("")
      logging.info("Starting script")
      logging.info(f"Evaluating {method_name} (ID: {submission_id}) from {user_name}")
      logging.info(f"HF repo: {circuit_path}")
      logging.info(f"Circuit defined at {level} level")
      logging.info("")

      submission_repo, submission_prefix, revision = parse_huggingface_url(circuit_path)
      suffixes = ('.pt', '.json')
      revision = data['revision']

      local_temp_dir = os.path.join(temp_dir, CIRCUITS_DIR)
      snapshot_download(repo_id=submission_repo, allow_patterns=[f"{submission_prefix}*{suffix}" for suffix in suffixes],
                        revision=revision, local_dir=local_temp_dir)

      local_circuit_path = os.path.join(local_temp_dir, submission_prefix)

      circuit_dirs, tasks, model_names = [], [], []
      # Generate info required for circuit eval
      for dir in os.listdir(local_circuit_path):
        curr_task, curr_model = None, None
        # Look for task names in filename
        for task in list(TASKS_TO_HF_NAMES.keys()):
            if dir.startswith(task) or f"_{task}" in dir:
                curr_task = task
        # Look for model names in filename
        for model in list(MODEL_NAME_TO_FULLNAME.keys()):
            if dir.startswith(model) or f"_{model}" in dir:
                curr_model = model
        if curr_task is None or curr_model is None:
          logging.warning(f"Skipping {dir}: could not find valid model or task name")
          continue
        circuit_dirs.append(os.path.join(local_circuit_path, dir))
        tasks.append(curr_task)
        model_names.append(curr_model)
        # Note model files as well? Just the root circuit dir seems necessary

      logging.info(f"Found tasks: {tasks}")
      logging.info(f"Found models: {model_names}")

      # 4. Evaluate circuit using https://github.com/hannamw/MIB-circuit-track/blob/main/run_evaluation.py
      # requires: 
      # parser.add_argument("--models", type=str, nargs='+', required=True)
      # parser.add_argument("--tasks", type=str, nargs='+', required=True)

      # TODO: does this need to go into a separate shell script? Not yet sure how a cronjob will hook into this part
      # TODO: if task.startswith('arc') and model_name.startswith('llama'): use 80G GPU
      #       else: use 40G GPU
      # TODO (Aaron): make compatible with multiple-circuit-file submissions, instead of just importances
      output_path = os.path.join(output_dir, method_name_saveable)

      # 4.1
      # Label the submission as running
      logging.info("Updating request status; uploading update to requests repo.")
      try:
        update_request(API, data, STATUS_RUNNING, filepath)
      except Exception as e:
        logging.error(f"Error updating status of request: {e}. Should retry.")

      for idx in range(len(circuit_dirs)):
        circuit_dir = circuit_dirs[idx]
        logging.info(f"Evaluating {circuit_dir}")
        task = tasks[idx]
        model_name = model_names[idx]
        split = "test"
        # We need to run evals with and without absolute to match our paper's eval setting
        for absolute in (True, False):
          results = None # In case run_eval breaks
          try:
            results = run_evaluation(circuit_dir, model_name, task, split, method_name, level, batch_size=20, head=None,
                        absolute=absolute)
          except Exception as e:
            logging.error(f"Error evaluating {model_name} on {task}: {e}")
          if results is None:
            logging.warning(f"Evaluation did not return results for {circuit_dir}")
            continue
          logging.info(f"Finished evaluation for {circuit_dir}")
          
          os.makedirs(output_path, exist_ok=True)
          with open(f"{output_path}/{task}_{model_name}_{split}_abs-{absolute}.pkl", 'wb') as f:
            pickle.dump(results, f)
          logging.info("Results saved.")
          logging.info("")

      # 5. validate results and update status
      tasks, models = set(), set()

      new_status = ''
      if not os.listdir(output_path): # If results directory is empty
        logging.error("No valid results.")
        new_status = STATUS_FAILED
      for filename in os.listdir(output_path):
        task, model, _, _ = filename.split("_")
        tasks.add(task)
        models.add(model)
      if len(tasks) < 2:
        logging.error(f"Not enough tasks found: need at least two, but only had {tasks}")
        new_status = STATUS_FAILED
      if len(models) < 2:
        logging.error(f"Not enough models found: need at least two, but only had {models}")
        new_status = STATUS_FAILED
      
      logging.info("Evaluation finished with no breaking errors.")
      new_status = STATUS_FINISHED

      # 5. Upload json to results repo
      # mib-bench/subgraph-results
      logging.info("Collecting results.")
      results_aggregated = {"method_name": method_name,
                            "results": []}
      results_by_model = {}
      for results_file in os.listdir(output_path):
        if not results_file.endswith(".pkl"):
          continue
        fp = os.path.join(output_path, results_file)
        print(results_file)
        task_name, model_name, _, is_absolute = results_file.split("_")
        task_name = task_name.replace("-", "_")
        if model_name not in results_by_model:
          results_by_model[model_name] = {}
        if task_name not in results_by_model[model_name]:
          results_by_model[model_name][task_name] = {}
        with open(fp, 'rb') as handle:
          results_data = pickle.load(handle)
          if "True" in is_absolute:
            results_by_model[model_name][task_name]["CMD"] = {
                           "edge_counts": results_data["weighted_edge_counts"],
                           "faithfulness": results_data["faithfulnesses"]}
          else:
            results_by_model[model_name][task_name]["CPR"] = {
                           "edge_counts": results_data["weighted_edge_counts"],
                           "faithfulness": results_data["faithfulnesses"]}
      for model_name in results_by_model:
        results_aggregated["results"].append({"model_id": model_name,
                                              "scores": results_by_model[model_name]})
      logging.info("Succesfully collected results.")

      results_aggregated_filename = method_name_saveable + ".json"
      results_aggregated_path = os.path.join(output_path, results_aggregated_filename)
      logging.info(f"Saving results in expected .json format to {results_aggregated_path}")
      with open(results_aggregated_path, 'w') as results_output_file:
        results_output_file.write(json.dumps(results_aggregated, indent=4))
          
      logging.info("Uploading results.")
      try:
        upload_results(API, results_aggregated_path)
      except Exception as e:
        logging.error(f"Error uploading results: {e}. Should retry.")
    
      # 6. Update status on requests repo
      # mib-bench/requests-subgraph
      logging.info("Updating request status; uploading update to requests repo.")
      try:
        update_request(API, data, new_status, filepath)
      except Exception as e:
        logging.error(f"Error updating status of request: {e}. Should retry.")

if __name__ == '__main__':
  main()