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
  submissions = filter_status(submissions, status=STATUS_FAILED) # STATUS_PENDING
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
      if "description" in data:
        description = data["description"]
      else:
        description = ""

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


      def process_directory(directory_path, path_prefix=""):
        """Recursively process directories to find circuits, handling abs/True/False subdirectories"""
        circuit_configs = []
        
        for dir_name in os.listdir(directory_path):
          dir_path = os.path.join(directory_path, dir_name)
          dir_lower = dir_name.lower()
          
          if not os.path.isdir(dir_path):
            continue
          
          # Check if this directory contains "abs" and "True"/"False"
          if "abs" in dir_lower and ("true" in dir_lower or "false" in dir_lower):
            # This is an abs/True or abs/False directory, recurse into it
            try:
              sub_configs = process_directory(dir_path, os.path.join(path_prefix, dir_name))
              circuit_configs.extend(sub_configs)
            except Exception as e:
              logging.warning(f"Could not process subdirectory {dir_path}: {e}")
            continue
          
          # Look for task and model names in directory name
          curr_task, curr_model = None, None
          for task in list(TASKS_TO_HF_NAMES.keys()):
            if dir_lower.startswith(task) or f"_{task}" in dir_lower:
              curr_task = task
              break
          
          for model in list(MODEL_NAME_TO_FULLNAME.keys()):
            if dir_lower.startswith(model) or f"_{model}" in dir_lower:
              curr_model = model
              break
          
          if curr_task is None or curr_model is None:
            logging.warning(f"Skipping {dir_name}: could not find valid model or task name")
            continue
          
          # Determine if absolute value is specified in the directory structure
          absolute_specified = None
          parent_path_lower = path_prefix.lower()
          if "abs" in parent_path_lower:
            if "true" in parent_path_lower:
              absolute_specified = True
            elif "false" in parent_path_lower:
              absolute_specified = False
          
          circuit_configs.append({
            'circuit_dir': dir_path,
            'task': curr_task,
            'model': curr_model,
            'absolute_specified': absolute_specified,
            'path_info': os.path.join(path_prefix, dir_name) if path_prefix else dir_name
          })
        
        return circuit_configs

      # Generate info required for circuit eval using recursive processing
      circuit_configs = process_directory(local_circuit_path)
      
      if not circuit_configs:
        logging.warning("No valid circuit configurations found")
        continue
      
      logging.info(f"Found {len(circuit_configs)} circuit configurations:")
      for config in circuit_configs:
        abs_info = f" (absolute={config['absolute_specified']})" if config['absolute_specified'] is not None else ""
        logging.info(f"  {config['task']}_{config['model']}{abs_info} at {config['path_info']}")


      #circuit_dirs, tasks, model_names = [], [], []
      # # Generate info required for circuit eval
      #for dir in os.listdir(local_circuit_path):
      #  curr_task, curr_model = None, None
      #  # Look for task names in filename
      #  for task in list(TASKS_TO_HF_NAMES.keys()):
      #      if dir.startswith(task) or f"_{task}" in dir:
      #          curr_task = task
      #  # Look for model names in filename
      #  for model in list(MODEL_NAME_TO_FULLNAME.keys()):
      #      if dir.startswith(model) or f"_{model}" in dir:
      #          curr_model = model
      #  if curr_task is None or curr_model is None:
      #    logging.warning(f"Skipping {dir}: could not find valid model or task name")
      #    continue
      #  circuit_dirs.append(os.path.join(local_circuit_path, dir))
      #  tasks.append(curr_task)
      #  model_names.append(curr_model)
      #  # Note model files as well? Just the root circuit dir seems necessary

      # logging.info(f"Found tasks: {tasks}")
      # logging.info(f"Found models: {model_names}")

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

      for config in circuit_configs:
        circuit_dir = config['circuit_dir']
        task = config['task']
        model_name = config['model']
        absolute_specified = config['absolute_specified']
        path_info = config['path_info']
        
        logging.info(f"Evaluating {circuit_dir}")
        split = "test"
        
        # Determine which absolute values to evaluate
        if absolute_specified is not None:
          # Absolute value is specified in directory structure, only run that one
          absolute_values = [absolute_specified]
        else:
          # No absolute value specified, run both True and False as before
          absolute_values = [True, False]
        
        for absolute in absolute_values:
          try:
            results = run_evaluation(circuit_dir, model_name, task, split, method_name, level, batch_size=20, head=None,
                        absolute=absolute)
          except Exception as e:
            logging.error(f"Error evaluating {model_name} on {task} with absolute={absolute}: {e}")
            continue
            
          if results is None:
            logging.warning(f"Evaluation did not return results for {circuit_dir} with absolute={absolute}")
            continue
            
          logging.info(f"Finished evaluation for {circuit_dir} with absolute={absolute}")
          
          os.makedirs(output_path, exist_ok=True)
          
          # Include path info in filename if from subdirectory
          path_suffix = f"_{path_info.replace('/', '#')}" if path_info != f"{task}_{model_name}" else ""
          filename = f"{output_path}/{task}_{model_name}{path_suffix}_abs-{absolute}.pkl"
          
          with open(filename, 'wb') as f:
            pickle.dump(results, f)
          logging.info(f"Results saved to {filename}")
          logging.info("")

      # TODO: if any new errors, comment out the above and use previous main() function


      #for idx in range(len(circuit_dirs)):
      #  circuit_dir = circuit_dirs[idx]
      #  logging.info(f"Evaluating {circuit_dir}")
      #  task = tasks[idx]
      # model_name = model_names[idx]
      #  split = "test"
      #  # We need to run evals with and without absolute to match our paper's eval setting
      #  for absolute in (True, False):
      #    results = None # In case run_eval breaks
      #    try:
      #      results = run_evaluation(circuit_dir, model_name, task, split, method_name, level, batch_size=20, head=None,
      #                  absolute=absolute)
      #    except Exception as e:
      #      logging.error(f"Error evaluating {model_name} on {task}: {e}")
      #    if results is None:
      #      logging.warning(f"Evaluation did not return results for {circuit_dir}")
      #      continue
      #    logging.info(f"Finished evaluation for {circuit_dir}")
      #    
      #    os.makedirs(output_path, exist_ok=True)
      #    with open(f"{output_path}/{task}_{model_name}_{split}_abs-{absolute}.pkl", 'wb') as f:
      #      pickle.dump(results, f)
      #    logging.info("Results saved.")
      #    logging.info("")

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
                            "description": description,
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