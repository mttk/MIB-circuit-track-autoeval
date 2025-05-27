import os
import pickle
from functools import partial
import argparse
import torch

from transformer_lens import HookedTransformer

from eap.graph import Graph
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.utils import TASKS_TO_HF_NAMES, MODEL_NAME_TO_FULLNAME, COL_MAPPING
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.evaluation import evaluate_area_under_curve, evaluate_area_under_curve_multifile
from MIB_circuit_track.circuit_loading import load_graph_from_json, load_graph_from_pt


def run_evaluation(circuit_path, model_name, task, split, method_name, level, batch_size=20, head=None,
                   absolute=False, debug=False):
    if f"{task.replace('_', '-')}_{model_name}" not in COL_MAPPING:
        print(f"Non-valid task/model combo: {task}_{model_name}")
        return None
    
    if model_name in ("qwen2.5", "gemma2", "llama3"):
        model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name],
                                                  attn_implementation="eager", torch_dtype=torch.bfloat16)
    else:
        model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name])
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    hf_task_name = f'mib-bench/{TASKS_TO_HF_NAMES[task]}'
    num_examples = 1 if debug else None
    dataset = HFEAPDataset(hf_task_name, model.tokenizer, split=split, task=task, model_name=model_name,
                           num_examples=num_examples)
    if head is not None:
        if len(dataset) < head:
            print(f"Warning: dataset has only {len(dataset)} examples, but head is set to {head}; using all examples.")
            head = len(dataset)
        dataset.head(head)
    dataloader = dataset.to_dataloader(batch_size=batch_size)
    metric = get_metric('logit_diff', task, model.tokenizer, model)
    attribution_metric = partial(metric, mean=False, loss=False)
    
    # method_name_saveable = f"{method_name}_{level}"
    if len(os.listdir(circuit_path)) == 1:
        for filename in os.listdir(circuit_path):
            if filename.endswith(".pt"):
                graph = Graph.from_pt(os.path.join(circuit_path, filename))
            elif filename.endswith(".json"):
                graph = Graph.from_json(os.path.join(circuit_path, filename))
            else:
                raise ValueError(f"Found {filename}, but is not .pt or .json")
            eval_auc_outputs = evaluate_area_under_curve(model, graph, dataloader,
                                            attribution_metric, level=level, absolute=absolute)
    elif len(os.listdir(circuit_path)) >= 2:
        eval_auc_outputs = evaluate_area_under_curve_multifile(circuit_path, model, model_name, dataloader,
                                                               attribution_metric, level=level,
                                                               absolute=absolute)

    weighted_edge_counts, area_under, area_from_1, average, faithfulnesses = eval_auc_outputs
    d = {
        "weighted_edge_counts": weighted_edge_counts,
        "area_under": area_under,
        "area_from_1": area_from_1,
        "average": average,
        "faithfulnesses": faithfulnesses
    }
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--tasks", type=str, nargs='+', required=True)
    parser.add_argument("--ablation", type=str, choices=['patching', 'zero', 'mean', 'mean-positional', 'optimal'], default='patching')
    parser.add_argument("--optimal_ablation_path", type=str, default=None)
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument("--method", type=str, default=None, help="Method used to generate the circuit (only needed to infer circuit file name)")
    parser.add_argument("--level", type=str, choices=['edge', 'node', 'neuron'], default='edge')
    parser.add_argument("--absolute", type=bool, default=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--circuit-dir", type=str, default='circuits')
    parser.add_argument("--circuit-files", type=str, nargs='+', default=None)
    parser.add_argument("--output-dir", type=str, default='results')
    args = parser.parse_args()

    i = 0
    for model_name in args.models:
        model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name])
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.ungroup_grouped_query_attention = True
        for task in args.tasks:
            if f"{task.replace('_', '-')}_{model_name}" not in COL_MAPPING:
                continue
            method_name_saveable = f"{args.method}_{args.ablation}_{args.level}"
            p = f"{args.circuit_dir}/{method_name_saveable}/{task.replace('_', '-')}_{model_name}/importances.pt"

            if args.circuit_files is not None:
                p = args.circuit_files[i]
                i += 1

            print(f"Loading circuit from {p}")
            if p.endswith('.json'):
                graph = Graph.from_json(p)
            elif p.endswith('.pt'):
                graph = Graph.from_pt(p)
            else:
                raise ValueError(f"Invalid file extension: {p.suffix}")
            
            hf_task_name = f'mib-bench/{TASKS_TO_HF_NAMES[task]}'
            dataset = HFEAPDataset(hf_task_name, model.tokenizer, split=args.split, task=task, model_name=model_name)
            if args.head is not None:
                head = args.head
                if len(dataset) < head:
                    print(f"Warning: dataset has only {len(dataset)} examples, but head is set to {head}; using all examples.")
                    head = len(dataset)
                dataset.head(head)
            dataloader = dataset.to_dataloader(batch_size=args.batch_size)
            metric = get_metric('logit_diff', task, model.tokenizer, model)
            attribution_metric = partial(metric, mean=False, loss=False)
            
            eval_auc_outputs = evaluate_area_under_curve(model, graph, dataloader, attribution_metric, level=args.level, 
                                                            log_scale=True, absolute=args.absolute, intervention=args.ablation,
                                                            optimal_ablation_path=args.optimal_ablation_path)
            weighted_edge_counts, area_under, area_from_1, average, faithfulnesses = eval_auc_outputs

            d = {
                "weighted_edge_counts": weighted_edge_counts,
                "area_under": area_under,
                "area_from_1": area_from_1,
                "average": average,
                "faithfulnesses": faithfulnesses
            }
            method_name_saveable = f"{args.method}_{args.ablation}_{args.level}"
            output_path = os.path.join(args.output_dir, method_name_saveable)
            os.makedirs(output_path, exist_ok=True)
            with open(f"{output_path}/{task}_{model_name}_{args.split}_abs-{args.absolute}.pkl", 'wb') as f:
                pickle.dump(d, f)
