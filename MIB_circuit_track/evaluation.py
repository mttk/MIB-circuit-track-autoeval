import math 
import os

from typing import Literal, Optional

import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from MIB_circuit_track.circuit_loading import load_graph_from_json, load_graph_from_pt
from eap.graph import Graph 
from eap.evaluate import evaluate_baseline, evaluate_graph

EDGE_COUNTS = {"gpt2": 32491, "qwen2.5": 179749, "gemma2": 74218, "llama3": 1592881, "interpbench": 1108}

def evaluate_area_under_curve(model: HookedTransformer, graph: Graph, dataloader, metrics, quiet:bool=False, 
                              level:Literal['edge', 'node','neuron']='edge', log_scale:bool=False, absolute:bool=True, 
                              intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching', 
                              intervention_dataloader:DataLoader=None, optimal_ablation_path:Optional[str]=None, 
                              no_normalize:Optional[bool]=False, apply_greedy:bool=False):
    baseline_score = evaluate_baseline(model, dataloader, metrics).mean().item()
    graph.apply_topn(0, True)
    corrupted_score = evaluate_graph(model, graph, dataloader, metrics, quiet=quiet, intervention=intervention, 
                                     intervention_dataloader=intervention_dataloader).mean().item() # ,  optimal_ablation_path=optimal_ablation_path
    
    if level == 'neuron':
        assert graph.neurons_scores is not None, "Neuron scores must be present for neuron-level evaluation"
        n_scored_items = (~torch.isnan(graph.neurons_scores)).sum().item()
    elif level == 'node':
        assert graph.nodes_scores is not None, "Node scores must be present for node-level evaluation"
        n_scored_items = (~torch.isnan(graph.nodes_scores)).sum().item()
    else:
        n_scored_items = len(graph.edges)
    
    percentages = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)

    faithfulnesses = []
    weighted_edge_counts = []
    for pct in percentages:
        this_graph = graph
        curr_num_items = int(pct * n_scored_items)
        print(f"Computing results for {pct*100}% of {level}s (N={curr_num_items})")
        if apply_greedy:
            assert level == 'edge', "Greedy application only supported for edge-level evaluation"
            this_graph.apply_greedy(curr_num_items, absolute=absolute, prune=True)
        else:
            this_graph.apply_topn(curr_num_items, absolute, level=level, prune=True)
        
        weighted_edge_count = this_graph.weighted_edge_count()
        weighted_edge_counts.append(weighted_edge_count)

        ablated_score = evaluate_graph(model, this_graph, dataloader, metrics,
                                       quiet=quiet, intervention=intervention,
                                       intervention_dataloader=intervention_dataloader,
                                       # optimal_ablation_path=optimal_ablation_path
                                       ).mean().item()
        if no_normalize:
            faithfulness = ablated_score
        else:
            faithfulness = (ablated_score - corrupted_score) / (baseline_score - corrupted_score)
        faithfulnesses.append(faithfulness)
    
    area_under = 0.
    area_from_1 = 0.
    for i in range(len(faithfulnesses) - 1):
        i_1, i_2 = i, i+1
        x_1 = percentages[i_1]
        x_2 = percentages[i_2]
        # area from point to 100
        if log_scale:
            x_1 = math.log(x_1)
            x_2 = math.log(x_2)
        trapezoidal = (x_2 - x_1) * \
                        (((abs(1. - faithfulnesses[i_1])) + (abs(1. - faithfulnesses[i_2]))) / 2)
        area_from_1 += trapezoidal 
        
        trapezoidal = (x_2 - x_1) * ((faithfulnesses[i_1] + faithfulnesses[i_2]) / 2)
        area_under += trapezoidal
    average = sum(faithfulnesses) / len(faithfulnesses)
    return weighted_edge_counts, area_under, area_from_1, average, faithfulnesses

def evaluate_area_under_curve_multifile(circuit_path: str, model: HookedTransformer, model_name: str, dataloader, metrics, quiet:bool=False, 
                                        level:Literal['edge', 'node','neuron']='edge', log_scale:bool=False, absolute:bool=True, 
                                        intervention: Literal['patching', 'zero', 'mean','mean-positional', 'optimal']='patching', 
                                        intervention_dataloader:DataLoader=None, optimal_ablation_path:Optional[str]=None, 
                                        no_normalize:Optional[bool]=False, apply_greedy:bool=False):
    baseline_score = evaluate_baseline(model, dataloader, metrics).mean().item()
    new_graph = Graph.from_model(model)
    new_graph.apply_topn(0, True)
    corrupted_score = evaluate_graph(model, new_graph, dataloader, metrics, quiet=quiet, intervention=intervention, 
                                     intervention_dataloader=intervention_dataloader,
                                     # optimal_ablation_path=optimal_ablation_path
                                     ).mean().item()
    
    weighted_edge_counts, faithfulnesses = [], []
    num_valid_circuits = 0
    for filename in os.listdir(circuit_path):
        if num_valid_circuits >= 9:
            break
        if filename.endswith(".pt"):
            graph = load_graph_from_pt(os.path.join(circuit_path, filename))
            num_valid_circuits += 1
        elif filename.endswith(".json"):
            graph = load_graph_from_json(os.path.join(circuit_path, filename))
            num_valid_circuits += 1
        else:
            print(f"Not a valid circuit: {filename}. Continuing.")
            continue
        weighted_edge_count = graph.weighted_edge_count()
        weighted_edge_counts.append(weighted_edge_count)
        ablated_score = evaluate_graph(model, graph, dataloader, metrics,
                                       quiet=quiet, intervention=intervention,
                                       intervention_dataloader=intervention_dataloader,
                                       # optimal_ablation_path=optimal_ablation_path
                                       ).mean().item()
        if no_normalize:
            faithfulness = ablated_score
        else:
            faithfulness = (ablated_score - corrupted_score) / (baseline_score - corrupted_score)
        faithfulnesses.append(faithfulness)
    
    # sort faithfulnesses and weighted_edge_counts jointly
    weighted_edge_counts, faithfulnesses = zip(*sorted(zip(weighted_edge_counts, faithfulnesses)))
    weighted_edge_counts = list(weighted_edge_counts)
    faithfulnesses = list(faithfulnesses)
    weighted_edge_counts.append(EDGE_COUNTS[model_name])
    faithfulnesses.append(1.0)

    percentages = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)
    area_under = 0.
    area_from_1 = 0.
    for i in range(len(faithfulnesses) - 1):
        i_1, i_2 = i, i+1
        x_1 = percentages[i_1]
        x_2 = percentages[i_2]
        # area from point to 100
        if log_scale:
            x_1 = math.log(x_1)
            x_2 = math.log(x_2)
        trapezoidal = (x_2 - x_1) * \
                        (((abs(1. - faithfulnesses[i_1])) + (abs(1. - faithfulnesses[i_2]))) / 2)
        area_from_1 += trapezoidal 
        
        trapezoidal = (x_2 - x_1) * ((faithfulnesses[i_1] + faithfulnesses[i_2]) / 2)
        area_under += trapezoidal
    average = sum(faithfulnesses) / len(faithfulnesses)
    return weighted_edge_counts, area_under, area_from_1, average, faithfulnesses

def compare_graphs(reference: Graph, hypothesis: Graph, by_node: bool = False):
    # Track {true, false} {positives, negatives}
    TP, FP, TN, FN = 0, 0, 0, 0
    total = 0

    if by_node:
        ref_objs = reference.nodes
        hyp_objs = hypothesis.nodes
    else:
        ref_objs = reference.edges
        hyp_objs = hypothesis.edges

    for obj in ref_objs.values():
        total += 1
        if obj.name not in hyp_objs:
            if obj.in_graph:
                TP += 1
            else:
                FP += 1
            continue
            
        if obj.in_graph and hyp_objs[obj.name].in_graph:
            TP += 1
        elif obj.in_graph and not hyp_objs[obj.name].in_graph:
            FN += 1
        elif not obj.in_graph and hyp_objs[obj.name].in_graph:
            FP += 1
        elif not obj.in_graph and not hyp_objs[obj.name].in_graph:
            TN += 1
    
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    TP_rate = recall
    FP_rate = FP / (FP + TN)

    return {"precision": precision,
            "recall": recall,
            "TP_rate": TP_rate,
            "FP_rate": FP_rate}

def evaluate_area_under_roc(reference: Graph, hypothesis: Graph, by_node: bool = False):
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []

    if by_node:
        ref_objs = reference.nodes
        hyp_objs = hypothesis.nodes
    else:
        ref_objs = reference.edges
        hyp_objs = hypothesis.edges
    
    num_objs = len(ref_objs.values())
    for pct in (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1):
        this_num_objs = pct * num_objs
        if by_node:
            raise NotImplementedError("")
        else:
            hypothesis.apply_greedy(this_num_objs)
        scores = compare_graphs(reference, hypothesis)
        tpr_list.append(scores["TP_rate"])
        fpr_list.append(scores["FP_rate"])
        precision_list.append(scores["precision"])
        recall_list.append(scores["recall"])
    
    return {"TPR": tpr_list, "FPR": fpr_list,
            "precision": precision_list, "recall": recall_list}