import argparse
import json

def compute_area(edge_counts, faithfulnesses):
    # Return None if either list is empty
    if not edge_counts or not faithfulnesses:
        return None, None, None

    # percentages = [e / max(edge_counts) for e in edge_counts]
    percentages = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    area_under = 0.
    area_from_100 = 0.
    for i in range(len(faithfulnesses) - 1):
        i_1, i_2 = i, i+1
        x_1 = percentages[i_1]
        x_2 = percentages[i_2]
        # area from point to 100
        # if log_scale:
        #     x_1 = math.log(x_1)
        #     x_2 = math.log(x_2)
        trapezoidal = (percentages[i_2] - percentages[i_1]) * \
                        (((abs(1. - faithfulnesses[i_1])) + (abs(1. - faithfulnesses[i_2]))) / 2)
        area_from_100 += trapezoidal 
        
        trapezoidal = (percentages[i_2] - percentages[i_1]) * ((faithfulnesses[i_1] + faithfulnesses[i_2]) / 2)
        area_under += trapezoidal
    average = sum(faithfulnesses) / len(faithfulnesses)
    return (area_under, area_from_100, average)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str, help=".json results file.")
    args = parser.parse_args()

    with open(args.results_file, 'r') as scores:
        data = json.load(scores)
        print("loaded successfully")
        method_name = data["method_name"]
        for idx in range(len(data["results"])):
            model_id = data["results"][idx]["model_id"]
            scores = data["results"][idx]["scores"]
            for task in data["results"][idx]["scores"]:
                for metric in data["results"][idx]["scores"][task]:
                    edge_counts = data["results"][idx]["scores"][task][metric]["edge_counts"]
                    faithfulness = data["results"][idx]["scores"][task][metric]["faithfulness"]
                    area_under, area_from_100, _ = compute_area(edge_counts, faithfulness)
                    if metric == "CPR":
                        print(f"{model_id} on {task} ({metric}): {area_under}")
                    elif metric == "CMD":
                        print(f"{model_id} on {task} ({metric}): {area_from_100}")
                    else:
                        print("shouldn't be here")

if __name__ == "__main__":
    main()