import matplotlib.pyplot as plt
import json
import os

def visualize():
    if not os.path.exists("evaluation/results/eval_results.json"):
        print("No results to visualize")
        return

    with open("evaluation/results/eval_results.json", "r") as f:
        data = json.load(f)
    
    metrics = data["metrics"]
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title("RAG Evaluation Metrics")
    plt.ylim(0, 1.1)
    
    os.makedirs("evaluation/results", exist_ok=True)
    plt.savefig("evaluation/results/viz_eval_results.png")
    print("Visualization saved to evaluation/results/viz_eval_results.png")

if __name__ == "__main__":
    visualize()
