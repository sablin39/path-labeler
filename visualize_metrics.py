import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def process_results(results):
    # Create a list to store the flattened data
    data = []
    
    for key, value in results.items():
        ground_truths = value['ground_truth'] if isinstance(value['ground_truth'], list) else [value['ground_truth']]
        
        # Add an entry for each ground truth label
        for gt in ground_truths:
            data.append({
                'ground_truth': gt,
                'exact_match': value['exact_match'],
                'google_bleu': value['google_bleu'],
                'meteor': value['meteor'],
                "rouge-1-r": value['rouge_score'][0]['rouge-1']['r']
            })
    
    return pd.DataFrame(data)

def plot_metrics(df, model_id):
    # Calculate mean metrics for each ground truth label
    metrics_by_label = df.groupby('ground_truth').agg({
        'exact_match': 'mean',
        'google_bleu': 'mean',
        'meteor': 'mean',
        "rouge-1-r": 'mean'
    }).reset_index()
    
    # Melt the dataframe for seaborn plotting
    melted_df = pd.melt(metrics_by_label, 
                        id_vars=['ground_truth'],
                        value_vars=['exact_match', 'google_bleu', 'meteor', "rouge-1-r"],
                        var_name='Metric',
                        value_name='Score')
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(data=melted_df, x='ground_truth', y='Score', hue='Metric')
    
    # Add value labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Average Metrics of results {model_id}')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'visualization/{model_id}_caption_eval.png')
    plt.close()

def main(args):
    # Load results
    # results_path='Qwen2.5-VL-3B-Instruct_caption_results.json'
    result_path=args.result_path
    results = load_results(result_path)
    
    # Process results into a dataframe
    df = process_results(results)
    
    # Create visualization

    plot_metrics(df, result_path.replace("_caption_results.json",""))

if __name__ == "__main__":
    parser=ArgumentParser()
    parser.add_argument("--result_path", type=str, default='Qwen2.5-VL-3B-Instruct_caption_results.json')
    args=parser.parse_args()
    main(args)
