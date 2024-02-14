import matplotlib.pyplot as plt
import seaborn as sns


def plot_average_scores_by_concept(data, concepts=None):

    if concepts is None:
        concepts = data['concept'].unique()
    
    filtered_data = data[data['concept'].isin(concepts)]

    numeric_cols = ['toxicity','severe_toxicity','obscene','threat','insult','identity_attack', 'politeness_score']
    print(numeric_cols)
    avg_scores_by_concept = filtered_data.groupby('concept')[numeric_cols].mean()

    plt.figure(figsize=(14, 8))
    for column in avg_scores_by_concept.columns:
        sns.lineplot(x=avg_scores_by_concept.index, y=avg_scores_by_concept[column], label=column)

    plt.title('Average Metric Scores by Concept')
    plt.ylabel('Average Score')
    plt.xlabel('Concept')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()




def plot_boxplots_grid(data, metrics=None, concepts=None, cols=3):

    if metrics is None:
        metrics = [col for col in data.columns if col not in ['Unnamed: 0', 'model', 'concept', 'job', 'obscene']]
    if concepts is None:
        concepts = data['concept'].unique()


    filtered_data = data[data['concept'].isin(concepts)]

    num_metrics = len(metrics)
    rows = (num_metrics + cols - 1) // cols 
    
    plt.figure(figsize=(18, 6 * rows))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(x='concept', y=metric, data=filtered_data)
        plt.title(f'{metric.capitalize()} by Concept')
        plt.ylim(0, 1)
        plt.tight_layout(pad=3.0)  

    plt.show()
