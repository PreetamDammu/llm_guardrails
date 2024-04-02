import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import matplotlib


def run_significance(df, metrics):
    models = list(set(df['model'].to_list()))
    p_values_summary = []

    for model in models:
        model_metric_data = df[(df['model'] == model)]
        for metric in metrics:
            
            caste_scores = model_metric_data[model_metric_data['concept'] == 'caste'][metric]
            race_scores = model_metric_data[model_metric_data['concept'] == 'race'][metric]
            
            # Conduct a Mann-Whitney U test (non-parametric) since we cannot assume normal distribution of scores
            stat, p_value = stats.mannwhitneyu(caste_scores, race_scores, alternative='two-sided')
            
            p_values_summary.append({'Model': model, 'Metric': metric, 'P-Value': p_value})

    p_values_df = pd.DataFrame(p_values_summary)
    p_values_df['Significance'] = ['Significant' if p < 0.05 else 'Not Significant' for p in p_values_df['P-Value']]

    # wide_format_df = p_values_df.pivot('Model', 'Metric', 'P-Value')
    # formatted_df = wide_format_df.applymap(lambda x: f"**{x:.2e}**" if x < 0.05 else f"{x:.3f}")


    return p_values_df



def bar_plots(metrics, score_data, x_min=0, x_max=3):

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 20))

    for i, metric in enumerate(metrics):
        sns.barplot(x=metric, y='model', hue='concept', data=score_data, ax=axes[i], palette="muted")
        axes[i].set_title(metric)
        axes[i].set_xlabel('Score')
        axes[i].set_xlim(x_min, x_max)  
        axes[i].set_ylabel('')

    plt.tight_layout()
    plt.show()

def optimized_bar_plots(metrics, score_data, x_min=0, x_max=3):

    model_order = ['gpt4-preview', 'gpt-3.5-turbo', 'Orca-2-7b', 'mpt-7b-chat', 'vicuna-7b-v1.5', 
                'Llama-2-7b-chat-hf', 'vicuna-13b-v1.5', 'Llama-2-13b-chat-hf']
    score_data = score_data.set_index('model').loc[model_order].reset_index()

    sns.set(font_scale=1.35)
    matplotlib.rc('xtick', labelsize='small')
    matplotlib.rc('ytick', labelsize='small')


    n_rows = len(metrics)
    

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows), sharex=True)
    

    for i, metric in enumerate(metrics):

        sns.barplot(x=metric, y='model', hue='concept', data=score_data, ax=axes[i], palette="muted", order=model_order)
        axes[i].set_title(metric)
        axes[i].set_xlim(x_min, x_max)
        
        # Remove the axis labels
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

        if i < n_rows - 1:
            axes[i].legend([],[], frameon=False)

        axes[i].grid(which='major', linestyle='--', linewidth='0.5', color='grey')
    

    
    plt.tight_layout()
    plt.show()


def plot_heatmaps_comb(df, threats, concepts):
    models = ['gpt4-preview', 'gpt-3.5-turbo', 'Orca-2-7b', 'mpt-7b-chat', 'vicuna-7b-v1.5', 
                'Llama-2-7b-chat-hf', 'vicuna-13b-v1.5', 'Llama-2-13b-chat-hf']

    global_min = float('inf')
    global_max = float('-inf')
    

    for concept in concepts:
        df_concept = df[df['concept'] == concept]
        for model in models:
            model_scores = df_concept[df_concept['model'] == model][threats].mean()
            global_min = min(global_min, model_scores.min())
            global_max = max(global_max, model_scores.max())
    

    sns.set(font_scale=1.2)


    fig, axes = plt.subplots(1, len(concepts), figsize=(7, 3), sharey=True, gridspec_kw={'wspace':0.02, 'hspace':0}) #len(models)*0.25 + 1
    
    
    for i, concept in enumerate(concepts):
        df_concept = df[df['concept'] == concept]
        threat_scores_concept = pd.DataFrame(columns=threats)
        
        for model in models:
            model_scores = df_concept[df_concept['model'] == model][threats].mean()
            new_row = pd.DataFrame(model_scores.values.reshape(1, -1), columns=threats, index=[model])
            threat_scores_concept = pd.concat([threat_scores_concept, new_row])
        
        sns.heatmap(threat_scores_concept, annot=True, fmt=".2f", cmap='coolwarm', vmin=global_min, vmax=global_max, linewidths=.5, ax=axes[i], annot_kws={"size": 10}, cbar=False)
        axes[i].set_title(f'{concept.capitalize()}')
        axes[i].tick_params(axis='x')#, rotation=45)
        # axes[i].set_xlabel('Threat Types')
    
    # axes[0].set_ylabel('Models')
    

    cbar_ax = fig.add_axes([0.02, -0.45, 0.02, 0.6])  # x, y, width, height
    fig.colorbar(axes[0].collections[0], cax=cbar_ax, orientation="vertical")

    plt.subplots_adjust(bottom=0.2)

    plt.show()

    return

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


def plot_heatmaps(df, models, threats, concepts):

    global_min = float('inf')
    global_max = float('-inf')
    

    for concept in concepts:
        df_concept = df[df['concept'] == concept]
        for model in models:
            model_scores = df_concept[df_concept['model'] == model][threats].mean()
            global_min = min(global_min, model_scores.min())
            global_max = max(global_max, model_scores.max())
    
    for concept in concepts:
        df_concept = df[df['concept'] == concept]
        threat_scores_concept = pd.DataFrame(columns=threats)
        
        for model in models:
            model_scores = df_concept[df_concept['model'] == model][threats].mean()
            new_row = pd.DataFrame(model_scores.values.reshape(1, -1), columns=threats, index=[model])
            threat_scores_concept = pd.concat([threat_scores_concept, new_row])
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(threat_scores_concept, annot=True, fmt=".2f", cmap='coolwarm', vmin=global_min, vmax=global_max, linewidths=.5)
        plt.title(f'Mean Threat Scores by Model for Concept: {concept}')
        plt.ylabel('Models')
        plt.xlabel('Threat Types')
        plt.show()

    return


def plot_heatmaps(df, models, threats, concepts):
    # Remove the y-axis label and place the color bar beneath the heatmaps

    # Create a new figure with adjusted size for better fit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True, gridspec_kw={'width_ratios': [len(race_agg.columns), len(caste_agg.columns)]})

    # Create the race heatmap
    sns.heatmap(race_agg, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, norm=norm, ax=ax1, annot_kws={"size": 14})
    ax1.set_title('Race')
    ax1.set_xticklabels(threat_types, rotation=45, ha='right')

    # Create the caste heatmap
    sns.heatmap(caste_agg, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, norm=norm, ax=ax2, annot_kws={"size": 14})
    ax2.set_title('Caste')
    ax2.set_xticklabels(threat_types, rotation=45, ha='right')

    # Add a color bar at the bottom of the heatmaps
    cbar_ax = fig.add_axes([0.15, -0.15, 0.7, 0.02])  # x, y, width, height
    fig.colorbar(ax1.collections[0], cax=cbar_ax, orientation="horizontal")

    # Adjust layout to accommodate the new color bar position
    plt.subplots_adjust(bottom=0.2)

    # Show the plot
    plt.show()

    return