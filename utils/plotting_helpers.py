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

def optimized_bar_plots(metrics, score_data, x_min=0, x_max=3, legendLoc = False):

    model_order = ['gpt-4-turbo', 'gpt-3.5-turbo', 'Orca-2-7b', 'mpt-7b-chat', 'vicuna-7b-v1.5', 
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

        if not legendLoc:
            if i < n_rows - 1:
                axes[i].legend([],[], frameon=False)
        else:
            if i != legendLoc:
                axes[i].legend([],[], frameon=False)

        axes[i].grid(which='major', linestyle='--', linewidth='0.5', color='grey')
    

    
    plt.tight_layout()
    plt.show()


def plot_heatmaps_comb(df, threats, concepts):
    models = ['gpt-4-turbo', 'gpt-3.5-turbo', 'Orca-2-7b', 'mpt-7b-chat', 'vicuna-7b-v1.5', 
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


def plot_heatmaps_side_by_side(df, concept1, concept2, threat_order):
    models = ['gpt-4-turbo', 'gpt-3.5-turbo', 'Orca-2-7b', 'mpt-7b-chat', 'vicuna-7b-v1.5', 
              'Llama-2-7b-chat-hf', 'vicuna-13b-v1.5', 'Llama-2-13b-chat-hf']

    sns.set(font_scale=1.2)
    
    # Filter the DataFrame for the two given concepts
    df_concept1 = df[df['concept'] == concept1]
    df_concept2 = df[df['concept'] == concept2]

    # Find global min and max for color scale normalization
    global_min = min(df_concept1[threat_order].min().min(), df_concept2[threat_order].min().min())
    global_max = max(df_concept1[threat_order].max().max(), df_concept2[threat_order].max().max())

    # Group and pivot data for both concepts
    def prepare_data(df_concept):
        avg_threat_metrics_concept = df_concept.groupby(['model', 'job'])[threat_order].mean().reset_index()
        heatmap_data_concept = avg_threat_metrics_concept.pivot(index='job', columns='model', values=threat_order)
        heatmap_data_concept.columns = heatmap_data_concept.columns.reorder_levels([1,0])
        heatmap_data_concept.sort_index(axis=1, level=0, inplace=True)
        return heatmap_data_concept

    heatmap_data_concept1 = prepare_data(df_concept1)
    heatmap_data_concept2 = prepare_data(df_concept2)

    num_models = len(models)

    # Generate side-by-side heatmaps with shared y-axis and single colorbar
    # fig, axes = plt.subplots(num_models, 2, figsize=(20, num_models * 4), sharey=True)
    fig, axes = plt.subplots(num_models, 2, figsize=(10, num_models * 2.5), sharey=True, sharex=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])  # Position for the colorbar

    for i, model in enumerate(models):
        sns.heatmap(heatmap_data_concept1[model].T, ax=axes[i, 0], cmap="coolwarm", vmin=global_min, vmax=global_max, 
                    cbar=False, annot=True)
        sns.heatmap(heatmap_data_concept2[model].T, ax=axes[i, 1], cmap="coolwarm", vmin=global_min, vmax=global_max, 
                    cbar=i == 0, cbar_ax=None if i else cbar_ax, annot=True)

        axes[i, 0].set_ylabel(model)
        axes[i, 0].set_xlabel('')
        axes[i, 1].set_xlabel('')

    # Set the xlabel on the last subplot for threat metrics
    axes[-1, 0].set_xlabel(concept1)
    axes[-1, 1].set_xlabel(concept2)
    
    # Adjust layout for better fit
    plt.tight_layout(rect=[0, 0, .9, 1])

    # Show the figure
    plt.show()