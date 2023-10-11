import pandas as pd
import matplotlib.pyplot as plt

def number_of_participants(input_df):
    new_df = input_df.copy()
    return new_df['Team'].unique().shape[0]

def goals(input_df):
    new_df = input_df.copy()
    return new_df[['Team','Goals']]

def sorted_by_goal(input_df):
    new_df = input_df.copy()
    return new_df.sort_values(by='Goals', ascending=False)

def avg_goal(input_df):
    new_df = input_df.copy()
    return new_df['Goals'].mean()

def countries_over_five(input_df):
    new_df = input_df.copy()
    return new_df.loc[new_df['Goals']>5, 'Team'].to_frame()

def countries_starting_with_g(input_df):
    new_df = input_df.copy()
    return new_df.loc[[x[0]=='G' for x in new_df['Team']], 'Team'].to_frame()

def first_seven_columns(input_df):
    new_df = input_df.copy()
    return new_df.head(7)

def every_column_except_last_three(input_df):
    new_df = input_df.copy()
    return new_df.iloc[:,0:-3]

def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    new_df = input_df.copy()
    return new_df.loc[[x in rows_to_keep for x in new_df[column_to_filter]], columns_to_keep]

def generate_quartile(input_df):
    new_df = input_df.copy()
    new_df['Quartile'] = pd.cut(new_df['Goals'], bins=[0, 2, 4, 5, 12], labels=[4, 3, 2, 1])
    return new_df

def average_yellow_in_quartiles(input_df):
    new_df = input_df.copy()
    return new_df.groupby('Quartile')['Passes'].mean().to_frame()


def minmax_block_in_quartile(input_df):
    new_df = input_df.copy()
    return new_df.groupby('Quartile')['Passes'].agg([min, max])


def scatter_goals_shots(input_df):
    new_df = input_df.copy()
    fig, ax = plt.subplots()
    ax.scatter(new_df['Goals'], new_df['Shots on target'])
    ax.set_title('Goals and Shot on target')
    ax.set_xlabel('Goals')
    ax.set_ylabel('Goals and Shot on target')
    return fig


def scatter_goals_shots_by_quartile(input_df):
    new_df = input_df.copy()
    fig, ax = plt.subplots()
    colors = {1: 'red', 2: 'green', 3: 'blue', 4: 'purple'}
    for quartile, color in colors.items():
        quartile_data = new_df[new_df['Quartile'] == quartile]
        ax.scatter(quartile_data['Goals'], quartile_data['Shots on target'], label=quartile, color=color)
    ax.set_title('Goals and Shot on target')
    ax.set_xlabel('Goals')
    ax.set_ylabel('Goals and Shot on target')
    ax.legend(title='Quartile')
    return fig


def generate_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory):
    pareto_distribution.rand.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):

        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(pareto_distribution.gen_rand())
            mean_trajectory.append(sum(trajectory) / (i + 1))

        trajectories.append(mean_trajectory)

    return trajectories
