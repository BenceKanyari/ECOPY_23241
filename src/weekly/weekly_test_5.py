import pandas as pd

def change_price_to_float(input_df):
    new_df = input_df
    new_df['item_price'] = [float(x[1:]) for x in new_df.item_price]
    return new_df


def number_of_observations(input_df):
    new_df = input_df.copy()
    return new_df.shape[0]


#%%
def items_and_prices(input_df):
    new_df = input_df.copy()
    return new_df[['item_name','item_price']]


def sorted_by_price(input_df):
    new_df = input_df.copy()
    return new_df.sort_values("item_price", ascending=False)



def avg_price(input_df):
    new_df = input_df.copy()
    return new_df['item_price'].mean()



def unique_items_over_ten_dollars(input_df):
    new_df = input_df.copy()
    new_df = new_df[new_df['item_price']>10]
    new_df = new_df.drop_duplicates(subset=['item_name','item_price', 'choice_description'])
    return new_df[['item_name', 'choice_description', 'item_price']]


def items_starting_with_s(input_df):
    new_df = input_df.copy()
    new_df = new_df[[x[0]=='S' for x in new_df['item_name']]]
    return new_df['item_name'].drop_duplicates()


def first_three_columns(input_df):
    new_df = input_df.copy()
    return new_df.iloc[:,0:3]


def every_column_except_last_two(input_df):
    new_df = input_df.copy()
    return new_df.iloc[:,:-2]


def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    new_df = input_df.copy()
    return new_df.loc[[x in rows_to_keep for x in new_df[column_to_filter]], columns_to_keep]


def generate_quartile(input_df):
    new_df = input_df.copy()
    new_df['Quartile'] = pd.cut(new_df['item_price'], bins=[0, 10, 20, 30,float('inf')], labels=['low-cost', 'medium-cost', 'high-cost', 'premium']).astype(str)
    return new_df


def average_price_in_quartiles(input_df):
    new_df = input_df.copy()
    return new_df.groupby('Quartile')['item_price'].mean()



def minmaxmean_price_in_quartile(input_df):
    new_df = input_df.copy()
    def mean(x):
        return sum(x)/len(x)
    return new_df.groupby('Quartile')['item_price'].agg([min, max, mean])


def gen_uniform_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    distribution.rand.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):

        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(distribution.gen_rand())
            mean_trajectory.append(sum(trajectory) / (i + 1))

        trajectories.append(mean_trajectory)

    return trajectories


def gen_logistic_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    distribution.rand.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):

        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(distribution.gen_rand())
            mean_trajectory.append(sum(trajectory) / (i + 1))

        trajectories.append(mean_trajectory)

    return trajectories


def gen_laplace_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    distribution.rand.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):

        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(distribution.gen_rand())
            mean_trajectory.append(sum(trajectory) / (i + 1))

        trajectories.append(mean_trajectory)

    return trajectories


def gen_cauchy_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    distribution.rand.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):

        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(distribution.gen_rand())
            mean_trajectory.append(sum(trajectory) / (i + 1))

        trajectories.append(mean_trajectory)

    return trajectories


def gen_chi2_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    distribution.rand.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):

        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(distribution.gen_rand())
            mean_trajectory.append(sum(trajectory) / (i + 1))

        trajectories.append(mean_trajectory)

    return trajectories