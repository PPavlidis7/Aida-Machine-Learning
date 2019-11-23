import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_names = [
    "PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "PRSA_Data_Changping_20130301-20170228.csv",
    "PRSA_Data_Dingling_20130301-20170228.csv",
    "PRSA_Data_Dongsi_20130301-20170228.csv",
    "PRSA_Data_Guanyuan_20130301-20170228.csv",
    "PRSA_Data_Gucheng_20130301-20170228.csv",
    "PRSA_Data_Huairou_20130301-20170228.csv",
    "PRSA_Data_Nongzhanguan_20130301-20170228.csv",
    "PRSA_Data_Shunyi_20130301-20170228.csv",
    "PRSA_Data_Tiantan_20130301-20170228.csv",
    "PRSA_Data_Wanliu_20130301-20170228.csv",
    "PRSA_Data_Wanshouxigong_20130301-20170228.csv",
]

file_id = [
    "Aoti",
    "Chan",
    "Ding",
    "Dong",
    "Guan",
    "Guch",
    "Huai",
    "Nong",
    "Shun",
    "Tian",
    "Wanl",
    "Wans",
]


def construct_plot(data, title):
    x = np.arange(len(file_id))  # the label locations
    width = 0.05  # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, data['CO'], color='b', width=width, label='CO')
    ax.bar(x + width / 2, data['SO2'], color='r', width=width, label='SO2')
    ax.bar(x + width, data['NO2'], color='g', width=width, label='NO2')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(file_id)
    ax.legend()
    fig.tight_layout()
    plt.show()


def create_plots(statistics_data):
    means = {
        'CO': [],
        'SO2': [],
        'NO2': [],
    }
    median = {
        'CO': [],
        'SO2': [],
        'NO2': [],
    }
    sample_variances = {
        'CO': [],
        'SO2': [],
        'NO2': [],
    }
    plt.rcParams["figure.figsize"] = [10, 7]

    for file_name in file_names:
        for col_name in ['CO', 'SO2', 'NO2']:
            means[col_name].append(statistics_data[file_name][col_name]['mean'])
            median[col_name].append(statistics_data[file_name][col_name]['media'])
            sample_variances[col_name].append(statistics_data[file_name][col_name]['sample_variance'])

    construct_plot(median, 'Median')
    construct_plot(means, 'Mean')
    construct_plot(sample_variances, 'Sample variance')


def main():
    statistics_data = {}
    for file_name in file_names:
        statistics_data[file_name] = {}
        file_data = pd.read_csv(file_name)
        file_data.dropna(subset=['CO', 'SO2', 'NO2'], inplace=True)

        for col_name in ['CO', 'SO2', 'NO2']:
            statistics_data[file_name][col_name] = {}
            statistics_data[file_name][col_name]['mean'] = statistics.mean(file_data[col_name])
            statistics_data[file_name][col_name]['harmonic_mean'] = statistics.harmonic_mean(file_data[col_name])
            statistics_data[file_name][col_name]['media'] = statistics.median(file_data[col_name])
            statistics_data[file_name][col_name]['low_median'] = statistics.median_low(file_data[col_name])
            statistics_data[file_name][col_name]['high_median'] = statistics.median_high(file_data[col_name])
            statistics_data[file_name][col_name]['population_std_dev'] = statistics.pstdev(file_data[col_name])
            statistics_data[file_name][col_name]['population_variance'] = statistics.pvariance(file_data[col_name])
            statistics_data[file_name][col_name]['sample_std_dev'] = statistics.stdev(file_data[col_name])
            statistics_data[file_name][col_name]['sample_variance'] = statistics.variance(file_data[col_name])

    create_plots(statistics_data)


if __name__ == '__main__':
    main()
