import json
import socket
import numpy as np
import pandas as pd
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt

# TODO: if storing in dataframe, use timestamp as index because it's an abstract integer
# TODO: check size of individual data (currently set to 1024 bytes), otherwise dict entries could have duplicates
# TODO: Create the separate file that actually executes and returns active message in real time when detected

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind(('localhost', 7890))
# s.listen(1)
# conn, addr = s.accept()
# while 1:
#     data = conn.recv(1024)
#     if not data:
#         break
#     conn.sendall(data)
# conn.close()

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect(('localhost', 7890))
# # s.sendall(b'Hello, world')
# data = s.recv(1024)
# s.close()
# print(repr(data))


def test_fxn_1():
    d = {}
    d_2 = {}
    i = 0

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 7890))
    while True:
        if i == 100:
            break
        data = s.recv(2048)
        d[i] = data
        x = json.loads(data)
        d_2[i] = x
        # pprint(data)
        i += 1
        if not data:
            break
    s.close()
    pprint(d)
    print("\n")
    pprint(d_2)
    print(type(d_2[0]))


# TODO: check if index is before a certain value, you'll hit a json decode error because two messages can fit in one
# ^ Could also just try handle at main and if hit that json error run again
# TODO: decide about global verbosity parameter/flag
# TODO: handle 'ConnectionRefusedError: [Errno 111] Connection refused' if jar file isn't running
def test_fxn():
    error_canary = 0
    i = 0
    d = {}
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 7890))

    while True:
        if i == 200:
            break

        data = s.recv(1024)
        if not data:
            break

        # TODO: buffer json objects instead of handling byte size
        try:
            formatted_data = json.loads(data)
        except json.decoder.JSONDecodeError as e:
            error_canary = 1
            print("\n\nOFFENDING DATA")
            print(data)
            print(type(data))
            print("\n\n")
            print(e)
            raise json.decoder.JSONDecodeError
            # formatted_data = {'data': 0, 'label': 'NULL', 'timestamp': 0}

        d[i] = formatted_data
        i += 1

    s.close()
    df = format_df(raw_dict=d)
    df = heuristic(df=df)
    final_df, acc = measure_performance(df=df)
    cols = ["timestamp", "data", "diff", "active_lag", "zero_threshold", "predicted", "label", "correct"]
    final_df = final_df[cols]
    trunc_df = final_df.drop(['timestamp'], axis=1)
    full_print_df(df=trunc_df)
    # graph_data(df=final_df)
    # graph_test_2(df=final_df)
    graph_missed_predictions(df=final_df)

    if error_canary == 1:
        full_print_df(df=final_df)


def full_print_df(df):
    with pd.option_context('display.max_rows', 400, 'display.max_columns', None):
        # pd.options.display.width = 0
        print(df)


# TODO: decide if dropping timestamp, or re-indexing with timestamp, they can be treated the same, ascending integers
def format_df(raw_dict):
    df = pd.DataFrame.from_dict(raw_dict, orient='index')
    df = df[df['timestamp'] != 0]
    df['predicted'] = "REST"
    df['correct'] = 0
    return df


# TODO: add other performance measures (accuracy, precision, cross-validation
# TODO: check AiFML for additional checks, not necessary but very robust
def measure_performance(df):
    df['correct'] = df.apply(check_label, axis=1)
    total_correct = df['correct'].sum()
    row_count = len(df.index)
    accuracy = total_correct / row_count
    print("Accuracy of Model: " + str(round(accuracy, 2)) + "  ( " + str(total_correct) + " / " + str(row_count) + " )")

    return df, accuracy


def check_label(row):
    if row['label'] == row['predicted']:
        return 1
    else:
        return 0


# brainstorm - absolute difference, moving average, adjustable step, lagged and weighted difference, normalize preproc
# ACCURACY - 68%
def test_heuristic_1(df):
    df['diff'] = 0
    for i in range(1, len(df)):
        df.loc[i, 'diff'] = df.loc[i, 'data'] - df.loc[i - 1, 'data']

    df['predicted'] = df.apply(apply_heuristic_1, axis=1)

    return df


# if previous predicted == activated and diff > -5000, then mark as activated
# elif diff > 5000 and data > 0 then mark as activated
# else pass
def apply_heuristic_1(row):
    if row['diff'] > 5000 and row['data'] > 0:
        return "ACTIVATION"
    else:
        return "REST"


# address lagged labels that the above did not
# because of dataframe relative row operation, this requires two passes; will not require this on the fly
# ACCURACY - 76%
def test_heuristic_2(df):
    df['diff'], df['active_lag'] = 0, 0
    # df['diff'] = 0
    for i in range(1, len(df)):
        df.loc[i, 'diff'] = df.loc[i, 'data'] - df.loc[i - 1, 'data']
    # Removing the below line yields 3% higher accuracy
    df['predicted'] = df.apply(apply_heuristic_1, axis=1)

    # for i in range(1, len(df)):
    #     df.loc[i, 'active_lag'] = lambda x: \
    #         x * 0 if df.loc[i - 1, 'predicted'] == "REST" \
    #         else (x + 2 if (df.loc[i - 1, 'predicted'] == "ACTIVATION" and df.loc[i - 2, 'predicted'] == "ACTIVATION")
    #               else x + 1)

    for i in range(1, len(df)):
        if df.loc[i - 1, 'predicted'] == "REST":
            df.loc[i, 'active_lag'] = 0
        elif df.loc[i - 1, 'predicted'] == "ACTIVATION" and df.loc[i - 2, 'predicted'] == "ACTIVATION":
            df.loc[i, 'active_lag'] = 2
        else:
            df.loc[i, 'active_lag'] = 1

    df['predicted'] = df.apply(apply_heuristic_2, axis=1)

    return df


def apply_heuristic_2(row):
    # TODO: look through data for false positives
    if row['diff'] > 10000 and row['data'] > 0:
        return "ACTIVATION"
    elif row['diff'] > -5000 and row['active_lag'] > 0:
        return "ACTIVATION"
    elif row['data'] > 10000 or row['data'] < -10000:
        return "ACTIVATION"
    else:
        return "REST"


def apply_active_lag(df):
    # Compute the lagged activation status, and fill the values
    for i in range(1, len(df)):
        if i == 0 or i == 1:
            df.loc[i, 'active_lag'] = 0
        elif df.loc[i - 1, 'predicted'] == "REST":
            df.loc[i, 'active_lag'] = 0
        elif df.loc[i - 1, 'predicted'] == "ACTIVATION" and df.loc[i - 2, 'predicted'] == "ACTIVATION":
            df.loc[i, 'active_lag'] = 2
        else:
            df.loc[i, 'active_lag'] = 1
    return df


def apply_diff_calc(df):
    # Compute the difference for column 'diff', and fill the values
    for i in range(1, len(df)):
        df.loc[i, 'diff'] = df.loc[i, 'data'] - df.loc[i - 1, 'data']
    return df


def apply_zero_threshold(df):
    # Compute if the current data value crossed 0 from the last data value, and fill the values
    for i in range(1, len(df)):
        if df.loc[i, 'data'] < 0 < df.loc[i - 1, 'data']:
            df.loc[i, 'zero_threshold'] = 1
        elif df.loc[i, 'data'] > 0 > df.loc[i - 1, 'data']:
            df.loc[i, 'zero_threshold'] = 1
        else:
            df.loc[i, 'zero_threshold'] = 0
    return df


def apply_heuristic(row):
    if row['data'] >= 80000 and row['active_lag'] == 0:
        return "REST"
    elif row['diff'] == 0:
        return "REST"
    elif row['diff'] > 10000 and row['data'] > 0:
        return "ACTIVATION"
    elif row['diff'] > -5000 and row['active_lag'] > 0:
        return "ACTIVATION"
    # TODO: too many false positives between 10K and 50K
    elif row['data'] > 15000 or row['data'] < -10000:
        return "ACTIVATION"
    else:
        return "REST"


# ACCURACY - 79%
def heuristic(df):
    # Initializes new columns:
    # 1. The numerical difference data(n) - data(n-1)
    # 2. A lagged categorical variable for activation (values 0-2), to retain some memory of earlier predicted state
    # 3. A boolean variable for whether or not a data point crossed the x-axis from the n-1th data point
    df['diff'], df['active_lag'], df['zero_threshold'] = 0, 0, 0

    df = apply_diff_calc(df=df)
    df = apply_active_lag(df=df)
    df = apply_zero_threshold(df=df)

    # Predict activation status based on the columns data, diff, and active_lag
    df['predicted'] = df.apply(apply_heuristic, axis=1)
    df = apply_active_lag(df=df)
    return df


def graph_data(df):
    sns.set(style="whitegrid")

    # Draw a scatter plot while assigning point colors and sizes to different
    # variables in the dataset
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(f, left=True, bottom=True)
    label_ranking = ["REST", "ACTIVATION"]
    sns.scatterplot(x="timestamp", y="data",
                    hue="label", size="active_lag",
                    palette=sns.palplot(sns.color_palette("RdBu_r", 7)),
                    hue_order=label_ranking,
                    sizes=(10, 30), linewidth=0,
                    data=df, ax=ax)

    plt.fill_between(df['data'], df['data'])
    plt.show()


def graph_test_2(df):
    sns.set(style="darkgrid")

    # Plot the responses for different events and regions
    # ax = sns.lineplot(x="timestamp", y="data", hue="label", markers=True, dashes=False,
    #                   ci="sd", err_style="band", data=df)

    # ax = sns.lineplot(x="timestamp", y="data", data=df)

    label_ranking = ["REST", "ACTIVATION"]
    l = sns.relplot(x="timestamp", y="data", kind="line", marker="o", hue="label", data=df, hue_order=label_ranking)
    g = sns.relplot(data=df, x="timestamp", y="data", marker="o", hue="label", hue_order=label_ranking)
    p = sns.relplot(data=df, x="timestamp", y="data", marker="o", hue="predicted", hue_order=label_ranking)

    plt.show()


def graph_missed_predictions(df):
    sns.set(style="darkgrid")
    df['marked'] = ""
    df['marked'] = df.apply(mark_incorrect, axis=1)
    validity_ranking = ["CORRECT", "INCORRECT"]
    label_ranking = ["REST", "ACTIVATION"]
    validity_graph = sns.relplot(data=df, x="timestamp", y="data", marker="o", hue="marked", hue_order=validity_ranking)
    ts_graph = sns.relplot(data=df, x="timestamp", y="data", kind="line", marker="o", hue="label",
                           hue_order=label_ranking)
    plt.show()


def mark_incorrect(row):
    if row['label'] == row['predicted']:
        return "CORRECT"
    else:
        return "INCORRECT"


def test_fxn_iterator():
    for i in range(100):
        print("Iteration " + str(i) + " - ")
        test_fxn()


def main():
    test_fxn()


if __name__ == '__main__':
    main()
