import json
import time
import socket
import pandas as pd
from pprint import pprint


def full_print_df(df):
    with pd.option_context('display.max_rows', 400, 'display.max_columns', None):
        # pd.options.display.width = 0
        print(df)


def test_fxn_iterator():
    for i in range(100):
        print("Iteration " + str(i) + " - ")
        test_fxn()


def format_df(raw_dict):
    df = pd.DataFrame.from_dict(raw_dict, orient='index')
    try:
        df = df[df['timestamp'] != 0]
    except KeyError as e:
        print("WARNING: Timestamp missing from data point")
        print(e)
    df['predicted'] = "REST"
    df['correct'] = 0
    return df


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


def test_fxn(n_epochs, n_iter):
    epochs = 0
    epochal_dict = {}
    acc_list = []

    while True:
        if epochs == n_epochs:
            break

        start_time = time.time()
        # start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        # print("Start Time: " + start_str)

        i = 0
        d = {}
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', 7890))

        while True:
            data = s.recv(1024)
            if i == n_iter:
                break
            if not data:
                break
            try:
                formatted_data = json.loads(data)
            except json.decoder.JSONDecodeError as e:
                break
            d[i] = formatted_data
            i += 1

        s.close()
        df = format_df(raw_dict=d)
        df = heuristic(df=df)
        df, acc = measure_performance(df=df)
        # cols = ["timestamp", "data", "diff", "active_lag", "predicted", "label", "correct"]
        # df = df[cols]

        end_time = time.time()
        # end_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        # print("End Time: " + end_str)
        elapsed_time = end_time - start_time
        print("Elapsed Time: " + str(elapsed_time))
        print("\n")

        epochal_dict[epochs] = acc
        acc_list.append(acc)
        epochs += 1

    # pprint(epochal_dict)
    acc_series = pd.Series(acc_list)
    acc_mean = acc_series.mean()
    acc_median = acc_series.median()
    print("Mean Accuracy of " + str(n_epochs) + " Epochs (n=" + str(n_iter) + "): " + str(round(acc_mean, 2)))
    print("Median Accuracy of " + str(n_epochs) + " Epochs (n=" + str(n_iter) + "): " + str(round(acc_median, 2)))


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


def main():
    test_fxn(n_epochs=100, n_iter=10)


if __name__ == '__main__':
    main()
