import json
import socket
import pandas as pd


# Prints a pandas dataframe in its entirely, instead of truncating rows with ellipses
def full_print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


# Converts a standard dictionary to a pandas dataframe
def format_df(raw_dict):
    df = pd.DataFrame.from_dict(raw_dict, orient='index')
    # Some dictionary elements are mis-formatted because of improperly received messages (zero or multiple data points)
    # This handles such exceptions and lets the malformed tuple through (it becomes discarded during analysis)
    try:
        df = df[df['timestamp'] != 0]
    except KeyError as e:
        cols = ['timestamp', 'data', 'label', 'predicted']
        df = pd.DataFrame(columns=cols)
        return df
    # Fills the column of predictions with rest state default
    df['predicted'] = "REST"

    return df


# Compute the lagged activation status for previous row(s), and fill the value for the current row
def apply_active_lag(df):
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


# Compute the arithmetic difference of the current row from the previous row
def apply_diff_calc(df):
    for i in range(1, len(df)):
        df.loc[i, 'diff'] = df.loc[i, 'data'] - df.loc[i - 1, 'data']
    return df


# Compute if the current data value crossed 0 from the last data value, and fill the values
# This metric is not used in the current heuristic, but could potentially be useful
def apply_zero_threshold(df):
    for i in range(1, len(df)):
        if df.loc[i, 'data'] < 0 < df.loc[i - 1, 'data']:
            df.loc[i, 'zero_threshold'] = 1
        elif df.loc[i, 'data'] > 0 > df.loc[i - 1, 'data']:
            df.loc[i, 'zero_threshold'] = 1
        else:
            df.loc[i, 'zero_threshold'] = 0
    return df


# Apply the heuristic rules for making a prediction of a rest or activation state
# This is the final function that gets applied to the buffer
def apply_heuristic(row):
    # Large spikes not preceded by activation are predicted as rest
    if row['data'] >= 80000 and row['active_lag'] == 0:
        return "REST"
    # Initial entry is predicted as rest
    # (Not always true, but we weight lagged activation and can't afford to start the series with a false positive)
    elif row['diff'] == 0:
        return "REST"
    # Medium positive change in direction with positive data is activation
    elif row['diff'] > 10000 and row['data'] > 0:
        return "ACTIVATION"
    # Medium negative change in direction and previous activation is activation
    elif row['diff'] > -5000 and row['active_lag'] > 0:
        return "ACTIVATION"
    # Medium positive data or medium negative data is activation
    # This rule is problematic and requires more conditions to reduce false positives
    elif row['data'] > 15000 or row['data'] < -10000:
        return "ACTIVATION"
    # If no other rules are triggered, data is predicted as rest
    else:
        return "REST"


def heuristic(df):
    # Initializes new columns:
    # 1. The numerical difference data(n) - data(n-1)
    # 2. A lagged categorical variable for activation (values 0-2), to retain some memory of earlier predicted state
    # 3. A boolean variable for whether or not a data point crossed the x-axis from the n-1th data point
    df['diff'], df['active_lag'], df['zero_threshold'] = 0, 0, 0

    # Apply three functions that compute the above-initialized columns that will be used in the final apply_heuristic()
    df = apply_diff_calc(df=df)
    df = apply_active_lag(df=df)
    df = apply_zero_threshold(df=df)

    # Predict activation status based on the columns data, diff, and active_lag
    df['predicted'] = df.apply(apply_heuristic, axis=1)
    df = apply_active_lag(df=df)
    return df


# Returns a message to the TCP server if the predicted label is activated
def output_signal(df, socket_client):
    for index, row in df.iterrows():
        if row['predicted'] == 'ACTIVATION':
            socket_client.sendall(b'Activation classified\n')


# Input is a dictionary generated from the raw JSON data and the socket connection
# Populates a dataframe, applies the heuristic rule set, and calls the function to send a message for each activation
def read_buffer(buffer_dict, socket_client):
    df = format_df(raw_dict=buffer_dict)
    df = heuristic(df=df)
    output_signal(df=df, socket_client=socket_client)


# Opens the connection to the TCP server, and continuously ingests data in small buffers to analyze and label
def connect_socket():

    # Continuous loop that will not be broken unless the server closes the connect or a keyboard interruption is sent
    while True:
        # Initialize a counter and empty dictionary that get reset for each buffer
        i = 0
        d = {}

        # Initialize a socket connection on the specified port 7890
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', 7890))

        # Inner continuous loop to generate a buffer where n=10
        while True:
            # Store the message received from the server, with a maximum size of 1024 bytes
            data = s.recv(1024)
            # When the counter reaches the buffer size (10), exit the inner continuous loop
            if i == 10:
                break
            # If the TCP server stops sending populated messages, exit the inner continuous loop
            if not data:
                break
            # Naively handling any messages that did not separate newlines
            # This is based on the assumption that false positives are less preferable than false negatives
            # With more time, a separate function to parse malformed messages could be executed instead of breaking
            try:
                formatted_data = json.loads(data)
            except json.decoder.JSONDecodeError as e:
                break

            # Populate the dictionary at index=i with the JSON-formatted data point {data:[], label:[], timestamp:[]}
            d[i] = formatted_data
            i += 1

        # Calls the function that handles formatting the dictionary, predicting labels, and sending back messages
        read_buffer(buffer_dict=d, socket_client=s)

