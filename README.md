# tcp-classifier
## Prerequisites
### Python Depencencies
In order to execute the program, you will need a working Python interpreter (> 3.6) with a few non-standard libraries.
You should already have an installed version of Python 3. From a terminal, you can check your version and interpreter location with:
    
    $ python3 —version && which python3

#### OS X
If the above command returns an error or a version less than 3.6.0, you can use [homebrew](https://brew.sh/) to install 
a more recent version (for directions on installing hombrew, see [here](https://docs.brew.sh/Installation)):
    
    $ brew install python3
    $ brew install pip3

#### Linux
Use your system's package manager (apt-get, pacman, etc.) to install:
    
    $ sudo pacman -S python python-pip

### Library Dependencies
To execute the main program, the only necessary libraries to import are 'json', 'socket', and 'pandas'. Of these, only pandas 
is a non-standard library.
To install the libraries for exploratory.py, use the Python package manager pip3 (some systems may alias as simply 'pip'):
    
    $ sudo pip install matplotlib pandas seaborn

## Executing the Program
From the directory 'tcp-classifier/', enter the following command to start up the TCP server:

    $ java -jar ./pison-challenge-0.2-all.jar

From the same directory, enter the following command in a separate terminal (some systems may alias as simply 'python'):

    $ python3 main.py

As directed by the project requirements, there will be no stdout in the terminal window executing the Python code. However, 
you will see the TCP server output "ACTIVATED" messages as the TCP client labels the activation state of incoming data.

## File Structure
### exploratory.py
This file is where I initially tested the socket client and heuristics. The goal is to be able to test methods on 
"frozen" data, to digest the real-time data from the TCP server and store it temporarily in Pandas dataframes to experiment 
with classification strategies more easily.

I purposefully left this file messy to make my process observable. It shows where I started with the problem. If you'd rather 
not install the graphing library seaborn, the below images show my workflow for experimenting with labeling. In the first 
image, I kept track of ideas that I had for label rules while looking at the graphed true labels. In the second image, the 
generated graphs show the line plot of true signals above, and my predicted labels vs. true labels on scatter plots below. 
In the third image, I have a graph of the line plot of true signals above, and my correct vs. incorrect labels below.
[Project Whiteboard](https://raw.githubusercontent.com/samlarson/tcp-classifier/master/docs/tcp_whiteboard.jpg)
[Project Workflow 1](https://raw.githubusercontent.com/samlarson/tcp-classifier/master/docs/tcp_workflow_1.png)
[Project Workflow 2](https://raw.githubusercontent.com/samlarson/tcp-classifier/master/docs/tcp_workflow_2.png)

### intermediary.py
The purpose of this file is to test any heuristics developed in exploratory.py. There are two main distinguishing features 
compared to exploratory.py. First, I am timing the total elapsed time for each iteration. Second, I have set two parameters 
for the function that connects to the TCP socket. The parameter 'n_iter' is the length of the buffer which gets converted 
to a dataframe, and 'n_epochs' is the number of times the buffer will be generated. After the specified number of epochs have 
been run, the mean and median accuracy from every buffer is computed and output to the terminal. This allowed me to experiment 
with the how the size of buffer affected accuracy, as well as the actual accuracy of a given heuristic when run many more times 
than in exploratory.py.

### tcp_classifier.py
This is the actual code submission, taking heuristic rules derived during exploration and applying them to the real-time data. 
The code has been commented and shortened for readability.

### main.py
This is the file that executes tcp_classifier.py. If in future iterations there were multiple Python files that needed to 
interact with each other, they would be imported here. You could also run the program from the tcp_classifier.py file itself, 
centralizing project execution here reduces complexity if I continue to work on the project.

## Design Considerations
### Simple Heuristic vs. ML Classifier
Taking the recommendation from the challenge prompt, I opted to write my own set of rules to evaluate the data based on visualizations 
in the interest of time. 

### Performance - Accuracy & Speed
There were three iterations of heuristics completed for the project before a final ruleset was decided on, located in 
exploratory.py. Their accuracy was as follows:

* Heuristic 1: 68%
* Heuristic 2: 76%
* Heuristic 3: 79%
* Final Heuristic: 80%

These scores were calculated with intermediary.py, checked with the following parameter configurations: 
* n_epochs=100 & n_iters=100
* n_epochs=100 & n_iters=1000
* n_epochs=1000 & n_iters=10

One interesting observation is that while the first three heuristics scored consistently at any of the above specified 
parameter values (for both mean and median accuracy), the final heuristic median accuracy varied noticeably with lower n_iters, 
hovering between 76%-90%.

Measuring program speed in intermediary.py, total elapsed time from buffer ingestion to label output averaged 0.24 seconds 
with a buffer size of 100 data points, and 0.024 seconds with a buffer size of 10 data points (the final buffer sized used). 
The following screenshot shows intermediary.py in action.

[Heuristic Performance](https://raw.githubusercontent.com/samlarson/tcp-classifier/master/docs/heuristic_performance.png)

### Buffer & Epoch Size
While the epoch size did not affect measured performance after a sufficient value (~100), buffer size certainly did. This 
demonstrates the heuristic's sensitivity to memory in the time series. While a buffer size of 100 pinned mean and median 
accuracy around 79%, reducing the buffer size to 10 increased mean accuracy but destabilized median results, as mentioned 
in the above section. The takeaway from this is that the extra memory with larger buffer sizes was not necessary to label 
the data, but additional rules would have to be added to the heuristic to reduce deviation in accuracy over the long-run. 
One other aspect not mentioned in the prompt but with real-world importance are latency requirements. Increased execution time 
by a factor of 10 linearly due to buffer size is probably not acceptable when digesting real-time data in production, so I 
decided to keep a lower buffer size for realism and accept the trade-off in variance.

### Handling Exceptions
There are only two exceptions handled in tcp_classifier.py. A 'JSONDecodeError' in connect_socket() will simply discard the 
data, and a 'KeyError' in format_df() will similarly ignore that row when generating predicted labels. 

### False Positives
One flaw that becomes clear after looking at several graphs of the current predictions is that the ruleset does not reliably 
label data points in the 15,000 - 20,000 'data' range, especially when a small incremental positive difference is followed 
by a large spike. The vast majority of the ~20% incorrect predictions occur here. Because I factor in the lagged activation 
in the rule set, any singular mis-labeled data before a large spike essentially guarantees a series of two to five incorrect 
predictions. I suspect additional rules factoring in features for crossing the x-axis, lagged small oscillations, or moving 
averages may alleviate this problem.

## Moving Forward
If I were to work on this project for longer, I have a few additional ideas to explore. I would add additional scores to 
be returned by the function measure_performance() besides accuracy. For example, I could measure the differences in performance 
based on the window size of the buffer, a matrix of the rules triggered for each data point to minimize overlap, and the cost 
in time for each function as it is changed. The most important feature I would add to the function would be a confusion matrix 
to evaluate how false positives and negatives change with difference heuristics.

To achieve higher performance, I would experiment with machine learning classifiers. The PyTorch framework provides enough out-of-the-box 
support for common models that with another day or two I could most likely train a model with comparable or higher performance. 
Analyzing the rest/activation signals from graphed data by eyeballing it was quick and relatively effective, but after a 
couple hours there were no other rules for me to easily write based on graph interpretation. In this stage I would also generate 
more feature columns, such as the unused 'zero_threshold'.

With more time, I would also change some of the design decisions that I made. For example, instead of discarding malformed 
data from the server (multiple non-separated messages), I would catch those exceptions with a separate function to unload into 
multiple JSON objects.
