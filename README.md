# stockHMM
In many real world scenarios it is of interest to model some observable signal to better understand its underlying properties. For example, in the realm of speech processing we may have a spectrum of frequencies of some word or phrase captured over a period of time. Discretize these spectrums into finite sized continuous observation vectors using a decompression method and denote the sequences of vectors as the stochastic process {O_t}. It is of interest to identify the actual word or phrase spoken given a sequence of observations. For example, in the case of determining a singular word we may choose to decompose a recording of that word into a set of time-indexed spectral vectors {O_1, O_2, ... , O_T}. Then, we may describe some underlying embedded stochastic process that produced those signals as being some chain of phonems corresponding to that word. The application of the HMM aims to capture both 1. the number of possible phonems and 2. a characterization of what these hidden phonems (or more generally, whatever is producing the signals) are doing over time. We can do so by assigning the underlying process some number of distinct states with associated state transitions i.e. a generic markov model. Then, we attach probability distributions to each state that generate the observed signals (in this case the stochastic process of spectral vectors {O_t}), as well as attach an initial state distribution to the model itself. It remains to adjust the parameters of this model to maximize its likelihood of producing the observations. This is done with an EM procedure, where we solve a constrained optimization problem for a lower bound on the log of the likelihood. 

Define 2 states:
- profitable/bull
- unstable/bear

Using yfinance to fetch stock data and ta to compute technical indicators, define the signal vectors as a list of various preprocessed stock attributes and indicators. 

We use the open-source library hmmlearn to implement a GMM approach. That is, fit gaussian mixtures to each state's emission density and use EM to maximize the likelihood of the model producing the time-indexed stochastic process of observable signals. 

We derive intial estimates by repeatedly fitting random initializations of model parameters and selecting the parameters corresponding to the highest log likelihood. Since we random initialize, I do not assign fixed labels to each state but rather let the state that "learns" the larger 5d return be the profitable state. Let the 5d return of a state be the expected of the means of its emission density's mixture's returns. 

Then, we take the fitted models (trained with stock data from 2010 - mid 2021) and backtest on a time-disjoint set of signals from late 2021 to the present to avoid future leakage in the tests. 

The backtest procedure is as follows:
(1) Initialize a window of observation signals with predetermined size at the beginning of the test set.
(2) Compute a most likely state sequence given the model on that window using the Viterbi algorithm.
(3) Let the last state of the Viterbi denote the predicted state at that time step 
(4) "Slide" the window one time step and repeat (2) - (3) until the training set has been covered.
(5) Compute the cumulative return over the period following the naiive strategy: Buy the close and sell tomorrow's close if the state is profitable, and do nothing if it is not. 

(Notice that in the above procedure we miss some signals as we should avoid using data with which the model was trained on)

Once the cumulative return using the naiive strategy has been determined, we can gauge it against other strategies to determine if alpha can be generated. 

Using a data plotting library we can visualize the effectiveness of the strategy and adjust the model to our liking. 

### Examples

# Train AMD, Test AMD
<img width="597" height="549" alt="image" src="https://github.com/user-attachments/assets/492382a3-13d0-42ee-9938-5d6e593a89a8" />

<img width="585" height="148" alt="image" src="https://github.com/user-attachments/assets/358b079a-df47-4ede-a0a2-5c6092d208ec" />

<img width="694" height="191" alt="image" src="https://github.com/user-attachments/assets/f378f401-2b0f-4df1-a476-06392e2407f6" />

****stats over 1151 trading days****
trained with 2009-10-15 to 2020-09-17.
backtested with 2020-12-14 to 2025-07-16.
cumulative return of 74.95%.
annualized return of 13.03%.
there was a 50.83% chance you bought on a particular day.
sharpe ratio of: 0.73

Baseline returns:
return captured from 2020-12-14 to 2025-07-16.
cumulative return of 64.19%.
annualized return of 11.47%.
if you had bought/sold everyday: 69.80%.

SPY returns:
return captured from 2020-12-14 to 2025-07-16.
cumulative return of 82.88%.
annualized return of 14.13%.

base seed for replication: 30352
logistic regression acc vs hmm: 0.8876961189099918

# Train AMD, Test TSLA
<img width="597" height="549" alt="image" src="https://github.com/user-attachments/assets/8d4e66e9-a475-4be1-8a7e-07185b1efbda" />

<img width="585" height="148" alt="image" src="https://github.com/user-attachments/assets/a2c4ddd1-040e-473e-8768-73af163aa69f" />

<img width="694" height="191" alt="image" src="https://github.com/user-attachments/assets/ca147e8c-c237-4af3-9698-b56618284947" />


****stats over 911 trading days****
trained with 2010-09-29 to 2021-08-31.
backtested with 2021-11-26 to 2025-07-16.
cumulative return of 341.63%.
annualized return of 50.81%.
there was a 47.31% chance you bought on a particular day.
sharpe ratio of: 1.68

Baseline returns:
return captured from 2021-11-26 to 2025-07-16.
cumulative return of -11.29%.
annualized return of -3.26%.
if you had bought/sold everyday: -14.00%.

SPY returns:
return captured from 2021-11-26 to 2025-07-16.
cumulative return of 42.89%.
annualized return of 10.38%.

base seed for replication: 30352
logistic regression acc vs hmm: 0.878475798146241
ALPHALPAALPHALPAALPHALPA
you beat the naiive strategy by 355.63%

# Train AMD, Test LOGI
<img width="597" height="549" alt="image" src="https://github.com/user-attachments/assets/fe1233f6-9c87-4fa8-bb08-c52c9684d4e9" />

<img width="585" height="148" alt="image" src="https://github.com/user-attachments/assets/5e176dbd-a43c-45e3-92e0-a1a7fa277b4b" />

<img width="694" height="191" alt="image" src="https://github.com/user-attachments/assets/e6fd8821-d785-4260-b3ae-5b38084c7871" />

****stats over 1151 trading days****
trained with 2009-10-15 to 2020-09-17.
backtested with 2020-12-14 to 2025-07-16.
cumulative return of 99.09%.
annualized return of 16.27%.
there was a 54.47% chance you bought on a particular day.
sharpe ratio of: 0.97

Baseline returns:
return captured from 2020-12-14 to 2025-07-16.
cumulative return of 13.26%.
annualized return of 2.76%.
if you had bought/sold everyday: 12.76%.

SPY returns:
return captured from 2020-12-14 to 2025-07-16.
cumulative return of 82.87%.
annualized return of 14.13%.

base seed for replication: 30352
logistic regression acc vs hmm: 0.8744838976052849
ALPHALPAALPHALPAALPHALPA
you beat the naiive strategy by 86.33%

### Summary
Overall, the GMM is a fascinating way to model shifts in some latent state of a stock. 

In some cases the model performs very well, beating the baseline and sometimes the S&P500. However, the model is often limited by a variety of factors

Fitting model parameters to one stock does not make it universally applicable, as those stocks may have different patterns of volatility, more pronounced trends, etc. 

In the case of a low number of mixtures the GMM will often degenerate into a simple thresholding model that checks if certain entries of the signal go above a specific value. We can attribute this to the fact that with a small number of mixtures the emissions become simple Gaussians, causing them to form two distinct "clouds" in the space of possible observations. Then the most likely state is just the one that the current observation is closest to, which negates the ability to probabilistically model the current state. 

The model with the naiive strategy of buying and selling next day does not perform well for long term investments. We attribute this to the fact that once the model's parameters have been set, they do not change throughout the entirety of the testing. This highlights the importance of stationarity in the signals that we use, as trends in the signal complete negate the initial learned parameter values. Alternatively, one could bypass this by using online learning to adapt to changes in market structures i.e. relearning the model parameters as time goes on. 

From the examples you can see that I did not necessarily model bull/bear states, but rather a perceived potential for profit at the current time given fixed window of signals from the past to now. One could turn this into a bull/bear regime model by increasing this window size, as well as picking more stable and smoother indicators. Methods can be used, such as explicit duration densities, to prevent highly oscillatory state transitions and better model extended durations of bull and bear markets. 

I used a naiive strategy in all tests by simply buying and selling the next day, but more complex methods can be used in application. 

### References
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257–286. https://doi.org/10.1109/5.18626

