# Python
Quantitative Finance Code in Python
The Pricing Library conatains classes to value Lookback Options, Binary or Digital Options and Barrier Options.All the options can be valued using Monte Cartlo Simulations and Analytcial or Closed Form Solutions.

Lookback Options will take inputs as Strike, Spot, Interest Rate, Volatility, Time, Number of Simulations and Number of Steps. For Floating lookback options input of strike is required, you can input any value and it shouldn't effect the results.

The valuation uses Geometric Brownian Motion to value the options as below:

<img src="https://latex.codecogs.com/gif.latex?S&space;=&space;rdt&plus;\sigma&space;\phi&space;\sqrt{dt}" title="S = rdt+\sigma \phi \sqrt{dt}" />
