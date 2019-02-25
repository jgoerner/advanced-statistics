# Advanced-statistics
Source code for the module "Advanced Statistics"

![science](https://media.giphy.com/media/MBVemoHuyw9Ik/giphy.gif)

# Structure
- [Readingguide I](https://github.com/jgoerner/advanced-statistics/blob/master/statistics/notebooks/Readingguide-1.ipynb) covers fundamentals of univariate & multivariate distributions
- [Readingguide II](https://github.com/jgoerner/advanced-statistics/blob/master/statistics/notebooks/Readingguide-2.ipynb) covers Gaussian Mixture Models and the general Expectation-Maximization-Algorithm
- [Readingguide III](https://github.com/jgoerner/advanced-statistics/blob/master/statistics/notebooks/Readingguide-3.ipynb) covers MCMC processes in  Bayesian statistics
- [Readingguide IV](https://github.com/jgoerner/advanced-statistics/blob/master/statistics/notebooks/Readingguide-4.ipynb) covers Probabilistic Programming


# Sample Content
##### Infered Mixture model
![](https://raw.githubusercontent.com/jgoerner/advanced-statistics/master/statistics/results/4-24-mixture-model.png)
##### Contour w/ levels
![](https://raw.githubusercontent.com/jgoerner/advanced-statistics/master/statistics/results/2-13-contour-2d.png)
##### Trace Plot
![](https://raw.githubusercontent.com/jgoerner/advanced-statistics/master/statistics/results/4-11-regression-half-cauchy.png)


# How to start the repository
1. create a file `jupyter.env` at `./config/`
2. put your secret Jupyter access token into `./config/jupyter.env` as `JUPTER_PASSWORD=...`
3. run `docker-compose up -d`
4. launch your webbrowser of choice <sup>*</sup>, open `http://localhost:8888` and start being awesome.

# Other Remarks
For an overview of statistical distributions, see [Distribution Cheatsheet](https://github.com/jgoerner/distribution-cheatsheet).

---
\* no, Internet Explorer is not a proper web browser...
