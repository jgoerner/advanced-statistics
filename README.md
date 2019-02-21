# Advanced-statistics
Source code for the module "Advanced Statistics"

# Structure
- [Readingguide I](https://github.com/jgoerner/advanced-statistics/blob/master/statistics/notebooks/Readingguide-1.ipynb) covers fundamentals of univariate & multivariate distributions
- [Readingguide II](https://github.com/jgoerner/advanced-statistics/blob/master/statistics/notebooks/Readingguide-2.ipynb) covers Gaussian Mixture Models and the general Expectation-Maximization-Algorithm
- [Readingguide III]() covers MCMC processes in  Bayesian statistics
- [Readingguide IV]() covers Probabilistic Programming

# How to start the repository
1. create a file `jupyter.env` at `./config/`
2. put your secret Jupyter access token into `./config/jupyter.env` as `JUPTER_PASSWORD=...`
3. run `docker-compose up -d`
4. launch your webbrowser of choice <sup>*</sup>, open `http://localhost:8888` and start being awesome.

# Other Remarks
For an overview of statistical distributions, see [Distribution Cheatsheet](https://github.com/jgoerner/distribution-cheatsheet).

---
\* no, Internet Explorer is not a proper web browser...
