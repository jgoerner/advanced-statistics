# 1-zimmer whg, preis < 300
print("1-room apt: {}".format(norm.cdf(x=300, loc=500, scale=70)))

# 2-zimmer whg, preis < 300
print("2-room apt: {}".format(norm.cdf(x=300, loc=750, scale=90)))

# x-zimmer whg, preis > 700
CDF = 0.0
for pi, n in zip(pis, N):
    CDF += pi * n.cdf(700)
print("X-room apt: {}".format(1-CDF))