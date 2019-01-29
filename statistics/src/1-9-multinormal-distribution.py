from scipy.stats import multinomial

# Build Distribution
mn = multinomial(n=5, p=[12/25, 8/25, 5/25])

# Find probability
print(mn.pmf([3,2,0]))

# Sample
print("\t\tWI\tIS\tTI")
for idx in range(1,16):
    [wi, its, ti] = mn.rvs()[0]
    print("Sample {idx}: \t {wi}\t {its} \t {ti}".format(**locals()))