from scipy import stats

x = [0.2545, 0.254125, 0.25575, 0.2535, 0.2535]
y = [0.464, 0.460125, 0.466, 0.4665, 0.475875]
print(stats.ttest_rel(x, y).pvalue)