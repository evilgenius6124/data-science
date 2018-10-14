# only important correlations and not auto-correlations
	threshold = 0.7
	important_corrs = (cor_mat[abs(cor_mat) > threshold][cor_mat != 1.0]) \
		.unstack().dropna().to_dict()
	unique_important_corrs = pd.DataFrame(
		list(set([(tuple(sorted(key)), important_corrs[key]) \
		for key in important_corrs])), columns=['attribute pair', 'correlation'])
	# sorted by absolute value
	unique_important_corrs = unique_important_corrs.ix[
		abs(unique_important_corrs['correlation']).argsort()[::-1]]
	unique_important_corrs
	
	
	
.apply(lambda x : format(x, 'f'))   to supress scientific notation from describe()