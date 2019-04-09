import numpy as np


def load_data(file):
	lines = []
	with open(file, 'r') as f:
		for line in f.readlines():
			line = line.strip().split(',')
			lines.append(line)
	lines = np.array(lines).astype(np.float32)
	dataset = lines[...,0:57]
	label = lines[...,57].astype(np.int8)
	return dataset, label

def preprocessing(data):
	### preprocessing the data to [0, 1]
	from sklearn import preprocessing
	scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(data)
	data = scaler.transform(data)
	### normalization
	data = preprocessing.normalize(data, norm='l2')
	### PCA: the result shows no improvement
	# from sklearn.decomposition import PCA
	# data = PCA(n_components = 57).fit_transform(data)
	return data

### This cross-validation object is a variation of KFold that returns stratified folds. 
### The folds are made by preserving the percentage of samples for each class.
def model_selection(data, label, k):
	from sklearn.model_selection import StratifiedKFold
	skf = StratifiedKFold(n_splits = k)
	skf.get_n_splits(data, label)
	return skf

def svm_model():
	from sklearn.svm import SVC
	model = SVC(C = 1.5, kernel = 'linear', gamma = 'auto')
	return model

def NN_model():
	from sklearn.neural_network import MLPClassifier
	model = MLPClassifier(activation='relu', solver='adam', alpha=0.0001, max_iter=200, learning_rate_init=0.001)
	return model

def table(cv_results, k, d):
	tp = cv_results['test_tp']
	fp = cv_results['test_fp']
	tn = cv_results['test_tn']
	fn = cv_results['test_fn']
	from prettytable import PrettyTable
	t = PrettyTable()
	t.add_column("Accuracy", np.around( cv_results['test_Accuracy'] , decimals = d))
	t.add_column("Precision", np.around( cv_results['test_Precision'] , decimals = d))
	t.add_column("Recall", np.around( cv_results['test_Recall'] , decimals = d))
	### the false positive rate is the fraction of non-spam testing examples that are misclassified as spam
	### fp / ( fp + tn ) 
	t.add_column("FP Rate", np.around( fp / ( fp + tn ) , decimals = d))
	### the false negative rate is the fraction of spam testing examples that are misclassified as nonspam
	### fn / ( fn + tp ) = 1 - recall rate
	t.add_column("FN Rate", np.around( fn / ( fn + tp ) , decimals = d))
	### the overall error rate is the fraction of overall examples that are misclassified. 
	t.add_column("Overall Error Rate", np.around( 1 - cv_results['test_Accuracy'] , decimals = d))
	average_error = np.round(np.average(1 - cv_results['test_Accuracy']), decimals = d)
	t.add_column("Average ER", [average_error]*k)
	print(t)

def save_model(model):
	from sklearn.externals import joblib
	joblib.dump(model, 'model.pickle')


def main():
	X, y = load_data("spambase.data")
	X = preprocessing(X)
	k = 10

	model = svm_model()
	model = NN_model()
	from sklearn.model_selection import cross_validate
	from sklearn.metrics import make_scorer, accuracy_score
	from sklearn.metrics import confusion_matrix
	def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
	def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
	def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
	def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
	scoring = {'Accuracy': 'accuracy', 'Precision': 'precision', 'Recall': 'recall',
				'tp': make_scorer(tp), 'tn': make_scorer(tn),
				'fp': make_scorer(fp), 'fn': make_scorer(fn)}
	cv_results = cross_validate(model, X, y, scoring = scoring, cv = k, n_jobs = -1,
						return_train_score=False)

	table(cv_results, k, 5)
	
	### save model
	# save_model(svm_model)
	



if __name__ == '__main__':
	main()



### old school way for K ford cross validation
	# skf = model_selection(X, y, 5)
	# for train_index, test_index in skf.split(X, y):
	# 	X_train, X_test = X[train_index], X[test_index]
	# 	y_train, y_test = y[train_index], y[test_index]
	# 	model = svm_model()
	# 	model.fit(X_train, y_train)
	# 	score = model.score(X_test, y_test)
	# 	# print(len(X_train))
	# 	print ("score =", score)