# Spam-Classification


# Requirement

```
numpy
sklearn
```


# Import essential library


```python
import numpy as np
```

# Loading the data and split it into dataset and label


```python
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
```

# Preprocessing the data
### The given data has different data range. So we have to standardize and normalize the data.


```python
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
```

# Support Vector Machine Classifier
### linear kernel gives the best result.


```python
def svm_model():
	from sklearn.svm import SVC
	model = SVC(C = 1.5, kernel = 'linear', gamma = 'auto')
	return model
```

# Multi-layer Perceptron Classifier
### Perform better than the svm classifier.


```python
def NN_model():
	from sklearn.neural_network import MLPClassifier
	model = MLPClassifier(activation='relu', solver='adam', alpha=0.0001, max_iter=200, learning_rate_init=0.001)
	return model
```

# Print the table of Accuracy, Precision, Recall, FP Rate, FN Rate, Overall Error Rate and Average Error Rate


```python
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
```

# Save Model


```python
def save_model(model):
	from sklearn.externals import joblib
	joblib.dump(model, 'model.pickle')
```

# Main Program


```python
def main():
	X, y = load_data("spambase.data")
	X = preprocessing(X)
	### k-fold
	k = 10
	### choose from svm and NN
	model = svm_model()
	# model = NN_model()
	from sklearn.model_selection import cross_validate
	from sklearn.metrics import make_scorer, accuracy_score
	from sklearn.metrics import confusion_matrix
	### compute tn, fp, fn, tp
	def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
	def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
	def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
	def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
	scoring = {'Accuracy': 'accuracy', 'Precision': 'precision', 'Recall': 'recall',
				'tp': make_scorer(tp), 'tn': make_scorer(tn),
				'fp': make_scorer(fp), 'fn': make_scorer(fn)}
    ### This cross-validation object is a variation of KFold that returns stratified folds. 
    ### The folds are made by preserving the percentage of samples for each class.
	cv_results = cross_validate(model, X, y, scoring = scoring, cv = k, n_jobs = -1,
						return_train_score=False)
	table(cv_results, k, 5)
	
	### save model
	# save_model(svm_model)
	
if __name__ == '__main__':
	main()
```

# SVM

    +----------+-----------+---------+---------+---------+--------------------+------------+
    | Accuracy | Precision |  Recall | FP Rate | FN Rate | Overall Error Rate | Average ER |
    +----------+-----------+---------+---------+---------+--------------------+------------+
    | 0.92842  |  0.93064  | 0.88462 | 0.04301 | 0.11538 |      0.07158       |   0.0748   |
    | 0.94143  |   0.9235  | 0.92857 | 0.05018 | 0.07143 |      0.05857       |   0.0748   |
    | 0.93709  |   0.9322  | 0.90659 | 0.04301 | 0.09341 |      0.06291       |   0.0748   |
    | 0.93913  |  0.93714  | 0.90608 | 0.03943 | 0.09392 |      0.06087       |   0.0748   |
    | 0.94348  |   0.9235  |  0.9337 | 0.05018 |  0.0663 |      0.05652       |   0.0748   |
    |  0.9413  |   0.885   |  0.9779 | 0.08244 |  0.0221 |       0.0587       |   0.0748   |
    | 0.95217  |  0.97041  | 0.90608 | 0.01792 | 0.09392 |      0.04783       |   0.0748   |
    | 0.93696  |  0.91304  | 0.92818 | 0.05735 | 0.07182 |      0.06304       |   0.0748   |
    | 0.87582  |    0.81   | 0.89503 | 0.13669 | 0.10497 |      0.12418       |   0.0748   |
    | 0.85621  |  0.83237  | 0.79558 | 0.10432 | 0.20442 |      0.14379       |   0.0748   |
    +----------+-----------+---------+---------+---------+--------------------+------------+

# Neural Network Classifier

    +----------+-----------+---------+---------+---------+--------------------+------------+
    | Accuracy | Precision |  Recall | FP Rate | FN Rate | Overall Error Rate | Average ER |
    +----------+-----------+---------+---------+---------+--------------------+------------+
    | 0.94143  |  0.94286  | 0.90659 | 0.03584 | 0.09341 |      0.05857       |  0.06109   |
    | 0.94794  |  0.93407  | 0.93407 | 0.04301 | 0.06593 |      0.05206       |  0.06109   |
    | 0.95445  |   0.9548  | 0.92857 | 0.02867 | 0.07143 |      0.04555       |  0.06109   |
    |  0.9413  |  0.93258  | 0.91713 | 0.04301 | 0.08287 |       0.0587       |  0.06109   |
    | 0.96087  |  0.95531  | 0.94475 | 0.02867 | 0.05525 |      0.03913       |  0.06109   |
    | 0.93696  |  0.88384  | 0.96685 | 0.08244 | 0.03315 |      0.06304       |  0.06109   |
    | 0.94783  |  0.95906  | 0.90608 | 0.02509 | 0.09392 |      0.05217       |  0.06109   |
    | 0.94783  |  0.93855  | 0.92818 | 0.03943 | 0.07182 |      0.05217       |  0.06109   |
    | 0.91068  |  0.86458  | 0.91713 | 0.09353 | 0.08287 |      0.08932       |  0.06109   |
    | 0.89978  |  0.89941  | 0.83978 | 0.06115 | 0.16022 |      0.10022       |  0.06109   |
    +----------+-----------+---------+---------+---------+--------------------+------------+

