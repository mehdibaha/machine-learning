import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
for a in range(1, 50):
	alpha = a/10
	print(f"alpha: {alpha}")
	regr = linear_model.Lasso(alpha=alpha)

	# Train the model using the training sets
	regr.fit(diabetes_X_train, diabetes_y_train)
	# Make predictions using the testing set
	diabetes_y_pred = regr.predict(diabetes_X_test)
	'''
	print("Training error: %.2f"
		  % mean_squared_error(diabetes_y_train, diabetes_y_pred))
	print('Training variance: %.2f' % r2_score(diabetes_y_train, diabetes_y_pred))
	'''
	print("Test error: %.2f"
		  % mean_squared_error(diabetes_y_test, diabetes_y_pred))
	print('Test variance: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
	print()
