import autokeras as ak

# Initialize the text classifier.
clf = ak.TextClassifier(max_trials=30)  # Initialize the text classifier to try 30 models

# Feed the text classifier with training data.
clf.fit(x_train, y_train, epochs=20)  # Train each model for 20 epochs

# Get the best model
model = clf.export_model()