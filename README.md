CMPSC 497 - Homework 1

Objective: Train a Logistic Regression or Support Vector Machine (SVM) classifier on word embeddings for a Part-of-Speech (POS) tagging task to achieve reasonable performance.

Dataset

Download the GloVe embeddings from the following link:

￼

Setup Instructions
	1.	Activate Virtual Environment:
Ensure you’re working within your Python virtual environment:

	source .venv/bin/activate


Install the necessary Python packages using pip:

	pip install numpy nltk pandas scikit-learn datasets

Note: Ensure that your Python version is compatible with the latest versions of these packages. For instance, scikit-learn requires Python 3.9 or newer.  ￼

After downloading the glove.6B.100d.txt file from the provided link, place it in your project directory.

Implementation

The implementation involves loading the GloVe embeddings, preparing the POS tagging dataset, and training a classifier. Ensure that the glove.6B.100d.txt file is correctly loaded and that the dataset is preprocessed appropriately before training the model.

For detailed guidance on installing and setting up these libraries, you can refer to the official documentation:
	•	NumPy:  ￼
	•	NLTK:  ￼
	•	scikit-learn:  ￼

By following these instructions, you should be able to set up your environment and proceed with the implementation of the POS tagging task using word embeddings.
