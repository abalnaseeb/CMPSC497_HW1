# CMPSC 497 - Homework 1
## Ali AlNaseeb

### **Objective**

Train a Logistic Regression or Support Vector Machine (SVM) classifier on word embeddings for a Part-of-Speech (POS) tagging task to achieve reasonable performance.

### **Dataset**

Download the GloVe embeddings from the following link:

	https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt
 OR
 
 	https://nlp.stanford.edu/data/glove.6B.zip

After downloading the **glove.6B.100d.txt** file, place it in your project directory.

Setup Instructions

Follow the instructions based on your operating system.

## **For macOS & Linux**

Open Terminal and navigate to your project directory:

	cd path/to/your/project


Create & activate a virtual environment:

	python3 -m venv .venv
	source .venv/bin/activate


Install the required dependencies:

	pip install numpy nltk pandas scikit-learn datasets


Ensure that your Python version is compatible (Python 3.9 or newer recommended).
Verify Installation:

	python -c "import numpy; import nltk; import pandas; import sklearn; print('All packages installed successfully!')"

## **For Windows (CMD or PowerShell)**
Open Command Prompt or PowerShell and navigate to your project directory:

	cd path\to\your\project


Create & activate a virtual environment:

	python -m venv .venv
	.venv\Scripts\activate


Install the required dependencies:

	pip install numpy nltk pandas scikit-learn datasets


Ensure that your Python version is compatible (Python 3.9 or newer recommended).
Verify Installation:

	python -c "import numpy; import nltk; import pandas; import sklearn; print('All packages installed successfully!')"

Implementation Steps
	1.	Load GloVe Embeddings
	•	Ensure that glove.6B.100d.txt is correctly placed in your project directory.
	•	Load the embeddings into a dictionary.
	2.	Prepare the POS Tagging Dataset
	•	Use NLTK or another dataset source to preprocess the dataset.
	3.	Train a Classifier
	•	Train either a Logistic Regression or SVM classifier on word embeddings.
	4.	Evaluate Performance
	•	Use appropriate evaluation metrics to measure model performance.

References

For detailed guidance on installing and setting up these libraries, you can refer to their official documentation:
	•	NumPy
	•	NLTK
	•	scikit-learn

This should provide clear setup instructions for Windows, macOS, and Linux users while keeping everything structured and easy to follow.
