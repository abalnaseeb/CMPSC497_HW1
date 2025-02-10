CMPSC 497 - Homework 1

Author: Ali AlNaseeb

Objective

Train a Logistic Regression or Support Vector Machine (SVM) classifier on word embeddings for a Part-of-Speech (POS) tagging task to achieve reasonable performance.

Dataset

Download the GloVe embeddings from one of the following sources:
	•	Kaggle: GloVe 6B 100d Dataset
	•	Stanford NLP: GloVe 6B.zip

After downloading, extract glove.6B.100d.txt and place it in your project directory.

Setup Instructions

For macOS & Linux:
1.	Activate the Virtual Environment:

	source .venv/bin/activate


2.	Install Required Packages:

pip install numpy nltk pandas scikit-learn datasets


3.	Ensure Python Version Compatibility:
•	Ensure that Python 3.9 or newer is installed to avoid compatibility issues with scikit-learn and other dependencies.

For Windows:
1.	Activate the Virtual Environment:
   .venv\Scripts\activate


3.	Install Required Packages:

pip install numpy nltk pandas scikit-learn datasets


	3.	Ensure Python Version Compatibility:
	•	Check your Python version by running:

python --version


	•	If needed, install Python 3.9 or newer from python.org.

Implementation

1. Load GloVe Embeddings:

Ensure the glove.6B.100d.txt file is correctly loaded from your project directory.

2. Prepare the POS Tagging Dataset:

Use the Hugging Face Datasets Library to load the batterydata/pos_tagging dataset.

3. Train the Model:

Train a Logistic Regression or SVM classifier using word embeddings and POS tagging labels.

Reference Links
	•	NumPy Installation: NumPy Documentation
	•	NLTK Installation: NLTK Documentation
	•	scikit-learn Installation: scikit-learn Documentation

By following these instructions, you should be able to set up your environment and proceed with the implementation of the POS tagging task using word embeddings. 🚀
