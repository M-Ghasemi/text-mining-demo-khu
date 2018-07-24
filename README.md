# Text Analytics with Python

This project is based on the "Text Analytics with Python" book and it's created with demonstration and educational purposes.

course: Computational Data Mining

University: Kharazmi

## Getting Started

You can execute "text_minig_presentation.py" with python or "text_minig_presentation.ipynb" with jupyter notebook.

If you just want to see the demo results without executing, then open text_mining_presentation.html.

This demo contains:

* **Preprocessing  (tokenization, lemmatization, stemming, stop words elimination, etc)**

* **Vector space modeling (TF-IDF)**

* **Text querying (a smart way)**

* **Topic modeling (LSI, NMF)**

Unfortunately, the functions have no documentation but the names are self-explanatory.

## Installing

I suppose you have python3.5, pip3 and git installed on a linux system. It's better to use a virtual environment to avoid further conflicts with system wide packages.

**Step 1**: Get a copy of project:

* **```git clone https://github.com/M-Ghasemi/text-mining-demo-khu.git```**

**Step 2**: Install required packages with executing the following line. make sure that you are in the same directory with requirements file.

* **```pip install -r requirements.txt --no-index --find-links```**


**Step 3**: run ipython and install nltk required files (you can also install all files by ```nltk.download('all')```)

* **```import nltk```**
* **```nltk.download('punkt')```**
* **```nltk.download('stopwords')```**
* **```nltk.download('wordnet')```**
* **```nltk.download('averaged_perceptron_tagger')```**

**Step 4**: run jupyter notebook and click on the "text_mining_presentation"

* **```jupyter notebook```**

## Authors

* **[Mohammad Sadegh Ghasemi](https://www.linkedin.com/in/mohammad-sadegh-ghasemi-40)**
