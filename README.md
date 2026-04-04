# Tokopedia Review Analysis (Sentiment + Topic Modeling)

## Project Overview
This project analyzes customer reviews from Tokopedia to understand:
- Customer sentiment (positive, neutral, negative)
- Key issues and themes using topic modeling
- Patterns in review behavior

The final output is an interactive dashboard built with Streamlit. Nah, I failed to deploy it online. Well, pity of me. Any suggestions or corrections are really appreciated. Please, and thanks. 
---

## Dataset & Preprocessing

## Dataset
I used two dataset from Kaggle: 
1. [Tokopedia Product Reviews 2025](https://www.kaggle.com/datasets/salmanabdu/tokopedia-product-reviews-2025) dataset that contain around 65k samples of product reviews scraped from Tokopedia (in Indonesian of course); it contains 13 columns including review text, date, shop and product information, rating, and sentiment label. 
2. [kamus-alay](https://www.kaggle.com/datasets/oktasn/kamus-alay) dataset that contain informal Indonesian slang words to normalize text into standard Indonesian language. It wasn't 100 percent suitable for this case, but hey what's that horrific things over there??? 

### Data Cleaning
- Removed duplicate reviews
- Reduced excessive positive samples to mitigate imbalance (Actually, because i can't stand the training time. Yeah, it's not the best but whatever.)
- Final dataset still highly skewed (~92.6% positive)

### Text Preprocessing
Steps applied:
- Lowercasing
- Remove links, HTML tags, punctuation, newlines, words containing numbers, single characters, and extra spaces. 

Notes:
- No lemmatization (too computationally expensive -- as i said it earlier)
- No stopword removal (some stopwords carry meaning in context -- tho I'm not sure: like the word `tidak` or `enggak` that may imply the negative tone of a sentence)

### Additional Cleaning
- Removed empty text (0 words)
- Normalized slang words

---

## Exploratory Data Analysis

### Review Length
- 99% of reviews have < 60 words (cleaned one)
- Median length ≈ 9 words

### Most Common Words   
Including **stopwords** we have: 

**Positive Reviews**
- dan, cepat, sesuai, barang, bagus, pengiriman

**Negative Reviews**
- yang, tidak, di, enggak, sudah, dan, saya

**Neutral Reviews**
- yang, dan, di, tidak, ada, tapi, enggak

---

## Modeling (Sentiment Classification)

### Setup
- TF-IDF (5000 features)
- N-grams: (1,2)
- Train-test split: 80/20 (stratified)

### Models Used
- Naive Bayes (tuned with GridSearchCV)
- Logistic Regression
- HistGradientBoosting

---

## Evaluation Insights
- Accuracy is misleading due to class imbalance
- Macro F1-score used as main metric

### Results
- Logistic Regression ≈ Naive Bayes
- HistGradientBoosting performs slightly better on minority classes
- However, training time is ~20x longer

### Final Choice
✅ Logistic Regression (best trade-off between performance and efficiency)

---

## Probability Adjustment (Improvement)

Two approaches tested:

### Case 1
Adjust using class prior `p/prior`: 
- Improves recall (negative & neutral)
- Reduces positive class performance
- No overall F1 improvement

### Case 2 (Chosen)
Adjust using `p / ((prior + 0.5) / 2)`:  
- Improves recall & F1 (negative & neutral)
- Maintains strong performance on positive class
- Overall F1 increased from **0.51 → 0.59**

---

## Topic Modeling

### Method
- LDA (Latent Dirichlet Allocation)
- TF-IDF vectorizer:
  - max_df = 0.95
  - min_df = 5
  - ngram (1,2)
- Indonesian + custom + domain stopwords

### Approach
- 1 vectorizer
- Separate LDA models for:
  - Positive reviews
  - Negative reviews

### Topics Identified
1. Product Quality
2. Delivery & Packaging
3. Product Match / Expectation

---

## Dashboard Features

### 1. Overview
- Total reviews
- Sentiment distribution
- Quarterly negative trend
- Review length distribution

### 2. Topic Insights
- Distribution of topics
- Key themes:
  - Product quality
  - Delivery & packaging
  - Product match / expectation

### 3. Review Explorer
- User input text
- Sentiment prediction (with adjusted probability)
- Topic inference

---

## Tech Stack
- Python
- Scikit-learn
- Pandas / NumPy
- Matplotlib, Seaborn
- Streamlit

---

## Key Takeaways
- Handling class imbalance is critical in real-world datasets
- Accuracy alone is not a reliable metric
- Simple models (Logistic Regression) can outperform complex ones when optimized properly
- Topic modeling provides additional business insights beyond sentiment classification

---

## Miscellaneous

### A Log   
**Should've made this cleaner, but oh gosh**
1. Wed, 11 march: plan, ideas, brainstorming, get the data: tokped
2. Fri, 13 march: copasting from kaggle's notebooks, chatting with GPT, trying in kaggle doing raw reviews, do text cleaning, stuck at stopwords and lemmatization
3. Mon, 23 march: explore #2, local, done classif, yet to change the viz and model eval 
4. What to do now, 25 march: time to refactor, but not yet ready  — how about doing more. A quick model eval → let’s say it’s so easy. Topic modeling → kinda done well well well well well well well 
5. 2 april: Building dashboard i guess
6. 3 april – Refactor things. From playground notebooks → into eda, modeling, evaluation. To chatgpt → uhm what  now? Oh the conclusion.
7. 4 april -- Lastly – do the markdown and else. Explain and showcase. 

### Oh waht? 
- READme viz
    - overview dashboard -- main screen
    - topic insights -- something visual 
    - (optional) model comparison chart
    - i guess we don't need viz's viz?
    - avoid: too many charts, wordcloud spam, raw eda plots.... (sounds too much like me) 
- GO visual in PAGEs! (oh, so add images after explanation eh?)
    - after problem info: the dashboard ......
    - after sentimetn discussion: sentiment dist
    - after topic modeling explanation: topic what??? 
    - explorer: try it yourself!
    - else: overview, sentiment dist, negative trend chart, topic insights, explorer UI, still avoid messy EDA

### Author's Note. 
1. use kaggle or colab
2. download the notebooks & model to local
3. evaluate and make the app. LOL. 
4. we DON't have to train the model in our burik local computer.
5. and we AREN't that great anyway.

```
# as a starter
mkdir house-price-prediction
cd house-price-prediction

# init the git, then
mkdir app data models notebooks outputs scripts

# create venv
python -m venv .venv
source .venv/Scripts/activate #or .venv/bin/activate
python -m pip install ipykernel [list of packages]
```

### For useless LinkedIn poset: 
1. hook, what u did, key insights: bullet, what makes it interesting, CTA
2. 1/2 images only: dashboard overview & maybe explorer UI

```
- 92% of customer reviews are positive... but is that really true? 
- I built a sentiment + topic analysis dashboard using Tokopedia reviews (Although fail to deploy online. Sigh!)
- I found that
 - Accuracy is misleading due to imbalance
 - Logistic Regression outperformed complex models in efficiency
 - Topic modeling revealed 3 key issues: product quality, delivery & packaging, and expectation mismatch 
- I also adjusted prediction probabilities to improve minorities class detection
- Try the app / check the repo / kill ys!
```