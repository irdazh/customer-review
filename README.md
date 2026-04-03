# Customer Review 
NLP things

- I'll copy from the website or so lol.
- So, what's in here? 
- Oh gosh, it'll be super complicated. I hate local things. Should've just done it earlier, I guess. Nope?  
- 


Nothing, lol. 

## Dataset
Dataset used was [Tokopedia Product Reviews 2025](https://www.kaggle.com/datasets/salmanabdu/tokopedia-product-reviews-2025) from Kaggle dataset. I also use [kamus-alay](https://www.kaggle.com/datasets/oktasn/kamus-alay) dataset that contain informal Indonesian slang words to normalize text into standard Indonesian language. It wasn't 100 percent suitable for this case, but what's that? 

## A Log







## Author's Note. 
1. change the word counter viz --- no need to use plotly 
2. basic, light modeling: using local
3. complete modeling: using kaggle or colab whatsoever. 
4. and download the model, and just use them lol
5. not so training in here hehe. 


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

### to chatgpt
1. no duplicate reviews
2. due too many dataset --> reduce the positive sentiment
3. then we get 92.6% of positive sentiment
4. mostly are makanan&minuman, and olahraga --- tho it's still useless for now. 

1. to text and lowercase
3. remove links 4. remove html tags 5. remove punctuation 
6. remove newlines 7. remove words containing numbers 
7. remove single characters 8. remove extra spaces
5. no lemmatization since it took too long
6. no stopwords removal since some of them actually important (i guess)
7. for cleaned words, length mostly (99 percent) below 60, with median just around 9. 

More Cleaning
1. drop 0 word
2. normalized from slang words

More eda, most common words
1. positive: dan, cepat, sesuai, barang, bagus, pengiriman
2. negative: yang, tidak, di, enggak , sudah, dan, saya
3. neutral: yang, dan, di, tidak, ada, tapi, enggak

Modeling
1. read clean data
2. model 1: sentiment classification (pos, neu, net)
3. using tfidf: 5k features with ngrams 1,2 
4. train test: 80/20, stratified
5. 3 model used: nb, lr, and histgradient
6. as for nb need to choose the parameter var_smoothing --> use gridsearch
7. get the result

Kinda conclusion
- accuracy is pretty useless: all has similar accuracy (due to highly imbalanced dataset)
- rather we could see from macro averaged f1-score (all label are considered equal): lr and nb got similar performance, while hg is a bit better, especially dealing with minority classes (negative and neutral)
- but, considering it took 20 times longer training using hgb than lr and nb, then 
- let's choose lr instead for future use: in case need a retraining, well that's that

A bit of improvement, on lr model
- case 1: dividing the predict proba with their prior probability for extreme adjustment 
- case 2: dividng the predict proba with mean (prior probability and 0.5) to imitate MID-method in threshold moving. 
- case 1: increase recall in negative and neutral class, increase f1 score in neutral class but decrease f1 score in positive class, overall, the same overall f1 score
- case 2: increase recall and f1 score in negative and neutral class, f1 score in positive class stay the same, increase in overall f1 score from 0.51 to 0.59
- we will implement case 2 with 0.54 in precision and 0.53 in recall for negative class. pretty good i guess. 
- since we don't really care about neutral class, and then for positive class it still has a good accuracy 0.97 precision, and 0.98 recall. 

Topic modeling
1. Nah, just use LDA for now (kapan2 use BERTopic)
2. Using vectorizer, with max_df 0.95, and min_df = 5, with indonesian stopwords, added with custom stopwords and domain stopwords, 
an ngram range 1, 2
3. Vectorizer fit transform to all data, 
4. but then transform again for negative text sentiment --> do LDA for negative 
5. and also transform again for positive text --> do LDA for positive
6. So 1 vectorizer and 2 lda model. 
7. 3 topics: product quality, delivery & packaging, and product matching


App contain: 
1. overview: total reviews, pos, and negative; sentiment distribution; quarterly negative trend; and review length distribution
2. topic insights: product quality, delivery and packaging, product match --- with pretty similar proportion 25 to 40 percent thingy
3. explorer: where user can input text to analyze and get the sentiment and also the topic. 
