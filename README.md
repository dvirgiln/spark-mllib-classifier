# Spark MLLib Spam Classification finder
This repository contains some code examples about how to use the Spark MLLib library. The MLLib features that has been used are:
* Cleaning of the input data, using StopWordsRemover
* Usage of Feature Extractors: CountVectorizer, HashingTF and Word2Vec
* Usage of classification models: LogisticRegression and NaiveBayes.

## Goal
It is received as input a file that contains SMS data, classified by "spam" and "ham".

It is received as input a file that contains SMS data, classified by "spam" and "ham".

```
ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
ham	Ok lar... Joking wif u oni...
spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
...
```

The goal is to find a good model that can predict if an sms is spam or ham.

We want to try different classification models and  get the model/feature extractor parameters that perform better.

Instead of trying manually what are the best parameters, it has been create a suite of code that automates this logic.

## Explanation
Spark MLLib facilitates a lot the usage of Machine learning combined with dataframes. It is really easy to clean the data and transform it into features.

Then there are plenty of different models that can be used. In our case, for the input data that we have, it is a clear example of a classification problem.

For basic classification based on two categories, it is recommended to use NaiveBayes and LogisticRegression.

My initial approach was to first use the 2 different algorithm. Once I got some results I tried to tune the models to find out the best configuration for every model. Then I realised that I could automate the model tuning.

This code basically allows the user to try the different models with different Feature Extractors to find what is it the best configuration for:
* Feature Extractor Config: find out what it is the best parameters for the Feature Extractor selected by the user. Possible values are HashingTF and CountVectorizer.
* Classification Model Config: what are the best parameters for the classification model selected by the user. Possible values are LogisticRegression or NaiveBayes.

## Execution
To execute the program, it is just required to execute the Main class. It will require some human interaction though the console to define which model and which feature transformer model to use.

This is how the console interaction looks like:

```
prompt> This program build models and calculate for the selected model what are the most efficient parameters.
prompt>  Select Feature Transformer model. Press 0 for HashingTF or 1 for CountVectorizer: 1
prompt> CountVectorizer numberOfFeatures int range. Press enter for defaults. Press any other key to configure it...s
prompt> Enter Range Initial Value(Int): 800
prompt> Enter Range End Value(Int): 850
prompt> Enter Range By Value(Int): 50
prompt> CountVectorizer minDocFreqList int range. Press enter for defaults. Press any other key to configure it...s
prompt> Enter Range Initial Value(Int): 7
prompt> Enter Range End Value(Int): 15
prompt> Enter Range By Value(Int): 1
	Feature Transformers setup:
		 CountVectorizerSequenceBuilder(Range 800 to 850 by 50,Range 7 to 15)

prompt>  Select the classification model. Press 0 for NaiveBayes or 1 for LogisticRegression: 1
prompt> LogisticRegression maxIter int range. Press enter for defaults. Press any other key to configure it...sdf
prompt> Enter Range Initial Value(Int): dsa
prompt> Range Initial Value is not an Integer. Please retry.
prompt> Enter Range Initial Value(Int): 100
prompt> Enter Range End Value(Int): 100
prompt> Enter Range By Value(Int): 10
prompt> LogisticRegression regParam double range. Press enter for defaults. Press any other key to configure it...
prompt> LogisticRegression elasticNetParam double range. Press enter for defaults. Press any other key to configure it...
prompt> LogisticRegression threshold double range. Press enter for defaults. Press any other key to configure it...s
prompt> Enter Range Initial Value(Double): 0.5
prompt> Enter Range End Value(Double): 0.5
prompt> Enter Range By Value(Double): 0.1
	Models setup:
		 LogisticRegressionSequenceBuilder(features,label,Range 100 to 100 by 10,NumericRange 0.0 to 0.2 by 0.1,NumericRange 0.0 to 0.2 by 0.1,NumericRange 0.5 to 0.5 by 0.1)
```


## Conclusions
These are the results I found for 3 different executions:
```
Best: CountVectorizerFeatureTransfomer(800,8)  naiveBayesSmoothing=4.200000000000001  accuracy=99.42857142857143
Best: HashingTFFeatureTransfomer(850,10)  naiveBayesSmoothing=1.0  accuracy=92.95238095238095
Best: CountVectorizerFeatureTransfomer(950,5) LogisticRegression accuracy=98.28571428571429 threshold=0.5 maxIter=100 regParam=0.1  elasticNetParam=0.0
```
As you can notice the NaiveBayes is the more accurate model. Accuracy is 99.42857142857143. The configuration used is:

* Feature Extractor: CountVectorizer with vocabSize=800 and minDF=8.
* Classification Model: NaiveBayes with smoothing=4.2

In this case NaiveBayes performs better than the LogisticRegression model. What I have seen is that if you do not tune the LogisticRegression with some specific values, it never detects spam. But with the correct configuration it can have a 98.28% accuracy.

The features extractor it is really important, as that it converts the initial dataframe into features that can be consumed by the classification model. As part of the tests, it is concluded that CountVectorizer performs better than HashingTF.

About the Words2Vec, apparently to have good results, it is required to have a bigger dataset. I tried to use it, but NaiveBayes didnt like, as some of the feature values are negatives, and apparently NaiveBayes requires positive values as features.
