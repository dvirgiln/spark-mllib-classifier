package com.david.email.classifier.feature

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

/*
  Trait that defines the operation of a Feature Transformer
 */
trait FeatureTransformer {
  def name: String

  /*
   * Having a dataframe as input that contains an input column with words,
   * it converts that non featured column into a featured-style column.
   */
  def getFeatures(df: DataFrame, inputColumn: String, outputColumn: String): DataFrame
}

/*
  Class that applies the HashingTF algorithm to an input dataframe to convert a column with words into a vector of features.
 */
case class HashingTFFeatureTransfomer(numFeatures: Int, minDocFreq: Int) extends FeatureTransformer {
  override def name: String = "Hashing TF"

  override def getFeatures(df: DataFrame, inputColumn: String, outputColumn: String): DataFrame = {
    val hashingTF = new HashingTF()
      .setInputCol(inputColumn).setOutputCol(s"raw_$outputColumn").setNumFeatures(numFeatures)

    val features = hashingTF.transform(df)
    val idf = new IDF().setInputCol(s"raw_$outputColumn").setOutputCol(outputColumn).setMinDocFreq(minDocFreq)
    val idfModel = idf.fit(features)
    idfModel.transform(features)
  }
}

/*
  Class that applies the CountVectorizer algorithm to an input dataframe to convert a column with words into a vector of features.
 */
case class CountVectorizerFeatureTransfomer(vocabSize: Int, minDF: Int) extends FeatureTransformer {
  override def name: String = "Count Vectorizer"

  override def getFeatures(df: DataFrame, inputColumn: String, outputColumn: String): DataFrame = {
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol(inputColumn)
      .setOutputCol(outputColumn)
      .setVocabSize(vocabSize)
      .setMinDF(minDF)
      .fit(df)

    cvModel.transform(df)
  }
}





