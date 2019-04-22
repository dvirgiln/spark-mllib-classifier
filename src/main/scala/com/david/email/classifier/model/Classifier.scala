package com.david.email.classifier.model

import com.david.email.classifier.feature.FeatureTransformer
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.linalg.{SparseVector, Vector}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/*
 * Contains the common code used by all the classificators. Available Classificators are LogisticRegression and NaiveBayes.
 */
trait Classifier {
  def getBestFeatureTransformerAndModel(df: DataFrame, inputCol: String, outputCol: String,
                                        labelCol: String, featureTransformerConfigs: Seq[FeatureTransformer])
                                       (implicit sparkSession: SparkSession): Unit

  /*
   * Function that calculates the accuracy of a fitted model.
   * It requires as inputs, the trained model and a Dataset that contains tests features.
   */
  def calculateAccuracy[M <: ProbabilisticClassificationModel[Vector, M]](model: ProbabilisticClassificationModel[Vector, M],
                                                                          test: Dataset[Row],
                                                                          initialValueCol: String ="initial",
                                                                          predictedCol: String = "predicted",
                                                                          featuresCol: String = "features",
                                                                          labelCol: String = "label")
                                                                         (implicit sparkSession: SparkSession): Double = {

    import sparkSession.implicits._
    //For all the test data it predicts the value. The output dataframe contains 2 columns, one with the initial label and
    val predictionsDF = test.map(a => (a.getAs[Integer](labelCol), model.predict(a.getAs[SparseVector](featuresCol))))
    //Converting from dataset into a dataframe and putting correct labels. Defining it as a temp view to allow querying using SQL notation.
    predictionsDF.toDF(initialValueCol, predictedCol).createOrReplaceTempView("Result")
    val total = sparkSession.sql("SELECT COUNT(*) FROM Result").collect()(0).getLong(0)
    val spam = sparkSession.sql(s"SELECT COUNT(*) FROM Result where $predictedCol = $initialValueCol").collect()(0).getLong(0)
    (spam.toDouble / total.toDouble) * 100
  }

}
