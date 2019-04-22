package com.david.email.classifier.model

import org.apache.spark.ml.classification.NaiveBayes

import scala.collection.immutable.NumericRange
/*
  Builder class that builds a sequence of different NaiveBayes models.
 */
case class NaiveBayesSequenceBuilder(featuresCol: String, labelCol: String, naiveBayesModelTypeList: List[String] = ("multinomial" :: Nil),
                                     smoothingList: NumericRange.Inclusive[BigDecimal] = (BigDecimal(1) to 5 by 0.2)) {

  def setSmoothing(smoothingListOpt: Option[NumericRange.Inclusive[BigDecimal]]) = {
    smoothingListOpt match {
      case Some(value) => this.copy(smoothingList = value)
      case None => this
    }
  }

  def build(): Seq[NaiveBayes] = {
    //Computes all the different combinations of input parameters
    val configs = for {
      smoothing <- smoothingList
      naiveBayesModelType <- naiveBayesModelTypeList

    } yield (smoothing, naiveBayesModelType)

    configs.map { case (smoothing, naiveBayesModelType) =>
      // Creates a new NaiveBayes model
      new NaiveBayes()
        .setModelType(naiveBayesModelType)
        .setFeaturesCol(featuresCol)
        .setLabelCol(labelCol)
        .setSmoothing(smoothing.toDouble)
    }
  }
}
