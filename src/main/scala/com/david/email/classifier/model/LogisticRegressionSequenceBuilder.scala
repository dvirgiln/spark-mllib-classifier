package com.david.email.classifier.model

import org.apache.spark.ml.classification.LogisticRegression

import scala.collection.immutable.NumericRange

/*
  Builder class that builds a sequence of different LogisticRegression models.
 */
case class LogisticRegressionSequenceBuilder(featuresCol: String, labelCol: String, maxIterList: Range = (100 to 110 by 10),
                                             regParamList: NumericRange.Inclusive[BigDecimal] = (BigDecimal(0.0) to 0.2 by 0.1),
                                             elasticNetParamList: NumericRange.Inclusive[BigDecimal] = (BigDecimal(0.0) to 0.2 by 0.1),
                                             thresholdList: NumericRange.Inclusive[BigDecimal] = (BigDecimal(0.0) to 0.2 by 0.1)) {

  def setRegParamList(regParamListOpt: Option[NumericRange.Inclusive[BigDecimal]]): LogisticRegressionSequenceBuilder = {
    regParamListOpt match {
      case Some(value) => this.copy(regParamList = value)
      case None => this
    }
  }

  def setElasticNetParamList(elasticNetParamListOpt: Option[NumericRange.Inclusive[BigDecimal]]): LogisticRegressionSequenceBuilder = {
    elasticNetParamListOpt match {
      case Some(value) => this.copy(elasticNetParamList = value)
      case None => this
    }
  }

  def setThresholdList(thresholdListOpt: Option[NumericRange.Inclusive[BigDecimal]]): LogisticRegressionSequenceBuilder = {
    thresholdListOpt match {
      case Some(value) => this.copy(thresholdList = value)
      case None => this
    }
  }

  def setMaxIterList(maxIterListOpt: Option[Range]): LogisticRegressionSequenceBuilder = {
    maxIterListOpt match {
      case Some(value) => this.copy(maxIterList = value)
      case None => this
    }
  }

  def build(): Seq[LogisticRegression] = {
    //Computes all the different combinations of input parameters
    val configs = for {
      maxIter <- maxIterList
      regParam <- regParamList
      elasticNetParam <- elasticNetParamList
      threshold <- thresholdList
    } yield (maxIter, regParam, elasticNetParam, threshold)

    configs.map { case (maxIter, regParam, elasticNetParam, threshold) =>
      // Creates a new LogisticRegression model
      new LogisticRegression()
        .setMaxIter(maxIter)
        .setRegParam(regParam.toDouble)
        .setElasticNetParam(elasticNetParam.toDouble).setThreshold(threshold.toDouble)
    }
  }
}
