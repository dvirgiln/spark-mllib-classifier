package com.david.email.classifier.model

import com.david.email.classifier.feature.FeatureTransformer
import org.apache.log4j.Logger
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{DataFrame, SparkSession}

/*
 * The aim of this class is to get the best logistic regression model from a sequence of input models.
 * It iterates over all the model configurations and all the different Feature Transformers configurations  and find the best
 * FeatureTransformer configuration and the best LogisticRegression model configuration.
 */
class LogisticRegressionClassifier(models: Seq[LogisticRegression]) extends Classifier {
  lazy val logger = Logger.getLogger(getClass)


  /*
   * Function that displays the best model configuration/feature transformer configuration for an input dataset.
   */
  override def getBestFeatureTransformerAndModel(df: DataFrame, inputCol: String, outputCol: String, labelCol: String,
                                                 featureTransformerConfigs: Seq[FeatureTransformer])
                                                (implicit sparkSession: SparkSession): Unit = {
    //Initial value to be used in the fold operation.
    val initialValue = (Option.empty[FeatureTransformer], Option.empty[LogisticRegression], 0.0)
    //Iterate over the different feature transformers.
    val (bestFTOpt, bestMOpt, bestAcc) = featureTransformerConfigs.foldLeft(initialValue) { case ((accum, featureTransformer)) =>
      //Get the features from the feature transformer
      val features = featureTransformer.getFeatures(df, inputCol, outputCol)
      //Split the features into training and test
      val splits = features.randomSplit(Array(0.9, 0.1), seed = 11L)
      val training = splits(0).cache()
      val test = splits(1)
      //Now we fold over the different models, having as an input value the best accumulated value.
      models.foldLeft(accum) { case ((foldedAccum@(bestFeatureTransformer, bestModel, bestAccuracy), classificationModel)) =>
        val model = classificationModel.fit(training.toDF)
        val accuracy = calculateAccuracy(model, test)
        logger.info(s"Calculation: $featureTransformer accuracy=$accuracy threshold=${model.getThreshold} " +
          s"maxIter=${model.getMaxIter} regParam=${model.getRegParam}  elasticNetParam=${model.getElasticNetParam}")
        (bestFeatureTransformer, bestModel, bestAccuracy, accuracy) match {
          case (None, None, _, acc) => (Some(featureTransformer), Some(classificationModel), accuracy)
          case (Some(ft), Some(m), bestAcc, acc) if (acc > bestAcc) =>
            (Some(featureTransformer), Some(classificationModel), accuracy)
          case _ => foldedAccum
        }
      }
    }

    ((bestFTOpt, bestMOpt, bestAcc): @unchecked) match {
      case (Some(bestFt), Some(bestM), bestAcc) => logger.info(s"Best: $bestFt accuracy=$bestAcc " +
        s"threshold=${bestM.getThreshold} " + s"maxIter=${bestM.getMaxIter} regParam=${bestM.getRegParam}  " +
        s"elasticNetParam=${bestM.getElasticNetParam}")
      case (None, None, _) => logger.info("No feature transformer and model has been identified. Acc is 0.0. Check it out.")
    }
  }
}
