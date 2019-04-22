package com.david.email.classifier.feature

/*
  Builder class that builds a sequence of different Count Vectorizer.
 */
case class CountVectorizerSequenceBuilder(numberOfFeaturesList: Range = (700 to 1000 by 50),
                                          minDocFreqList: Range = (5 to 15)) extends FeatureTransformerSequenceBuilder {
  def setNumberOfFeaturesList(numberOfFeaturesListOpt: Option[Range]): CountVectorizerSequenceBuilder = {
    numberOfFeaturesListOpt match {
      case Some(value) => this.copy(numberOfFeaturesList = value)
      case None => this
    }
  }

  def setMinDocFreqList(minDocFreqListOpt: Option[Range]): CountVectorizerSequenceBuilder = {
    minDocFreqListOpt match {
      case Some(value) => this.copy(minDocFreqList = value)
      case None => this
    }
  }

  def build(): Seq[FeatureTransformer] = {
    val configs = for {
      numberOfFeatures <- numberOfFeaturesList
      minDocFreq <- minDocFreqList
    } yield (numberOfFeatures, minDocFreq)
    configs.map { case (numberOfFeatures, minDocFreq) =>
      CountVectorizerFeatureTransfomer(numberOfFeatures, minDocFreq)
    }
  }
}