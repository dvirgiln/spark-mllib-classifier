package com.david.email.classifier.feature

/*
  Trait that defines the build operation that returns a sequence of FeatureTransformer objects.
  The feature transformer could be either a HashingTF or a CountVectorizer feature transformer.
 */
trait FeatureTransformerSequenceBuilder {
  def build(): Seq[FeatureTransformer]
}
