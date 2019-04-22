package com.david.email.classifier

import com.david.email.classifier.feature.{CountVectorizerSequenceBuilder, FeatureTransformer, FeatureTransformerSequenceBuilder, HashingTFSequenceBuilder}
import com.david.email.classifier.model._
import org.apache.log4j.Logger
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import scala.io.StdIn
import scala.util.{Failure, Success, Try}

object Main extends App {
  lazy val logger = Logger.getLogger(getClass)

  val filteredCol = "filtered"
  val featuresCol = "features"
  val labelCol = "label"

  logger.info(s"Starting Main")
  implicit val sparkSession = SparkSession.builder
    .master("local")
    .appName("Email-Classifier")
    .getOrCreate()

  import PromptUtils._

  prompt()
  print("This program build models and calculate for the selected model what are the most efficient parameters. ")

  /*
   * Function that does the initial transformation of the data. It does cleansing of the data, and make usage of the StopWordsRemover,
   * that removes the non useful words, as articles, prepositions...
   */
  private def initialTransformation(initial: Dataset[String], sparkSession: SparkSession): DataFrame = {
    //Funtion that converts a String into a tuple with 2 values, first is the label, and the second is all the SMS words.
    val initialFunc = (a: String) => (a.substring(0, a.indexOf("\t")).trim, a.substring(a.indexOf("\t") + 1).trim.toLowerCase)
    import sparkSession.implicits._
    val tupled = initial.map(initialFunc)
    //This function converts a String into an index value.
    val convertKeyToIndex = (key: String) => if (key.equals("ham")) 0 else if (key.equals("spam")) 1 else -1
    // This function converts a String into an array of words. As well it removes all the elements that are not letters and numbers.
    val convertValue = (s: String) => s.split(" ").map(_.replaceAll("[^A-Za-z0-9]", ""))
    //convert tupled dataset into a Label[Int], Features[Array[String]] dataset
    val indexedKey = tupled.map { case (key, value) => (convertKeyToIndex(key), convertValue(value)) }
    val df = indexedKey.toDF(labelCol, "raw")
    StopWordsRemover.loadDefaultStopWords("english")
    //Stop remover to apply into our dataframe. It requres to have as input a column with type Array[String]. The output will be a MLLib Vector.
    val remover = new StopWordsRemover()
      .setInputCol("raw")
      .setOutputCol("filtered")
    // Applying the StopWords remover
    val cleanedDF = remover.transform(df).select(labelCol, "filtered")
    //Caching this value, as it will be used by all of our computations.
    cleanedDF.cache
  }

  /*
   * Function that contains the logic that creates the Classifier. It reads the configuration values from Stdin.
   *
   * It asks the user to define which classification model would like to use: LogisticRegression or NaiveBayes.
   * It asks the user to define the model parameters ranges to use.
   */
  private def getClassifier: Classifier = {
    prompt()
    print(" Select the classification model. Press 0 for NaiveBayes or 1 for LogisticRegression: ")
    val input = Try(StdIn.readInt())
    input match {
      case Success(value) => value match {
        case 0 =>
          val smoothingOpt = getDoubleRange("Naive bayes smoothing double range. Press enter for defaults. Press any other key to configure it...")
          val modelsBuilder = NaiveBayesSequenceBuilder(featuresCol, labelCol).setSmoothing(smoothingOpt)
          println("\tModels setup: ")
          println(s"\t\t $modelsBuilder")
          new NaiveBayesClassifier(modelsBuilder.build())
        case 1 =>
          val maxIterListOpt = getIntRange("LogisticRegression maxIter int range. Press enter for defaults. Press any other key to configure it...")
          val regParamListOpt = getDoubleRange("LogisticRegression regParam double range. Press enter for defaults. Press any other key to configure it...")
          val elasticNetParamListOpt = getDoubleRange("LogisticRegression elasticNetParam double range. Press enter for defaults. Press any other key to configure it...")
          val thresholdListOpt = getDoubleRange("LogisticRegression threshold double range. Press enter for defaults. Press any other key to configure it...")
          val modelsBuilder = LogisticRegressionSequenceBuilder(featuresCol, labelCol).
            setElasticNetParamList(elasticNetParamListOpt).
            setMaxIterList(maxIterListOpt).
            setRegParamList(regParamListOpt).
            setThresholdList(thresholdListOpt)
          println("\tModels setup: ")
          println(s"\t\t $modelsBuilder")
          new LogisticRegressionClassifier(modelsBuilder.build())
        case _ =>
          prompt(false)
          print("Please type 0 for NaiveBayes or 1 for LogisticRegression")
          getClassifier
      }
      case Failure(exc) => {
        prompt(false)
        print("Please type 0 for NaiveBayes or 1 for LogisticRegression")
        getClassifier
      }
    }
  }

  /*
   * Function that contains the logic that creates the FeatureTransformer. It reads the configuration values from Stdin.
   *
   * It asks the user to define which feature transformer model would like touse: HashingTF or CountVectorizer.
   * It asks the user to define the model parameters ranges to use.
   */
  private def getFeatureTransformerSequenceBuilder: FeatureTransformerSequenceBuilder = {
    prompt()
    print(" Select the feature extractor model. Press 0 for HashingTF or 1 for CountVectorizer: ")
    val input = Try(StdIn.readInt())
    val output =input match {
      case Success(value) => value match {
        case 0 =>
          val numberOfFeaturesListOpt = getIntRange("HashingTF numberOfFeatures int range. Press enter for defaults. Press any other key to configure it...")
          val minDocFreqListOpt = getIntRange("HashingTF minDocFreqList int range. Press enter for defaults. Press any other key to configure it...")
          val modelsBuilder = HashingTFSequenceBuilder().
            setNumberOfFeaturesList(numberOfFeaturesListOpt).
            setMinDocFreqList(minDocFreqListOpt)
          modelsBuilder
        case 1 =>
          val numberOfFeaturesListOpt = getIntRange("CountVectorizer numberOfFeatures int range. Press enter for defaults. Press any other key to configure it...")
          val minDocFreqListOpt = getIntRange("CountVectorizer minDocFreqList int range. Press enter for defaults. Press any other key to configure it...")
          val modelsBuilder = CountVectorizerSequenceBuilder().
            setNumberOfFeaturesList(numberOfFeaturesListOpt).
            setMinDocFreqList(minDocFreqListOpt)
          modelsBuilder
        case _ =>
          prompt(false)
          print("Please type 0 for HashingTF or 1 for CountVectorizer")
          getFeatureTransformerSequenceBuilder
      }
      case Failure(exc) => {
        prompt(false)
        print("Please type 0 for HashingTF or 1 for CountVectorizer")
        getFeatureTransformerSequenceBuilder
      }
    }
    println("\tFeature Transformers setup: ")
    println(s"\t\t $output")
    output
  }
  val ftSeqBuilder = getFeatureTransformerSequenceBuilder
  val classifier = getClassifier
  val cleanedDF = initialTransformation(sparkSession.read.textFile("src/main/resources/SMSSpam"), sparkSession)
  classifier.getBestFeatureTransformerAndModel(cleanedDF, filteredCol, featuresCol, labelCol, ftSeqBuilder.build())
  logger.info(s"End Main")
}
