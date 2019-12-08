package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, IDFModel, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.{DataFrame, SparkSession}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    //Stage 0 : chargement du DF

    val df: DataFrame = spark.read.parquet("src/main/resources/prepared_trainingset")

    df.show()

    //Stage 1

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val df_tokenized = tokenizer.transform(df)

    //Stage 2

    val stopWords_remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    val df_filtered = stopWords_remover.transform(df_tokenized)

    //Stage 3

    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("vector")
      .fit(df_filtered)

    val df_vectorized = cvModel.transform(df_filtered)

    //Stage 4

    val idfModel: IDFModel = new IDF()
      .setInputCol("vector")
      .setOutputCol("tfidf")
      .fit(df_vectorized)

    val df_tfidf = idfModel.transform(df_vectorized)

    //Stages 5 & 6

    val country_indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val currency_indexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    val df_country_indexed = country_indexer.fit(df_tfidf).transform(df_tfidf)
    val df_indexed = currency_indexer.fit(df_country_indexed).transform(df_country_indexed)

    //Stages 7 & 8

    val onehot_encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    val df_encoded = onehot_encoder.fit(df_indexed).transform(df_indexed)

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot","currency_onehot"))
      .setOutputCol("features")

    val df_features = assembler.transform(df_encoded)

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    //Pipeline

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWords_remover, cvModel, idfModel, country_indexer, currency_indexer, onehot_encoder, assembler, lr))

    //Model training
    val splitted_df = df.randomSplit(Array(0.9,0.1))

    val training = splitted_df(0)
    val test = splitted_df(1)
    val model =  pipeline.fit(training)

    //Sauvegarde
    model.write.overwrite().save("src/main/resources/first-model")

    //Test
    val dfWithSimplePredictions = model.transform(test)

    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    println("f1-score du modèle simple = "+ evaluator.evaluate(dfWithSimplePredictions))

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10E-8,10E-6,10E-4,10E-2))
      .addGrid(cvModel.minDF,Array(55.0,75.0,95.0))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    println("Entraînement du modèle (cette étape est assez longue)")
    val modelGrid = trainValidationSplit.fit(training)

    //Sauvegarde du modèle entraîné sur la grille
    modelGrid.write.overwrite().save("src/main/resources/model_on_grid")

    //Test du modèle sélectionné
    val dfWithPredictions = modelGrid.transform(test)

    dfWithPredictions.groupBy("final_status", "predictions").count.show()

    println("f1-score du modèle entraîné sur la grille de paramètres = "+ evaluator.evaluate(dfWithPredictions))
  }

}
