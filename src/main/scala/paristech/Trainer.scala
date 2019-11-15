package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, IDFModel, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
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

    val df: DataFrame = spark.read.parquet("df_saved/prepared_trainingset")

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

    df_indexed.select("country2","country_indexed","currency2","currency_indexed").show()

    //Stages 7 & 8

    val onehot_encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    val df_encoded = onehot_encoder.fit(df_indexed).transform(df_indexed)

    df_encoded.select("country2","currency2","country_indexed", "currency_indexed","country_onehot", "currency_onehot").show()

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot","currency_onehot"))
      .setOutputCol("features")

    val df_features = assembler.transform(df_encoded)

    df_features.select("features").show(false)

  }
}
