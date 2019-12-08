package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._
    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/
    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("src/main/resources/train_clean.csv")

    println("\n")

    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    println("\n")

    df.show()

    println("\n")

    df.printSchema()

    println("\n")

    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline", $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()

    println("\n")

    dfCasted
      .select("goal", "backers_count", "final_status","disable_communication")
      .describe()
      .show

    //Etude de la colonne disable_communication
    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)

    //On remarque que les valeurs "normales" de cette colonne sont False et True

    dfCasted.filter($"disable_communication"=!="False").filter($"disable_communication"=!="True").show()

    //On voit que les entrees qui ne sont ni a True ni a False sont des entrées invalides
    //Elles sont complètement décalées vers la droite, donc sans statut final
    //On remarque aussi qu'il y a tres peu de lignes a True. On peut donc filtrer les lignes invalides et supprimer cette colonne
    val df2: DataFrame = dfCasted
      .filter(($"disable_communication"==="False" || $"disable_communication"==="True"))
      .drop("disable_communication")

    //Il ne semble pas y avoir de duplicats dans les donnees puisqu'a nom égales, les campagnes ont des moments de lancement différents.
    df2.groupBy("name","launched_at").count().sort($"count".desc).show()

    df2.filter($"name".contains("Canceled")).count()
    //On remarque qu'il y a 4309 kickstarters qui ont été annulés

    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    //Pour la partie sur les country/currency, j'ai un peu triché en supprimant les lignes "problématiques" depuis le début puisqu'elles
    //ne contiennent pas de statut final. Après ce filtrage, il n'y a plus de problèmes de colonnes décalées (et le nombre de colonnes
    //supprimées ainsi (environ 700) est négligeable).

    val df3 : DataFrame = dfNoFutur
      .withColumn("days_campaign",datediff(from_unixtime($"deadline"),from_unixtime($"launched_at")))
      .withColumn("hours_prepa",hour(from_unixtime($"launched_at"-$"created_at")))

    val df_pruned : DataFrame = df3.drop("launched_at").drop("created_at").drop("deadline")

    val df_str_min : DataFrame = df_pruned
      .withColumn("name", lower(col("name")))
      .withColumn("desc", lower(col("desc")))
      .withColumn("keywords", lower(col("keywords")))

    df_str_min.show()

    val df_final : DataFrame = df_str_min
      .withColumn("text",concat(col("name"),lit(" "),col("desc"),lit(" "),col("keywords")))

    df_final.filter(df_final("days_campaign").isNull
      || df_final("days_campaign").isNaN
      || df_final("days_campaign")===""
      || df_final("hours_prepa").isNull
      || df_final("hours_prepa").isNaN
      || df_final("goal").isNull
      || df_final("goal").isNaN
      || df_final("goal")===""
      || df_final("country").isNull
      || df_final("country").isNaN
      || df_final("country")===""
      || df_final("currency").isNull
      || df_final("currency").isNaN
      || df_final("currency")==="").show()

    //Aucune valeur nulle dans ce DF !

    df_final.write.mode("overwrite").parquet("src/main/resources/preprocessed")
  }

}