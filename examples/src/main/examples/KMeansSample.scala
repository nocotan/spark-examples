import org.apache.spark.ml.clustering.KMeans

object KMeansSample {
  def run() {
    val dataset = spark.read.format("libsvm").load("/opt/spark/data/mllib/sample_kmeans_data.txt")

    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(dataset)

    val WSSSE = model.computeCost(dataset)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
  }
}

