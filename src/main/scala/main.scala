import java.util.logging.{FileHandler, Logger, SimpleFormatter}
import java.io.{File, PrintWriter}
import breeze.linalg.{DenseVector, DenseMatrix, csvread}
import breeze.stats.regression.leastSquares
import breeze.linalg.sum
import breeze.numerics.pow
import breeze.stats.mean


object Main extends App {

  System.setProperty(
    "java.util.logging.SimpleFormatter.format",
    "%1$tF %1$tT %4$s %5$s%6$s%n"
  )

  if (args.length == 0) {
    println("Mate, i need at least one parameter and it's better if it's a path to diabetes.csv")
  }

  val trainFileName = args(0)
  val testFileName = args(1)
  val outputFileName = args(2)

  val logger = Logger.getLogger("Regression app")
  val handler = new FileHandler("regression.log")
  val formatter = new SimpleFormatter()
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  logger.info(f"trainFileName: ${trainFileName}, testFileName: ${testFileName}, outputFileName: ${outputFileName}")

  val file: File = new File(trainFileName)
  val data: DenseMatrix[Double] = csvread(file, skipLines=1)
  val matrix = data.toDenseMatrix
  val y: DenseVector[Double] = matrix(::, -1)
  val ones: DenseVector[Double] = DenseVector.ones(data.rows)
  val X = DenseMatrix.horzcat(new DenseMatrix(data.rows,1, ones.toArray), matrix(::, 0 until data.cols - 1))

  val result = leastSquares(X, y)
  val predict = X * result.coefficients

  //  result.rSquared
  val r2_train = 1 - (sum(pow(y - predict, 2)) / sum(pow(y - mean(predict), 2)))
  logger.info(f"R2 on train: ${r2_train}%.3f")

  val testDataFile: File = new File(testFileName)
  val testData: DenseMatrix[Double] = csvread(testDataFile, skipLines = 1)
  val testMatrix = data.toDenseMatrix
  val y_test: DenseVector[Double] = testMatrix(::, -1)
  val X_test = DenseMatrix.horzcat(new DenseMatrix(testData.rows,1, ones.toArray), testMatrix(::, 0 until testData.cols - 1))
  val predict_test = X_test * result.coefficients

  val r2_test = 1 - (sum(pow(y_test - predict_test, 2)) / sum(pow(y_test - mean(predict_test), 2)))
  logger.info(f"R2 on test: ${r2_test}%.3f")
  val writer = new PrintWriter(new File(outputFileName))

  for (el <- predict_test) {
    writer.write(f"$el\n")
  }
  writer.close()
}
