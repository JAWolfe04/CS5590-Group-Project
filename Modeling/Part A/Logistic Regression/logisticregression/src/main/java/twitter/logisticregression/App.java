package twitter.logisticregression;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class App 
{
    public static void main( String[] args )
    {
    	SparkSession spark = SparkSession.builder().appName("ml").master("local[*]").getOrCreate();
    	Logger.getRootLogger().setLevel(Level.ERROR);
    	
    	Dataset<Row> data = spark.read().json("C:\\Users\\Jonathan\\Desktop\\UMKC\\CS 5590"
    			+ "\\CS5590-Group-Project\\Increment 2\\Project 1a\\SourceCode\\PoliticalData.json");
    	
    	data = data.filter("tweets_per_day is not null");
    	
    	
    	VectorAssembler assembler = new VectorAssembler()
    			  .setInputCols(new String[]{"tweets_per_day", "retweet_count", "followers_count", "friends_count", "listed_count"})
    			  .setOutputCol("features");
    	
    	Dataset<Row> featurized = assembler.transform(data);
    	
    	StringIndexer labelIndexer = new StringIndexer().setInputCol("political").setOutputCol("label");
    	Dataset<Row> labeledData = labelIndexer.fit(featurized).transform(featurized);
    	
    	Dataset<Row>[] splits = labeledData.randomSplit(new double[] {0.7, 0.3});
    	Dataset<Row> trainingData = splits[0];
    	Dataset<Row> testData = splits[1];
    	
    	// Logistic Regression
    	LogisticRegressionModel lrModel = new LogisticRegression().fit(trainingData);
    	
    	Dataset<Row> logPredictions = lrModel.transform(testData);
        
    	// Decision Tree
        DecisionTreeClassificationModel dtModel = new DecisionTreeClassifier()
										          .setLabelCol("label")
										          .setFeaturesCol("features")
										          .fit(trainingData);       

        Dataset<Row> dtPredictions = dtModel.transform(testData); 
        
        // Random Forest
        RandomForestClassificationModel rfModel = new RandomForestClassifier()
											        .setLabelCol("label")
											        .setFeaturesCol("features")
											        .fit(trainingData);
        
        Dataset<Row> rfPredictions = rfModel.transform(testData);
        
        // Naive Bayes
        NaiveBayesModel nbModel = new NaiveBayes().fit(trainingData);
        
        Dataset<Row> nbPredictions = nbModel.transform(testData);
        
        // Boost
        GBTClassificationModel gbtModel = new GBTClassifier()
									        .setLabelCol("label")
									        .setFeaturesCol("features")
									        .fit(trainingData);
        
        Dataset<Row> gbtPredictions = gbtModel.transform(testData);
        
        // Test Error Calculations
        MulticlassClassificationEvaluator accEvaluator = new MulticlassClassificationEvaluator()
      		  .setLabelCol("label")
      		  .setPredictionCol("prediction")
      		  .setMetricName("accuracy");
        
        MulticlassClassificationEvaluator f1Evaluator = new MulticlassClassificationEvaluator()
        		  .setLabelCol("label")
        		  .setPredictionCol("prediction")
        		  .setMetricName("f1");
        
        MulticlassClassificationEvaluator precEvaluator = new MulticlassClassificationEvaluator()
      		  .setLabelCol("label")
      		  .setPredictionCol("prediction")
      		  .setMetricName("weightedPrecision");
        
        MulticlassClassificationEvaluator recEvaluator = new MulticlassClassificationEvaluator()
      		  .setLabelCol("label")
      		  .setPredictionCol("prediction")
      		  .setMetricName("weightedRecall");
      	
        System.out.println("Logistic Regression Model:");
        System.out.println("Test Error = " + (1.0 - accEvaluator.evaluate(logPredictions)));
        System.out.println("F1 = " + f1Evaluator.evaluate(logPredictions));
        System.out.println("Precision = " + precEvaluator.evaluate(logPredictions));
        System.out.println("Recall = " + recEvaluator.evaluate(logPredictions));
      
        System.out.println("\nDecision Tree Model:");
        System.out.println("Test Error = " + (1.0 - accEvaluator.evaluate(dtPredictions)));
        System.out.println("F1 = " + f1Evaluator.evaluate(dtPredictions));
        System.out.println("Precision = " + precEvaluator.evaluate(dtPredictions));
        System.out.println("Recall = " + recEvaluator.evaluate(dtPredictions));
        
        System.out.println("\nRandom Forest Model:");
        System.out.println("Test Error = " + (1.0 - accEvaluator.evaluate(rfPredictions)));
        System.out.println("F1 = " + f1Evaluator.evaluate(rfPredictions));
        System.out.println("Precision = " + precEvaluator.evaluate(rfPredictions));
        System.out.println("Recall = " + recEvaluator.evaluate(rfPredictions));
        
        System.out.println("\nNaive Bayes Model:");
        System.out.println("Test Error = " + (1.0 - accEvaluator.evaluate(nbPredictions)));
        System.out.println("F1 = " + f1Evaluator.evaluate(nbPredictions));
        System.out.println("Precision = " + precEvaluator.evaluate(nbPredictions));
        System.out.println("Recall = " + recEvaluator.evaluate(nbPredictions));
        
        System.out.println("\nGradient-Boosted Tree Model:");
        System.out.println("Test Error = " + (1.0 - accEvaluator.evaluate(gbtPredictions)));
        System.out.println("F1 = " + f1Evaluator.evaluate(gbtPredictions));
        System.out.println("Precision = " + precEvaluator.evaluate(gbtPredictions));
        System.out.println("Recall = " + recEvaluator.evaluate(gbtPredictions));
    }
}
