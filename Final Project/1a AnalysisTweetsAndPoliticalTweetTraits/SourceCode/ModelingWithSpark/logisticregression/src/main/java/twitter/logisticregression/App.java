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
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
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
    	
    	evaluateModel("Logistic Regression", lrModel.transform(testData));
        
    	// Decision Tree
        DecisionTreeClassificationModel dtModel = new DecisionTreeClassifier()
										          .setLabelCol("label")
										          .setFeaturesCol("features")
										          .fit(trainingData);       

        evaluateModel("Decision Tree", dtModel.transform(testData));
        
        // Random Forest
        RandomForestClassificationModel rfModel = new RandomForestClassifier()
											        .setLabelCol("label")
											        .setFeaturesCol("features")
											        .fit(trainingData);
        
        evaluateModel("Random Forest", rfModel.transform(testData));
        
        // Naive Bayes
        NaiveBayesModel nbModel = new NaiveBayes().fit(trainingData);
        
    	evaluateModel("Naive Bayes", nbModel.transform(testData));
        
        // Boost
        GBTClassificationModel gbtModel = new GBTClassifier()
									        .setLabelCol("label")
									        .setFeaturesCol("features")
									        .fit(trainingData);
        
        evaluateModel("Gradient-Boosted Tree", gbtModel.transform(testData));
    }
    
    private static void evaluateModel(String name, Dataset<Row> predication) {
    	BinaryClassificationEvaluator AUCEvaluator = new BinaryClassificationEvaluator()
        		  .setLabelCol("label")
        		  .setRawPredictionCol("prediction")
        		  .setMetricName("areaUnderROC");
    	
    	long total = predication.count();
    	long TP = predication.filter("label = 1 AND prediction = 1").count();
    	long FN = predication.filter("label = 1 AND prediction = 0").count();
    	long FP = predication.filter("label = 0 AND prediction = 1").count();
    	long TN = total - TP - FN - FP;
    	double accuracy = (double)(TP + TN) / total;
    	double precision = TP / (double)(TP + FP);
    	double sensitivity = TP / (double)(TP + FN);
    	double specificity = TN / (double)(TN + FP);
    	double f1 = 2 * ((precision * sensitivity) / (precision + sensitivity));
    	
    	System.out.println("\n" + name + " Model:");
    	
    	System.out.println("\nConfusion Matrix:");
    	System.out.println("TP = " + TP + "\tFP = " + FP);
    	System.out.println("FN = " + FN + "\tTN = " + TN);
    	
    	System.out.println("\nAccuracy = " + accuracy);
    	System.out.println("Precision = " + precision);
    	System.out.println("Sensitivity = " + sensitivity);
    	System.out.println("Specificity = " + specificity);
    	System.out.println("F1 Score = " + f1);
    	System.out.println("AUC = " + AUCEvaluator.evaluate(predication));
    }
}
