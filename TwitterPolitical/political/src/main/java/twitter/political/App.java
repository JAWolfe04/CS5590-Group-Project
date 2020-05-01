package twitter.political;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;

import static org.apache.spark.sql.functions.*;

public class App 
{
    public static void main( String[] args )
    {
    	SparkSession spark = SparkSession.builder().appName("Political").master("local").getOrCreate();
    	SQLContext sqlContext = new SQLContext(spark);
    	Logger.getRootLogger().setLevel(Level.ERROR);
    	
    	sqlContext.udf().register("isPolitical", (UDF1<String, Integer>)(columnValue) -> {
    		String[] triggers = {"trump", "@realdonaldtrump", "president", "government", "administration", 
        			"obama", "@teamtrump", "biden", "govt", "@realdonaldtrump’s", "voted", "donald", "media", 
        			"@teampelosi", "journalists", "vote", "@speakerpelosi", "democrats", "senate", "federal",
        			"bipartisan", "republicans", "campaign", "maddow", "trumps", "legislation", "pres", "pelosi",
        			"democrat", "representatives", "governments", "maralago", "trump’s", "dems", "economy", "@vp",
        			"@reuters", "foreign", "parliamentary", "@whitehouse", "sen", "fauci", "fox", "@billdeblasio",
        			"@gavinnewsom", "conspiracy", "boomer", "department"};
    		if(columnValue != null && !columnValue.isEmpty()) {
	        	String testTrigger = columnValue.toLowerCase().replaceAll("[\\,\\.\\|\\(\\)\\:\\'\\?\\-\\!\\;\\#\"\\$\\d]","");
	        	for(String trigger : triggers) {
	        		if(testTrigger.contains(trigger)) {
	        			return 1;
	        		}
	        	}
    		}
        	
        	return 0;
    	}, DataTypes.IntegerType);
    	
    	Dataset<Row> data = spark.read().json("C:\\Users\\Jonathan\\Desktop\\Shared Folder\\all.txt");
    	data = data.withColumn("political",  callUDF("isPolitical", col("full_text")));
    	data = data.filter("retweet_count is not null and user.followers_count is not null and "
    			+ "user.friends_count is not null and user.listed_count is not null and id is not null"
    			+ " and created_at is not null and user.created_at is not null and user.statuses_count is not null");
    	data = data.withColumn("created_at", to_timestamp(data.col("created_at"), "EEE MMM dd HH:mm:ss '+0000' yyyy"));
    	data = data.withColumn("user_created_at", to_timestamp(data.col("user.created_at"), "EEE MMM dd HH:mm:ss '+0000' yyyy"));
    	data = data.withColumn("days_since_started", datediff(data.col("created_at"), data.col("user_created_at")));
		data = data.withColumn("tweets_per_day", data.col("user.statuses_count") .divide(data.col("Days_since_started")));
    	//data.select("political").groupBy("political").agg(count(lit(1))).show();
		data = data.select("id","political", "tweets_per_day", "retweet_count", "user.followers_count", 
				"user.friends_count", "user.listed_count");
    	data.coalesce(1).write().option("header", "true")
    	.json("C:\\Users\\Jonathan\\Desktop\\Shared Folder\\Political.json");
    }
}
