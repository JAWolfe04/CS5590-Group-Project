package twitter.wordcount;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.LongWritable.DecreasingComparator;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;

import org.apache.hadoop.fs.Path;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

public class App {
	public static class Map1 extends Mapper<Object, Text, Text, IntWritable> {
		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();
		
		@SuppressWarnings("deprecation")
		@Override
		public void map(Object key, Text value, Context context)
			throws IOException, InterruptedException {
			String tweet = value.toString();
			try {
				JsonObject jsonObject = new JsonParser().parse(tweet).getAsJsonObject();
				String text = jsonObject.get("full_text").getAsString();
				text = text.toLowerCase().replaceAll("[\\,\\.\\|\\(\\)\\:\\'\\?\\-\\!\\;\\#\"\\$\\d]","");
				if (text != null && text.length() > 0){
					StringTokenizer tokenizer = new StringTokenizer(text);
					while (tokenizer.hasMoreTokens()) {
						word.set(tokenizer.nextToken());
						context.write(word, one);
					}
				}
			} 
			catch (JsonSyntaxException e) {
				Logger.getRootLogger().log(Level.ERROR, tweet);
				e.printStackTrace();
			}
			catch(IllegalStateException e) {
				e.printStackTrace();
			}
		}
	}
	
	public static class Reduce1 extends Reducer<Text, IntWritable, Text, IntWritable>{
		private IntWritable result = new IntWritable();
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context) 
			throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable value : values) 
				sum += value.get();
			result.set(sum);
            		context.write(key, result);
		}
	}
	
	public static class Map2 extends Mapper<LongWritable, Text, LongWritable, Text> {
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] wordCount = value.toString().split("\\s+");
			context.write(new LongWritable(Long.parseLong(wordCount[1])), new Text(wordCount[0]));
		}
	}
	
	public static class Reduce2 extends Reducer<LongWritable, Text, Text, Text> {
		@Override
	       protected void reduce(LongWritable key, Iterable<Text> trends, Context context) throws IOException, InterruptedException {

	           for (Text val : trends) { context.write(new Text(val.toString()), new Text(key.toString())); }
	       }
	}
	
	public static void main(String[] args) throws Exception {
		String intermediatePath = "/user/cloudera/twitterwordcount/intermediate";
		Configuration conf = new Configuration();
		Job job1 = Job.getInstance(conf, "wordcount1");
		job1.setJarByClass(App.class);
		job1.setMapperClass(Map1.class);
		job1.setReducerClass(Reduce1.class);
		FileInputFormat.addInputPath(job1, new Path("/user/cloudera/twitterwordcount/input"));
		FileOutputFormat.setOutputPath(job1, new Path(intermediatePath));
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(IntWritable.class);
		
		if(!job1.waitForCompletion(true)) {
			System.out.println("Job1 failed, exiting");
			System.exit(1);
		}
		
		Job job2 = Job.getInstance(conf, "wordcount2");
		job2.setJarByClass(App.class);
		job2.setInputFormatClass(TextInputFormat.class);
		job2.setMapperClass(Map2.class);
		job2.setReducerClass(Reduce2.class);
		job2.setSortComparatorClass(DecreasingComparator.class);
		FileInputFormat.addInputPath(job2, new Path(intermediatePath));
		FileOutputFormat.setOutputPath(job2, new Path("/user/cloudera/twitterwordcount/output"));
		job2.setOutputKeyClass(LongWritable.class);
		job2.setOutputValueClass(Text.class);
		System.exit(job2.waitForCompletion(true) ? 0 : 1);
	}
}
