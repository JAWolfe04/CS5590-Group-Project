package TweetCollector;

import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;

//--------------------------------------------------------------------------
// TweetFile Class
//
// This class saves tweets to a folder named "Input". Each file is less than
// 64MB in size. This class assumes there is a folder called "Input" in the
// project directory.
//--------------------------------------------------------------------------
public class TweetFile {
	private static int fileCount = 1;
	private static File tweetFile = new File("Input/TweetFile" + fileCount + ".txt");
	
	// Constructor creates the initial file to save the tweets
	TweetFile() { createFile(); }
	
	//--------------------------------------------------------------------------
	// createFile
	//
	// This function creates a new file if it does not already exist. The file
	// name is set in "tweetFile" prior to calling this function
	//--------------------------------------------------------------------------
	private static void createFile() {
		try {
			// Attempt to create a new file, otherwise warn that it already exists.
			// This class only appends to files, it does not override
			if (tweetFile.createNewFile()) {
		        System.out.println("File created: " + tweetFile.getName());
		      } 
			else {
		        System.out.println("File already exists.");
		      }
			} 
		catch (IOException e) {
	      System.out.println("An error occurred.");
	      e.printStackTrace();
	    }
	}
	
	//--------------------------------------------------------------------------
	// saveTweets
	//
	// tweets - String of 1000 tweets with each tweet JSON on a different line
	//
	// This function saves the passed tweets to the file designated by tweetFile
	// This function appends tweets to the end of the file.
	//--------------------------------------------------------------------------
	public static void saveTweets(String tweets) {
		try {
		      FileWriter fileWriter = new FileWriter(tweetFile, true);
		      BufferedWriter bufferWriter = new BufferedWriter(fileWriter);
		      bufferWriter.write(tweets);
		      bufferWriter.close();
		      
		      // Separate tweet files into files less than 64MB
		      if(tweetFile.length() >= 61000000) {
		    	  // Create a new file when the size of the file in the next
		    	  // attempt to save may exceed 64MB. Files containing retweets tend
		    	  // to have a size of 5.5MB per save and without tend around 3.3MB
		    	  ++fileCount;
		    	  tweetFile = new File("Input/TweetFile" + fileCount + ".txt");
		    	  createFile();
		      }
		    } 
		catch (IOException e) {
		      System.out.println("An error occurred.");
		      e.printStackTrace();
		    }
	}
}
