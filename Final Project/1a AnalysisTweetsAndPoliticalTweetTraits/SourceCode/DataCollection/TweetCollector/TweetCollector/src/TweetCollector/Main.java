package TweetCollector;

import twitter4j.*;
import twitter4j.Query.ResultType;

import java.time.LocalTime;

import TweetCollector.TweetFile;

public class Main {
	static final Twitter twitter = new TwitterFactory().getInstance();
    static final String hashTag="#coronavirus";
    private static StringBuilder tweets = new StringBuilder("");
    private static int tweetCount = 0;
    static final int count = 100;
    static long sinceId = 0;
    static long numberOfTweets = 0;

	public static void main(String[] args) throws InterruptedException {
		Query queryMax = new Query(hashTag);
		queryMax.setLang("en");
		queryMax.setResultType(ResultType.mixed);
        queryMax.setCount(count);
        getTweets(queryMax, "maxId");
        queryMax = null;
        
        do{
            Query querySince = new Query(hashTag);
            querySince.setLang("en");
            querySince.setResultType(ResultType.mixed);
            querySince.setCount(count);
            querySince.setSinceId(sinceId);
            getTweets(querySince, "sinceId");
            querySince = null;
        }while(checkIfSinceTweetsAreAvaliable());
	}
	
	private static boolean checkIfSinceTweetsAreAvaliable() throws InterruptedException {
        Query query = new Query(hashTag);
        query.setLang("en");
        query.setResultType(ResultType.mixed);
        query.setCount(count);
        query.setSinceId(sinceId);
        try {
            QueryResult result = twitter.search(query);
            if(result.getTweets()==null || result.getTweets().isEmpty()){
                query = null;
                return false;
            }
        } catch (TwitterException te) {
        	handleTwitterException(te);
        }catch (Exception e) {
            System.out.println("Something went wrong: " + e);
            Thread.sleep(1000 * 60 * 2);
        }
        return true;
    }
	
	private static void getTweets(Query query, String mode) throws InterruptedException {
        boolean getTweets=true;
        long maxId = 0;
        long whileCount=0;
 
        while (getTweets){
            try {
                QueryResult result = twitter.search(query);
                if(result.getTweets()==null || result.getTweets().isEmpty()){
                    getTweets=false;
                }else{
                    System.out.println("Gathered " + result.getTweets().size() + " tweets. Total tweets: " + numberOfTweets);
                    int forCount=0;
                    for (Status status: result.getTweets()) {
                        if(whileCount == 0 && forCount == 0){
                            sinceId = status.getId();//Store sinceId in database
                            System.out.println("sinceId= "+sinceId);
                        }
                        tweets.append(TwitterObjectFactory.getRawJSON(status).toString() + "\n");
                        ++tweetCount;
                        if(tweetCount == 1000) 
                        	saveTweets();
                        if(forCount == result.getTweets().size()-1){
                            maxId = status.getId();
                        }
                        forCount++;
                    }
                    numberOfTweets=numberOfTweets+result.getTweets().size();
                    query.setMaxId(maxId-1); 
                }
            }catch (TwitterException te) {
            	handleTwitterException(te);
            }catch (Exception e) {
                System.out.println("Something went wrong: " + e);
                Thread.sleep(1000 * 60 * 2);
            }
            whileCount++;
        }
        System.out.println("Total tweets count======="+numberOfTweets);
        saveTweets();
    }
	
	public static void saveTweets() {
		TweetFile.saveTweets(tweets.toString());
		tweets.setLength(0);
		tweetCount = 0;
	}
	
	public static void handleTwitterException(TwitterException te) throws InterruptedException {    	
        if(te.exceededRateLimitation()) {
        	System.out.println(LocalTime.now() + ": Exceeded Limit, waiting 15 minutes");
        	Thread.sleep(1000 * 60 * 15);
        } else {
        	System.out.println("Couldn't connect: " + te);
        	Thread.sleep(1000 * 60 * 2);
        }
	}
	
}
