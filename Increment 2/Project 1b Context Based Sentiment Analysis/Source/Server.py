
import tweepy
from tweepy.auth import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import json

# Set up your credentials from http://apps.twitter.com
consumer_key    = 'unkekfNvz3gfqBIlrjnj30urn'
consumer_secret = 'T03zwJhdgggdpKPcEbt9M1IVOHtHrGrM9aoQzz9jsjgx1nuTUy'
access_token    = '838364671014993920-i6AWtUOoaTxfhz9DkhuB3baGbyd9tFZ'
access_secret   = 'rPTV5wQZkl1VapxLGQFM3jAIzUgeqKZJLWDbD6j5Dx2oA'

class TweetsListener(StreamListener):

    def __init__(self, csocket):
        self.client_socket = csocket

    def on_data(self, data):
        try:
            s=self.client_socket
            s.listen(5)
            c, addr = s.accept()
            print("Received request from: " + str(addr))
            msg = json.loads( data )
            user=json.loads( json.dumps(msg['user']) )
            sdata=msg['text'].replace('\n','')+' ~@ '+(user['location'] if user['location'] is not None else 'None')+' ~@ '+msg['source']
            print(sdata.encode('utf-8'))
            c.send(sdata.encode('utf-8'))
            c.close()
        
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True

def sendData(c_socket):
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    
    twitter_stream = Stream(auth, TweetsListener(c_socket))
    twitter_stream.filter(track=['Coronavirus', 'COVID-19', 'Pandemic', 'COVID19', 'covid19', 'covid-19', 'coronavirus', 'pandemic'])


if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "192.168.1.239"
    port = 5551
    s.bind((host, port))
    print("Listening on port: %s" % str(port))
    sendData(s)
