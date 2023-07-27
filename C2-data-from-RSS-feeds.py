import feedparser

RSS_URL = "http://feeds.bbci.co.uk/news/rss.xml"

def process_feed(rss_url):
    feed = feedparser.parse(rss_url)
    # attributes of the feed
    print (feed['feed']['title'])
    print (feed['feed']['link'])
    print (feed.feed.subtitle)
        
    for post in feed.entries:
        print (post.link)
        print (post.title)
        # save to database
        print (post.summary)

process_feed(RSS_URL)