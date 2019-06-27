import tweepy
from DataHelper import get_crawled_tweets_id_list, clear_crawled_tweets_id_list, put_crawled_tweets_id_list, init_saved_tweets_file, added_to_csv_files
from DataHelper import add_log, set_api
from DataHelper import search_tweets_by_phrase
import time


def crawl_political(oauth_api, if_clear=False):
	BASE_PATH = "../develop/political"
	SAVE_FILE_NAME, LOG_FILE_NAME = init_saved_tweets_file(BASE_PATH)


	if if_clear:
		clear_crawled_tweets_id_list(BASE_PATH)

	search_since_date = None

	key_phrase_list = ['Trump', 'Biden', 'Sanders', 'Harris', 'Warren']

	for key_phrase in key_phrase_list:
		add_log(LOG_FILE_NAME, "Searching tweets with key_phrase: (%s)" % key_phrase)
		search_tweets_by_phrase(oauth_api, key_phrase, SAVE_FILE_NAME, BASE_PATH, search_since_date)

	return




if __name__ == '__main__':
    oauth_api = set_api(proxyUrl="http://127.0.0.1:7078")

    crawl_political(oauth_api, if_clear=False)


