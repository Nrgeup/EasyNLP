import tweepy
import csv
import json
import time
import codecs
import os
import time


def set_api(proxyUrl=None):
    key_list = [
        'consumer_key',
        'consumer_secret',
        'access_token',
        'access_token_secret',
    ]
    with open("./my-private-token.txt") as f:
        _index = 0
        for line in f:
            [key_name, key_value] = line.strip().split(':')
            assert key_list[_index] == key_name
            print("%s:\t%s" % (key_list[_index], key_value))
            key_list[_index] = key_value
            _index += 1

    auth = tweepy.OAuthHandler(key_list[0], key_list[1])
    auth.set_access_token(key_list[2], key_list[3])
    if proxyUrl is not None:
        api = tweepy.API(auth, 
            wait_on_rate_limit=True,
            proxy=proxyUrl
            )
    else:
        api = tweepy.API(auth,
            wait_on_rate_limit=True,
            )
    return api


def get_crawled_tweets_id_list(base_path, if_print=True):
    tweets_file = base_path + '/raw-tweets/crawled-tweets-id-list.txt'
    tweets_id_list = []
    if os.path.exists(tweets_file):
        with open(tweets_file) as fin:
            for item in fin:
                item = int(item.strip())
                tweets_id_list.append(item)
    else:
        file = open(tweets_file,'w')
        file.close()
    if if_print:
        print("Load crawled tweets id list over, num: %d" % len(tweets_id_list))
    return tweets_id_list


def get_crawled_hashtag_list(base_path, if_print=True):
    hashtag_file = base_path + '/raw-tweets/crawled-hashtag-list.txt'
    hashtag_id_list = []
    if os.path.exists(hashtag_file):
        with open(hashtag_file) as fin:
            for item in fin:
                item = str(item.strip())
                hashtag_id_list.append(item)
    else:
        file = open(hashtag_file,'w')
        file.close()
    if if_print:
        print("Load crawled tweets id list over, num: %d" % len(hashtag_id_list))
    return hashtag_id_list


def clear_crawled_tweets_id_list(base_path):
    with open(base_path + '/raw-tweets/crawled-tweets-id-list.txt', 'w') as fout:
        pass
    print("Clear crawled tweets ids and file")
    return


def put_crawled_tweets_id_list(tweets_id_list, base_path):
    with open(base_path + '/raw-tweets/tweets_id.txt', 'w') as fout:
        for item in tweets_id_list:
            fout.write(str(item)+'\n')
    print("Write crawled tweets id list over, num: %d" % len(tweets_id_list))
    return tweets_id_list


def init_saved_tweets_file(base_path):
    time_name = (time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))) 
    log_file_name = base_path + '/raw-tweets/log_%s.txt' % time_name


    save_tweets_file_name = base_path + '/raw-tweets/tweets_results_%s.csv' % time_name
    csv_file = open(save_tweets_file_name, 'w')
    csvfile = codecs.open(save_tweets_file_name, 'w', 'utf_8_sig')
    writer = csv.writer(csvfile, dialect='excel')
    writer.writerow(["tweets_id", "full_text", "create_time", "user_id", "hashtags"])
    csvfile.close()

    print("Create saved csv file: %s! " % save_tweets_file_name)
    return save_tweets_file_name, log_file_name


def init_saved_hashtag_file(base_path):
    time_name = (time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))) 
    log_file_name = base_path + '/raw-tweets/log_%s.txt' % time_name

    save_hashtag_file_name = base_path + '/raw-tweets/hashtag_results_%s.txt' % time_name

    print("Create saved hash file: %s! " % save_hashtag_file_name)
    return save_hashtag_file_name, log_file_name



def added_to_csv_files(csv_file, contents):
    with open(csv_file,"a") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows(contents)


def add_log(log_file_name, print_str):
    print(print_str)
    with open(log_file_name, 'a') as f:
        f.write("%s: %s\n" % ((time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))), print_str))
    return


def time_to_date(twitter_time):
    date = time.strftime('%Y-%m-%d', time.strptime(twitter_time,'%a %b %d %H:%M:%S +0000 %Y'))
    return date


def search_hashtags_by_target(api, query_key, save_file_name, base_path, since_id, max_id):
    crawled_hashtag = get_crawled_hashtag_list(base_path, if_print=False)
    crawled_hashtag_file_name = base_path + '/raw-tweets/crawled-hashtag-list.txt'
    num_i = 0

    last_id = max_id
    result = True

    with open(crawled_hashtag_file_name, 'a') as crawled_hashtag_file:
        while True:
            try:
                while result:
                    result = api.search(
                        q=query_key, 
                        lang="en",
                        count=100, 
                        tweet_mode='extended', 
                        since_id=since_id,
                        max_id=last_id,
                    )

                    for tweet_status in result:
                        if 'retweeted_status' in dir(tweet_status):
                            # tweet_status = tweet_status.retweeted_status
                            continue
                            
                        # For Debug
                        print("-----------------")
                        info_data = tweet_status._json
                        for item in info_data:
                            print(item)
                            print(info_data[item])
                            print("-" * 5)


                        if tweet_status.retweeted == False and tweet_status.is_quote_status == False:
                            tweets_id=tweet_status.id
                            hashtags=[str(item['text']) for item in tweet_status.entities['hashtags']]

                            for hashtag in hashtags:
                                if hashtag not in crawled_hashtag:
                                    crawled_hashtag.append(hashtag)
                                    # Save to file 
                                    crawled_hashtag_file.write(hashtag + "\n")
                            num_i += 1
                            print("Num:%d" % num_i, end="\r")
                            
                            # input("=======")
            except Exception as e:
                print(e) # , end="\r")
                continue
            break

    return current_date


def search_tweets_by_phrase(api, query_key, save_file_name, base_path, since_date):
    crawled_tweets_ids = get_crawled_tweets_id_list(base_path, if_print=False)
    crawled_tweets_id_file_name = base_path + '/raw-tweets/crawled-tweets-id-list.txt'
    
    num_i = 0

    save_csv_file = codecs.open(save_file_name, 'a', 'utf_8_sig')
    writer = csv.writer(save_csv_file, dialect='excel')
    with open(crawled_tweets_id_file_name, 'a') as crawled_tweets_id_file:
        while True:
            try:
                for tweet_status in tweepy.Cursor(
                        api.search,
                        q=query_key, 
                        lang="en",
                        tweet_mode='extended', 
                        since=since_date,
                        # max_id=last_id,
                    ).items():

                    if 'retweeted_status' in dir(tweet_status):
                        # tweet_status=tweet_status.retweeted_status
                        continue
                         
                            
                    # print("-----------------")
                    # info_data = tweet_status._json
                    # for item in info_data:
                    #     print(item)
                    #     print(info_data[item])
                    #     print("-" * 5)
                    # input("======1")

                    if tweet_status.retweeted == False and tweet_status.is_quote_status == False:
                        tweets_id=tweet_status.id
                        if tweets_id in crawled_tweets_ids:
                            continue

                        full_text=tweet_status.full_text.replace('\t', ' ').replace('\n', ' ')  # .encode('utf-8')

                        # print("full_text")
                        # print(full_text)

                        if "https://" in  full_text or "http://" in full_text:
                            continue

                        create_time=str(tweet_status.created_at)
                        user_id=tweet_status.user.id
                        hashtags=[str(item['text']) for item in tweet_status.entities['hashtags']]
                        num_i += 1
                        print("Num:%d" % num_i, end="\r")
                                

                        # For Debug
                        # print("="*10)
                        # print("tweets_id:", tweets_id)
                        # print("full_text:", full_text)
                        # print("create_time:", create_time)
                        # print("user_id:", user_id)
                        # print("hashtags:", hashtags)
                        # input("======2")
                                
                        # Save to file 
                        writer.writerow([
                            '#'+str(tweets_id),
                            full_text,
                            str(create_time),
                            '#'+str(user_id),
                            ';'.join(hashtags)
                            ])
                        crawled_tweets_id_file.write(str(tweets_id) + '\n')

            except Exception as e:
                print(e) # , end="\r")
                continue
            break
    return



