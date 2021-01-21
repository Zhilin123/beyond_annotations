#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import praw
from tqdm import tqdm
import time
import json
from psaw import PushshiftAPI

def get_all_reddit_post_ids_with_helpful(start_day=0, end_day=365, day_interval=1):
    relevant_post_ids = []
    for k in tqdm(range(start_day,end_day+1,day_interval)):
        date_before = str(k)+'d'
        date_after = str(k+day_interval) +'d'
        comment_gen = api.search_comments(author='AdviceFlairBot', limit=1000, after=date_after, before=date_before)
        comment_list = list(comment_gen)
        for i in comment_list:
            post_id = i.link_id
            relevant_post_ids.append(post_id)
    relevant_post_ids = list(set(relevant_post_ids))
    return relevant_post_ids

def remove_space(string):
    return ''.join([i for i in string if i != ' '])

def process_all_comments(comments_list):
    helpful_authors = []
    author_to_post = {}
    #print(comments_list)
    author_to_time = {}
    for j in comments_list:
        if j.author == 'AdviceFlairBot':
            award_msg_partial = "1 point awarded."
            if award_msg_partial in j.body:
                username = j.body.split()[5][3:]
                helpful_authors.append(username)
                #print(username)
        elif j.author == 'AutoModerator':
            pass
        #remove bots
        elif "Bot" in j.author or "bot" in j.author:
            pass
        else:
            author = j.author
            post = j.body
            time_posted = j.created_utc
            if author in author_to_post:
                author_to_post[author] = author_to_post[author] + " " + post
            else:
                author_to_post[author] = post
            if author in author_to_time:
                author_to_time[author] = min(int(time_posted), author_to_time[author])
            else:
                author_to_time[author] = int(time_posted)

    helpful_author_to_post = {}
    #unhelpful_author_to_post = {}
    for i in helpful_authors:
        try:
            helpful_author_to_post[i] = author_to_post[i]
            del author_to_post[i]
        except:
            pass

    return author_to_post, helpful_author_to_post, author_to_time


def get_all_reddit_post(start_day=0, end_day=200, day_interval=1, intended_subreddit="Advice", min_score= 10, author_flair_text=None, restricted_set=None):
    """
    start_day = int, >= 0
    end_day = int, > start_day
    day_interval = int, > 0
    intended_subreddit = str
    min_score = int
    """
    for k in range(start_day,end_day,day_interval):
        date_before = str(k)+'d'
        date_after = str(k+day_interval) +'d'
        if author_flair_text != None:
            gen = api.search_submissions(subreddit=intended_subreddit, score = '>'+str(min_score), limit=1000, sort_type='score', sort='desc', after = date_after, before = date_before,author_flair_text= author_flair_text)
        else:
            gen = api.search_submissions(subreddit=intended_subreddit, score = '>'+str(min_score), limit=1000, sort_type='score', sort='desc', after = date_after, before = date_before)
        results = list(gen)
        print(len(results))
        for i in tqdm(results):
            post_id = i.id
            post_title = i.title
            post_url = i.url
            post_author = i.author
            post_time = i.created_utc

            try:
                if len(i.selftext) > 0:
                    post_text = i.selftext
                else:
                    post_text = ''
            except:
                post_text = ''

            if "t3_"+post_id in restricted_set:
                post_id_to_title[post_id] = post_title


                gen_comments = api.search_comments(link_id='t3_'+post_id,limit=200)
                all_comments = list(gen_comments)
                #print(len(all_comments))

                if len(all_comments) > 0 and len(post_text) > 1:
                    unhelpful_comments, helpful_comments, author_to_time = process_all_comments(all_comments)
                    # filter out the author's own comments
                    if post_author in unhelpful_comments:
                        del unhelpful_comments[post_author]
                        del author_to_time[post_author]
                    if len(helpful_comments) == 0:
                        # remove those posts in which author did not marked any comments as helpful --> could be because the author was too laxy to mark it
                        pass
                    else:
                        one_post = {
                                'post_id': post_id,
                                'post_author': post_author,
                                'post_title': post_title,
                                'post_text': post_text,
                                'post_time': post_time,
                                'helpful_comments':helpful_comments,
                                'unhelpful_comments': unhelpful_comments,
                                'comment_author_to_time':author_to_time
                            }
                        all_ids.append(one_post)



        print(str(k), " - ", str(k+day_interval)," days done")
        print("all_ids: ",len(post_id_to_title))
        print(intended_subreddit)


if __name__ == "__main__":
    api = PushshiftAPI()
    start_day = 0
    end_day = 300
    useful_reddit_ids = get_all_reddit_post_ids_with_helpful(start_day=start_day, end_day=end_day, day_interval=1)
    all_ids = []
    post_id_to_title = {}
    get_all_reddit_post(start_day=start_day, end_day=end_day, day_interval=1, intended_subreddit="Advice", min_score=0,restricted_set=set(useful_reddit_ids))
    data_dir = "data/"
    file_name = data_dir + "all_ids.json"

    with open(file_name, 'w+') as outfile:
        json.dump(all_ids, outfile)
