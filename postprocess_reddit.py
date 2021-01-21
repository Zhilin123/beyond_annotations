import random
import json

random.seed(42)

def load_raw_files(data_dir):
    with open(data_dir + "all_ids.json", "r") as read_file:
        all_ids = json.load(read_file)
    
    # these two files are mapping of post id to post title
    # they are used here because when downloading reddit posts initially,
    # we forgot to download the post titles
    # hence they were downloaded after
    
    with open(data_dir + "post_id_to_title.json", "r") as read_file:
         post_id_to_title = json.load(read_file)
    
    with open(data_dir + "post_id_to_title_old.json", "r") as read_file:
        post_id_to_title_old = json.load(read_file)
    return all_ids, post_id_to_title, post_id_to_title_old


def remove_duplicates(all_ids):
    all_ids_to_post = {}
    for one_post in all_ids:
        one_id = one_post['post_id']
        try:
            one_post['title'] = post_id_to_title[one_id]
        except:
            try:
                one_post['title'] = post_id_to_title_old[one_id]
            except:
                pass
        all_ids_to_post[one_id] = one_post
    new_all_ids = []
    for i in all_ids_to_post:
        new_all_ids.append(all_ids_to_post[i])
    return new_all_ids


"""
# format of each element in all_ids

one_post = {
            'post_id': post_id,
            'post_author': post_author,
            'post_text': post_text,
            'helpful_comments':helpful_comments,
            'unhelpful_comments': unhelpful_comments
        }
"""

def postprocess(all_ids):
    posts = []
    helpful_comments = []
    unhelpful_comments = []
    helpful_post_comments_tuples = []
    unhelpful_post_comments_tuples = []
    
    helpful_comments_per_post = []
    unhelpful_comments_per_post = []
    comments_per_post = []
    
    authors_with_helpful = {}
    authors_with_unhelpful = {}
    all_authors = {}
    author_to_helpful_comments = {}
    author_to_unhelpful_comments = {}
    
    all_authors = {}
    
    ratio_by_post = []
    
    sampled_authors = set()
    
    helpful_comments_one_post_one_author = []
    unhelpful_comments_one_post_one_author = []
    
    for one_post in all_ids:
        one_post_helpful = one_post['helpful_comments']
        one_post_unhelpful = one_post['unhelpful_comments']
        if 'title' in one_post:
            post_text = one_post['title'] + ' ' + one_post['post_text']
        else:
            post_text = one_post['post_text']
        posts.append(post_text)
        helpful_comments_per_post.append(len(one_post_helpful))
        unhelpful_comments_per_post.append(len(one_post_unhelpful))
        comments_per_post.append(len(one_post_helpful)+len(one_post_unhelpful))
        ratio_of_help_comment = len(one_post_helpful)/(len(one_post_helpful)+len(one_post_unhelpful))
        ratio_by_post.append(ratio_of_help_comment)
        chance = random.random()
        first_time_help = True
        first_time_unhelp = True
        for i in one_post_helpful:
            helpful_comments.append(one_post_helpful[i])
            helpful_post_comments_tuples.append((post_text,one_post_helpful[i]))
            if i not in authors_with_helpful:
                authors_with_helpful[i] = 1
                author_to_helpful_comments[i] = [one_post_helpful[i]]
            else:
                authors_with_helpful[i] += 1
                author_to_helpful_comments[i].append(one_post_helpful[i])
            try:
                all_authors[i] += 1
            except:
                all_authors[i] = 1
            if chance > 0.5 and first_time_help :
                if i not in sampled_authors:
                    helpful_comments_one_post_one_author.append(one_post_helpful[i])
                    first_time_help = False
                    sampled_authors.add(i)
        for i in one_post_unhelpful:
            unhelpful_comments.append(one_post_unhelpful[i])
            unhelpful_post_comments_tuples.append((post_text,one_post_unhelpful[i]))
            if i not in authors_with_unhelpful:
                authors_with_unhelpful[i] = 1
                author_to_unhelpful_comments[i] = [one_post_unhelpful[i]]
            else:
                authors_with_unhelpful[i] += 1
                author_to_unhelpful_comments[i].append(one_post_unhelpful[i])
            try:
                all_authors[i] += 1
            except:
                all_authors[i] = 1
            if chance < 0.5 and first_time_unhelp :
                if i not in sampled_authors:
                    unhelpful_comments_one_post_one_author.append(one_post_unhelpful[i])
                    first_time_unhelp = False
                    sampled_authors.add(i)

    helpful_comment_from_author_with_one = []
    unhelpful_comment_from_author_with_one = []
    ratio_by_author = []
    author_to_all_text = {}
    author_to_score = {}
    
    for author in all_authors:
        if all_authors[author] == 1:
            if author in author_to_helpful_comments:
                comment = author_to_helpful_comments[author][0]
                helpful_comment_from_author_with_one.append(comment)
            elif author in author_to_unhelpful_comments:
                comment = author_to_unhelpful_comments[author][0]
                unhelpful_comment_from_author_with_one.append(comment)
        if author in authors_with_helpful:
            n_helpful = authors_with_helpful[author]
        else:
            n_helpful = 0
        if all_authors[author] > 20:
            ratio_for_author = n_helpful/all_authors[author]
            author_to_score[author]= ratio_for_author
            ratio_by_author.append(ratio_for_author)
            all_text = ' '
            if author in author_to_helpful_comments:
                all_text += ' '.join(author_to_helpful_comments[author])
            if author in author_to_unhelpful_comments:
                all_text += ' '.join(author_to_unhelpful_comments[author])   
            author_to_all_text[author] = all_text
            

    return helpful_comments_one_post_one_author, unhelpful_comments_one_post_one_author, helpful_comments, unhelpful_comments, author_to_score


def save_files(data_dir):
    
    file_name = data_dir + "helpful_comments_one_post_one_author.json"
    
    with open(file_name, 'w+') as outfile:  
        json.dump(helpful_comments_one_post_one_author, outfile) 
    
    file_name = data_dir+ "unhelpful_comments_one_post_one_author.json"
    with open(file_name, 'w+') as outfile:  
        json.dump(unhelpful_comments_one_post_one_author, outfile) 
    
    file_name = data_dir+"helpful_comments.json"
    
    with open(file_name, 'w+') as outfile:  
        json.dump(helpful_comments, outfile) 
    
    file_name = data_dir+"unhelpful_comments.json"
    with open(file_name, 'w+') as outfile:  
        json.dump(unhelpful_comments, outfile) 
    
    file_name = data_dir+"authors_to_scores_20.json"  
    with open(file_name, 'w+') as outfile:  
        json.dump(author_to_score, outfile) 

if __name__ == "__main__":
    data_dir = "data/"
    all_ids, post_id_to_title, post_id_to_title_old = load_raw_files(data_dir)
    all_ids = remove_duplicates(all_ids)
    helpful_comments_one_post_one_author, \
    unhelpful_comments_one_post_one_author, \
    helpful_comments, unhelpful_comments, author_to_score = postprocess(all_ids)
    save_files(data_dir)
