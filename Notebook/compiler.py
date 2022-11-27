import praw

class reddit_phy:

    # def __init__(self,
    #              client_id,
    #              client_secret,
    #              user_agent,
    #              subreddit_name):

    #     self.client_id = client_id
    #     self.client_secret = client_secret
    #     self.user_agent = user_agent
    #     self.subreddit_name = subreddit_name
    #     self.subreddit = praw.Reddit(client_id=client_id,
    #                                  client_secret=client_secret,
    #                                  user_agent=user_agent).subreddit(
    #                                      self.subreddit_name)

    def submission_getter(subreddit,
                          search=None,
                          search_sort_by='revelance',
                          no_of_submission=10):

        submissions = subreddit.search(search, sort=search_sort_by)

        sub_list = []
        count = 1
        for sub in submissions:
            sub_list.append(sub)
            if count == 3:
                break
            count += 1
        
        return sub_list
     
    def comment_getter(submission_list,
                       no_of_comments=10):
        
        sub_coms = {sub: [] for sub in submission_list}
        for sub in submission_list:
            sub_coms[sub] = sub.comments[: no_of_comments]

        return sub_coms

if __name__ == '__main__':


    client_id = 'JkwBK3M4E5CfK11a8oKkcw'
    client_secret = 'llrNKLscK7Wcu8Fg28uoNsw5-I2LMw'
    user_agent = 'psssat'

    wsb = praw.Reddit(client_id=client_id,
                      client_secret=client_secret,
                      user_agent=user_agent).subreddit(
                          'wallstreetbets')
    
    wsb_subs = reddit_phy.submission_getter(subreddit=wsb,
                                        search='Discussion')

    wsb_subs_coms = reddit.comment_getter(wsb_subs,
                                          no_of_comments=5)


    
