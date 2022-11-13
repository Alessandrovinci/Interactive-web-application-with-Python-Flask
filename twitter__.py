from flask import Blueprint

twitter_app = Blueprint('twitter_app', __name__)

####################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections
import string
import tweepy as tw
import nltk
from nltk.corpus import stopwords
from nltk import bigrams
import re
from textblob import TextBlob
import numpy as np
import networkx as nx
from COMPANY_watson import ToneAnalyzerV3
from COMPANY_cloud_sdk_core.authenticators import IAMAuthenticator
import pickle
import time
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords    
from nltk.stem import WordNetLemmatizer 
from gensim.corpora import Dictionary
from gensim.models import LdaModel
# 1. Wordcloud of Top N words in each topic
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from IPython.display import display, HTML
from flask import Flask, g, render_template,request,redirect
from wordcloud import  WordCloud, STOPWORDS, ImageColorGenerator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime as dt
from COMPANY_watson import ToneAnalyzerV3
from COMPANY_cloud_sdk_core.authenticators import IAMAuthenticator
###################################################

# HELPERS #
def tokenizer(text):
    tokens=[nltk.word_tokenize(passage.lower()) for passage in text ]
    return tokens

def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

def punct(punctuations,tokens):
    b_filter=[]
    for passage in tokens:
        new_sentence = [word for word in passage if not word in punctuations]
        b_filter.append(new_sentence)
    return b_filter

stop_words = stopwords.words('english')
def stopwords(stop_words,tokens):
    a_filter=[]
    for passage in tokens:
        new_sentence = [word for word in passage if not word in stop_words]
        a_filter.append(new_sentence)
    return a_filter

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    c_filter=[]
    for passage in tokens:
        new_sentence = [lemmatizer.lemmatize(word) for word in passage]
        c_filter.append(new_sentence)
    return c_filter


#function to compute the word frequencies in the text
def frequencies(tokens):
    d={}
    for passage in tokens:
        for word in passage:
            if word in d:
                d[word]+=1
            else:
                d[word]=1
    sorted_d=dict(sorted(d.items(), key=lambda item: item[1],reverse=True))
    return sorted_d

    
def aYN2(row):
    if row>-0.005 and row<=0.05:
        return 'Neutral'
    elif row>0.05:
        return 'Positive'
    else:
        return 'Negative'



##########################################################################################

#START
    
russels=["University of Birmingham",
"University of Bristol",
"University of Cambridge",
"Cardiff University",
"Durham University",
"University of Edinburgh",
"University of Exeter",
"University of Glasgow",
"Imperial College London",
"King's College London",
"University of Leeds",
"University of Liverpool",
"London School of Economics and Political Science",
"University of Manchester",
"Newcastle University",
"University of Nottingham",
"University of Oxford",
"Queen Mary University of London",
"Queen's University Belfast",
"University of Sheffield",
"University of Southampton",
"University College London",
"University of Warwick",
"University of York"
]




@twitter_app.route('/twitter_search.html',methods=["POST", "GET"])
def twitter_search():
    title='Filter'
    if request.method == "GET":
        
        return render_template('twitter_search.html')


@twitter_app.route('/update_database.html',methods=["POST", "GET"])
def update_database():
    title='Update'
    if request.method == "POST":
        yes=request.form.get('Yes')
        no=request.form.get('No')
        if yes=='yes':
            flag,new_all_full_tweet_object=scraping()
            if flag==0:

                with open('path/all_full_tweet_object_weekly.pickle', 'wb') as handle:
                    pickle.dump(new_all_full_tweet_object, handle, protocol=pickle.HIGHEST_PROTOCOL)
                for uni_name in new_all_full_tweet_object.keys():
                    uni_idx=russels.index(uni_name)
                    create_graphs(uni_name,uni_idx,new_all_full_tweet_object)

                return render_template('loading.html')
            else:
                return render_template("alert_input_scrape.html")
            
        else:
            return render_template('twitter_search.html')
    last_update=str(np.max([c.created_at for c in all_full_tweet_object_old['University of Cambridge']]))[0:10]
    return render_template('update_database.html',last_update=last_update)


@twitter_app.route('/tw_form.html', methods=["POST"])
def tw_form():
    title="Results"
    uni_name=request.form.get("uni_name")
    uni_idx=russels.index(uni_name)
    #IF you want to create new graphs ONLY for ONE university uncomment #create_graphs below and simply search for the university on the portal
    #create_graphs(uni_name,uni_idx)
    url_img=f'/static/images/cloud{uni_idx}.png'
    url_img2=f'/static/images/pol{uni_idx}.png'
    #url_img3=f'/static/images/watson1_{uni_idx}__2.png'
    url_img4=f'/static/images/watson2_{uni_idx}.png'
    url_imgTime=f'/static/images/overtime{uni_idx}.png'
    url_imgTopW=f'/static/images/topwords{uni_idx}.png'
    url_imgComp=f'/static/images/compound{uni_idx}.png'
    url_imgComp_weekly=f'/static/images/compound{uni_idx}_weekly.png'
    url_imgTopent=f'/static/images/topent{uni_idx}.png'
    url_imgTopbig=f'/static/images/topbig{uni_idx}.png'

    bb=[tweet.retweet_count for tweet in all_full_tweet_object_old[uni_name]]
    bb=sorted(range(len(bb)), key=lambda k: bb[k],reverse=True)[0:5]
    most_retweeted=np.array([tweet.full_text for tweet in all_full_tweet_object_old[uni_name]])[bb]
    most_retweeted=[remove_url(t) for t in most_retweeted]


    tokenized=all_tokens_old[uni_name]

    terms_bigram = [list(bigrams(tweet)) for tweet in tokenized]

    # Flatten list of bigrams in clean tweets
    bigrams_l = list(itertools.chain(*terms_bigram))

    # Create counter of words in clean bigrams
    bigram_counts = collections.Counter(bigrams_l)


    bigram_top_20=bigram_counts.most_common(10)
    bigrams_w=[w[0] for w in bigram_top_20]
    bigrams_c=[c[1] for c in bigram_top_20]


    return render_template("tw_form.html",title=title,url_img=url_img,uni_name=uni_name,url_img4=url_img4,
        retweets=most_retweeted,bigrams_w=bigrams_w,bigrams_c=bigrams_c,url_img2=url_img2,
        url_imgTime=url_imgTime,url_imgTopW=url_imgTopW,url_imgComp=url_imgComp,url_imgComp_weekly=url_imgComp_weekly,
        url_imgTopent=url_imgTopent,url_imgTopbig=url_imgTopbig)


@twitter_app.route('/update_databaseCOMPANY.html',methods=["POST", "GET"])
def update_databaseCOMPANY():
    title='Update'
    if request.method == "POST":
        yes=request.form.get('Yes')
        no=request.form.get('No')
        if yes=='yes':
            flag,new_all_tokens,new_all_tweets,new_all_full_tweet_object=scrapingCOMPANY()
            if flag==0:
                for uni_name in new_all_full_tweet_object.keys():
                    uni_idx=russels.index(uni_name)
                    create_graphsCOMPANY(uni_name,uni_idx)

                return render_template('loading.html')
            else:
                return render_template("alert_input_scrape.html")
        else:
            return render_template('twitter_search_COMPANY.html')
    last_update_list=[]
    for russ in russels:
        try:
            last_update=np.max([c.created_at for c in all_full_tweet_object_COMPANY_old[russ]])
            last_update_list.append(last_update)
        except:
            continue
        
    last_update=str(np.max(last_update_list))[0:10]
    return render_template('update_databaseCOMPANY.html',last_update=last_update)


@twitter_app.route('/twitter_search_COMPANY.html',methods=["POST", "GET"])
def twitter_search_COMPANY():
    title='Filter'
    if request.method == "GET":
        
        return render_template('twitter_search_COMPANY.html')

@twitter_app.route('/tw_COMPANY_form.html', methods=["POST"])
def tw_COMPANY_form():
    title="Results"
    uni_name=request.form.get("uni_name")
    uni_idx=russels.index(uni_name)
    #IF you want to create new graphs ONLY for ONE university uncomment #create_graphs below and simply search for the university on the portal
    #create_graphs(uni_name,uni_idx)
    url_img=f'/static/imagesCOMPANY/cloud{uni_idx}.png'
    url_img2=f'/static/imagesCOMPANY/pol{uni_idx}.png'
    #url_img3=f'/static/images/watson1_{uni_idx}__2.png'
    #url_img4=f'/static/images/watson2_{uni_idx}.png'
    url_imgTime=f'/static/imagesCOMPANY/overtime{uni_idx}.png'
    url_imgTopW=f'/static/imagesCOMPANY/topwords{uni_idx}.png'
    url_imgComp=f'/static/imagesCOMPANY/compound{uni_idx}.png'
    url_imgTopent=f'/static/imagesCOMPANY/topent{uni_idx}.png'
    url_imgTopbig=f'/static/imagesCOMPANY/topbig{uni_idx}.png'


    bb=[tweet.retweet_count for tweet in all_full_tweet_object_COMPANY_old[uni_name]]
    bb=sorted(range(len(bb)), key=lambda k: bb[k],reverse=True)[0:5]
    most_retweeted=np.array([tweet.full_text for tweet in all_full_tweet_object_COMPANY_old[uni_name]])[bb]
    most_retweeted=[remove_url(t) for t in most_retweeted]

    tokenized=all_tokens_COMPANY_old[uni_name]

    terms_bigram = [list(bigrams(tweet)) for tweet in tokenized]

    # Flatten list of bigrams in clean tweets
    bigrams_l = list(itertools.chain(*terms_bigram))

    # Create counter of words in clean bigrams
    bigram_counts = collections.Counter(bigrams_l)


    bigram_top_20=bigram_counts.most_common(10)
    bigrams_w=[w[0] for w in bigram_top_20]
    bigrams_c=[c[1] for c in bigram_top_20]


    return render_template("tw_COMPANY_form.html",title=title,url_img=url_img,uni_name=uni_name,
        retweets=most_retweeted,bigrams_w=bigrams_w,bigrams_c=bigrams_c,url_img2=url_img2,
        url_imgTime=url_imgTime,url_imgTopW=url_imgTopW,url_imgComp=url_imgComp,
        url_imgTopent=url_imgTopent,url_imgTopbig=url_imgTopbig)

#FOR CAMBRIDGE
#It's possible to visualize an example of the results obtainable with more data.
#In the specific, there are two pickled dictionaries containing tweets about COMPANY and University of Cambridge over time.
#To display the results you need to set tw_COMPANY_form2 instead tw_COMPANY_form.
#Go to line 25 of the template twitter_search_COMPANY.html
#replace <form action="/tw_COMPANY_form.html" method="POST"> with <form action="/tw_COMPANY_form2.html" method="POST">
#save and refresh the web app. Then go on COMPANY on twitter and search for Cambridge.
with open('C:/COMPANY/static/pickles/Camtok_COMPANY.pickle', 'rb') as handle:
    C_COMPANY_tok = pickle.load(handle)

with open('C:/COMPANY/static/pickles/CamTrue_COMPANY.pickle', 'rb') as handle:
    ents1 = pickle.load(handle)

@twitter_app.route('/tw_COMPANY_form2.html', methods=["POST"])
def tw_COMPANY_form2():
    title="Results"
    uni_name=request.form.get("uni_name")
    uni_idx=russels.index(uni_name)
    url_img=f'/static/imagesCOMPANY/Cambridge/cloud{uni_idx}.png'
    url_img2=f'/static/imagesCOMPANY/Cambridge/pol{uni_idx}.png'
    url_img4=f'/static/imagesCOMPANY/Cambridge/watson2_{uni_idx}.png'
    url_imgTime=f'/static/imagesCOMPANY/Cambridge/Time{uni_idx}.png'
    url_imgTopW=f'/static/imagesCOMPANY/Cambridge/topwords{uni_idx}.png'
    url_imgComp=f'/static/imagesCOMPANY/Cambridge/Comp{uni_idx}.png'
    url_imgTopent=f'/static/imagesCOMPANY/Cambridge/topentA{uni_idx}.png'
    url_imgTopbig=f'/static/imagesCOMPANY/Cambridge/topbig{uni_idx}.png'


    ALLwant=[]
    for item in ents1:
        want={}
        want['created_at']=item['created_at']
        if 'extended_tweet' in item.keys():
            tweet_text=item['extended_tweet']['full_text']
            ent_list=[]
            for ent in item['extended_tweet']['entities']['user_mentions']:
                ent_list.append(ent['name'])
        else:
            tweet_text=item['text']
            ent_list=[]
            for ent in item['entities']['user_mentions']:
                ent_list.append(ent['name'])

        want['text']=tweet_text
        #want['location']='location'
        want['entities']=ent_list
        want['retweet_count']=item['retweet_count']
        want['favorite_count']=item['favorite_count']
        ALLwant.append(want)

    adf=pd.DataFrame(ALLwant)


    most_retweeted=adf.sort_values(by=['retweet_count'],ascending=False).text.iloc[0:5].tolist()
    most_favorite=adf.sort_values(by=['favorite_count'],ascending=False).text.iloc[0:5].tolist()


    topent={}
    for item in ALLwant:
        if item['entities']!=[]:
            for name in item['entities']:
                if name in topent:
                    topent[name]+=1
                else:
                    topent[name]=1
    topent={k: v for k, v in sorted(topent.items(), key=lambda item: item[1],reverse=True)}
    top_20=list(topent.keys())[0:15]
    scores_ent=list(topent.values())[0:15]


    tokenized=C_COMPANY_tok

    terms_bigram = [list(bigrams(tweet)) for tweet in tokenized]

    # Flatten list of bigrams in clean tweets
    bigrams_l = list(itertools.chain(*terms_bigram))

    # Create counter of words in clean bigrams
    bigram_counts = collections.Counter(bigrams_l)


    bigram_top_20=bigram_counts.most_common(20)
    bigrams_w=[w[0] for w in bigram_top_20]
    bigrams_c=[c[1] for c in bigram_top_20]


    return render_template("tw_COMPANY_form2.html",title=title,url_img=url_img,uni_name=uni_name,top_20=top_20,scores_ent=scores_ent,
        retweets=most_retweeted,favorite=most_favorite,bigrams_w=bigrams_w,bigrams_c=bigrams_c,
        url_img2=url_img2,url_img4=url_img4,url_imgTime=url_imgTime,
        url_imgTopW=url_imgTopW,url_imgComp=url_imgComp,url_imgTopent=url_imgTopent,url_imgTopbig=url_imgTopbig)

def scraping():

    #####################################

    consumer_key= 'bPoE4mUtF4NkVYvmxaq86OlQ8'
    consumer_secret= 'Y92epzGZRRdOPBSxp1TjoZU5WmBy1AUaxabHaLbZnGBsFNpxz5'
    access_token= '1414537754151686146-OT8mryRlddBcYHgoNn1hUmHubXx10d'
    access_token_secret= 'vRMd4K0hZTpel2NCJJnmhlvXLXCe95mX0wUux4OpSvhNt'

    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api_tw = tw.API(auth, wait_on_rate_limit=True)

    #punctuations = list(string.punctuation)
    punctuations=[]
    punctuations.append('’')
    punctuations.append("'s")
    punctuations.append('“')
    punctuations.append('”')
    punctuations.append('click')
    punctuations.append('bio')
    punctuations.append('link')
    punctuations.append('u')
    punctuations.append('join')
    punctuations.append('help')


    extras_at=["@unibirmingham","@BristolUni","@Cambridge_Uni","@cardiffuni","@durham_uni","@EdinburghUni","@UniofExeter",
            "@UofGlasgow","@imperialcollege","@KingsCollegeLon","@UniversityLeeds","@LivUni","@LSEnews","@OfficialUoM",
            "@UniofNewcastle","@UniofNottingham","@UniofOxford","@QMUL","@QUBelfast","@sheffielduni","@unisouthampton",
            "@ucl","@warwickuni","@UniOfYork"]
    extras_no_at=["unibirmingham","bristoluni","cambridgeuni","cardiffuni","durham_uni","edinburghuni","uniofexeter",
            "uofglasgow","imperialcollege","kingscollegelon","universityleeds","livuni", "lsenews","officialuom","uniofnewcastle",
            "uniofnottingham","uniofoxford","qmul","qubelfast","sheffielduni","unisouthampton","university ucl","warwickuni","uniofyork"]

    names_removal=["birmingham","bristol","cambridge","cardiff","durham","edinburgh","exeter","glasgow","london","london","leeds","liverpool",
            "london","manchseter","newcastle","nottingham","oxford","london","belfast","sheffield","southampton","london","warwick","york"]

    ########## SENTIMENT ############
    from COMPANY_watson import ToneAnalyzerV3
    from COMPANY_cloud_sdk_core.authenticators import IAMAuthenticator
    #################################


    try:

        all_tokens={}
        all_tweets={}
        all_full_tweet_object={}
        for russ in russels:
            rus_i=russels.index(russ)
            this_search_tweets=[]
            
            search_term = f"{russ} -filter:retweets"
            search_tweets = api_tw.search(search_term,count=100,tweet_mode='extended')
            this_search_tweets+=search_tweets
            #all_tweets1 = [tweet.full_text for tweet in search_tweets]#if not tweet.lang!='en']
            for j in range(8):
                try:
                    idx=np.argmin([tweet.id for tweet in search_tweets])
                    lowest_id=[tweet.id for tweet in search_tweets][idx]
                    search_tweets = api_tw.search(search_term,count=100,tweet_mode='extended',max_id=lowest_id)
                    this_search_tweets+=search_tweets
                    #all_tweets1 +=  [tweet.full_text for tweet in search_tweets]# if not tweet.lang!='en']
                except:
                    break

            search_term = f"{extras_at[rus_i]} -filter:retweets"
            search_tweets = api_tw.search(search_term,count=100,tweet_mode='extended')
            this_search_tweets+=search_tweets
            #all_tweets2 = [tweet.full_text for tweet in search_tweets]# if not tweet.lang!='en']
            for j in range(8):
                try:
                    idx=np.argmin([tweet.id for tweet in search_tweets])
                    lowest_id=[tweet.id for tweet in search_tweets][idx]
                    search_tweets = api_tw.search(search_term,count=100,tweet_mode='extended',max_id=lowest_id)
                    this_search_tweets+=search_tweets
                    #all_tweets2 +=  [tweet.full_text for tweet in search_tweets]# if not tweet.lang!='en']
                except:
                    break


            #this_entities={k: v for k, v in sorted(this_entities.items(), key=lambda item: item[1],reverse=True)}
            #all_entities[russ]=this_entities
            #tweets_no_urls1 = [remove_url(tweet) for tweet in all_tweets1]
            #tweets_no_urls2 = [remove_url(tweet) for tweet in all_tweets2]

            #tweets_no_urls = tweets_no_urls1+tweets_no_urls2
            tweets_no_urls =[tw.full_text for tw in this_search_tweets if not tw.lang!='en']
            tweets_no_urls =[remove_url(tweet) for tweet in tweets_no_urls]
            tweets_no_urls = list(set(tweets_no_urls))
            try:
                tweets_no_urls.remove('')
            except:
                pass
            #store
            all_tweets[russ]=tweets_no_urls
            all_full_tweet_object[russ]=this_search_tweets
            

            tokenized=tokenizer(tweets_no_urls)
            tokenized=stopwords(stop_words,tokenized)
            tokenized=punct(punctuations,tokenized)
            to_remove=[russ,extras_at[rus_i],extras_no_at[rus_i],'university','University','Congratulations','congratulations','done']
            tokenized=punct(to_remove,tokenized)
            tokenized=punct(names_removal,tokenized)
            all_tokens[russ]=tokenized


        for university in all_tweets.keys():
            all_tweets_old[university]+=all_tweets[university]
            all_tweets_old[university]=list(set(all_tweets_old[university]))

            all_tokens_old[university]+=all_tokens[university]
            all_full_tweet_object_old[university]+=all_full_tweet_object[university]

            merged_tokens=[]
            for token in all_tokens_old[university]:
                a=' '.join(token)
                merged_tokens.append(a)

            merged_tokens=list(set(merged_tokens))
            tokens_no_duplicates=[]
            for no_dup in merged_tokens:
                a=no_dup.split(' ')
                tokens_no_duplicates.append(a)

            all_tokens_old[university]=tokens_no_duplicates

            all_full_tweet_object_old[university]+=all_full_tweet_object[university]

            uniques=[]
            idces=[]
            for i,obj in enumerate(all_full_tweet_object_old[university]):
                if obj.full_text in uniques:
                    continue
                else:
                    uniques.append(obj.full_text)
                    idces.append(i)
            objects_no_duplicates=[]
            for ids in idces:
                objects_no_duplicates.append(all_full_tweet_object_old[university][ids])

            all_full_tweet_object_old[university]=objects_no_duplicates

        with open('C:/COMPANY/static/pickles/tempt/all_tokens.pickle', 'wb') as handle:
                pickle.dump(all_tokens_old, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('C:/COMPANY/static/pickles/tempt/all_tweets.pickle', 'wb') as handle:
            pickle.dump(all_tweets_old, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('C:/COMPANY/static/pickles/tempt/all_full_tweet_object.pickle', 'wb') as handle:
            pickle.dump(all_full_tweet_object_old, handle, protocol=pickle.HIGHEST_PROTOCOL)

        flag=0
        return flag,all_full_tweet_object
    except:
        flag=1
        return flag,None
    

        
    ######### END OF SCRAPING FUNC #####################


#FIX DIFFERENCE SEARCH TWEETS AND OTHERS


def create_graphs(russ,uni_idx,new_all_full_tweet_object):
            #START
    ###############
    #GRAPH ALL_ENTITIES
    '''
    Declare 
    this_entities dict
    tokenized list
    tweets_no_urls list
    retweets array list 
    this_search_tweets list of full tweet object (for the date)
    '''

    api_s='j3iyvaYX6wE5LYltz01xuGk0sCYV8Eti_gzHKudwDc8o'
    url_s='https://api.eu-gb.tone-analyzer.watson.cloud.COMPANY.com/instances/771d6bc4-dda8-4d65-a283-1c2b654604d7'

    
    #this_entities=all_entities[russ]
    tokenized=all_tokens_old[russ]
    tweets_no_urls=all_tweets_old[russ]
    this_search_tweets= all_full_tweet_object_old[russ]


    this_entities={}
    for tweet in this_search_tweets:
        aaa=tweet.entities['user_mentions']
        for a in aaa:
            if a['name'] in this_entities:
                this_entities[a['name']]+=1
            else:
                this_entities[a['name']]=1

    this_entities={k: v for k, v in sorted(this_entities.items(), key=lambda item: item[1],reverse=True)}
    entities_list=list(this_entities.keys())
    matchers = ['Uni', 'dr','Dr','University','university','College','college','UC','UK','England','Prof','Scho','scho','Journal',
            'Academ','academic','London','Institute','institute','Sport','Alliance','COMPANY','COMPANY','PhD','MSc','Msc',
            'Bsc','BSc','Museum','@','British','Knowledge','Foundation','Centre','Service','Research','Group','Institution',
            'Associa','Gallery','Art','Politic','Stud','Council','Community','Carreer','Faculty','Department','dept','Alumni','News']
    top_25=[s for s in entities_list if any(xs in s for xs in matchers)][0:20]
    scores_ent=[this_entities[a] for a in top_25]

    tops={}
    for a in top_25:
        tops[a]=this_entities[a]
    tops=pd.Series(tops)
    plt.figure(figsize=(10,5))
    sns.barplot( tops.values, tops.index,alpha=0.8)
    plt.title('Most mentioned entities',fontsize=12)
    plt.xlabel('Count of the citations', fontsize=12)
    #plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'C:/COMPANY/static/images/topent{uni_idx}.png')

    #################
    #TOPIC MODELLING
    dictionary = Dictionary(tokenized)
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    #dictionary.filter_extremes(no_below=10, no_above=0.8)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    # Set training parameters.
    num_topics = 50
    chunksize = 2000
    passes = 20
    iterations = 500
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha= 0.00056,
        eta=0.1,
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )


    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')
        plt.savefig(f'C:/COMPANY/static/images/cloud{uni_idx}.png')
    ####################
    #GRAPH MOST USED WORDS
    #concatenate tokens in one sentence 
    tt=[]
    tt2=[]
    for t in tokenized:
        tt2+=t
        a=' '.join(t)
        tt.append(a)

    topwords=pd.DataFrame(tt2)
    topwords = topwords[0].value_counts()

    topwords = topwords[0:25]
    plt.figure(figsize=(10,5))
    sns.barplot(topwords.values, topwords.index, alpha=0.8)
    plt.title('Top Words Overall')
    plt.ylabel('Word from Tweet', fontsize=12)
    plt.xlabel('Count of Words', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'C:/COMPANY/static/images/topwords{uni_idx}.png')
    ####################
    #GRAPH MOST USED BIGRAMS
    terms_bigram=[list(bigrams(doc)) for doc in tokenized]

    bigrams_l=list(itertools.chain(*terms_bigram))
    bigram_counts=collections.Counter(bigrams_l)
    bigram_top_20=bigram_counts.most_common(10)
    bigrams_w=[w[0] for w in bigram_top_20]
    bigrams_c=[c[1] for c in bigram_top_20]
    ind=[f"{v[0]}_{v[1]}" for v in bigrams_w]
    vals=[v for v in bigrams_c]
    tops=pd.Series(ind,vals)
    plt.figure(figsize=(10,5))
    sns.barplot(tops.values,tops.index,alpha=0.8)
    plt.title("Most frequent bigrams", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(f'C:/COMPANY/static/images/topbig{uni_idx}.png')

    ####################
    #SENTIMENT ANALYSIS

    analyzer = SentimentIntensityAnalyzer()
    atweet_list = (t.__dict__ for t in this_search_tweets)
    adf = pd.DataFrame(atweet_list)
    #adf['lang'] = adf['full_text'].map(lambda x: detect(x))
    #adf = adf[adf['lang']=='en']
    asentiment = adf['full_text'].apply(lambda x: analyzer.polarity_scores(x))
    adf = pd.concat([adf,asentiment.apply(pd.Series)],1)

    ####################
    #NEG/POS/NEU PLOT  
    adf['YN']=adf['compound'].apply(aYN2)
    fig,ax=plt.subplots(figsize=(6,4.5))
    ax = sns.countplot(x="YN", data=adf)
    ax.set(xlabel='Mood around the Tweet', ylabel='Number of Tweets')
    plt.savefig(f'C:/COMPANY/static/images/pol{uni_idx}.png')
    ####################

    #SENTIMENT OVER TIME
    adf.sort_values(by='created_at', inplace=True)
    adf.index = pd.to_datetime(adf['created_at'])
    adf['mean'] = adf['compound'].expanding().mean()
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.scatter(adf['created_at'],adf['compound'], label='Tweet Sentiment')
    ax.plot(adf['created_at'],adf['mean'], color='y', label='Expanding Mean')
    ax.set(title=f'{russ} Tweets over Time', xlabel='Date', ylabel='Sentiment')
    ax.legend(loc='best')
    ax.xaxis.set_major_locator(plt.MaxNLocator(9))
    plt.xticks(rotation=30)
    fig.tight_layout()
    plt.savefig(f'C:/COMPANY/static/images/compound{uni_idx}.png')


    this_search_tweets_weekly= new_all_full_tweet_object[russ]
    analyzer2 = SentimentIntensityAnalyzer()
    atweet_list2 = (t.__dict__ for t in this_search_tweets_weekly)
    adf2 = pd.DataFrame(atweet_list2)
    #adf['lang'] = adf['full_text'].map(lambda x: detect(x))
    #adf = adf[adf['lang']=='en']
    asentiment2 = adf2['full_text'].apply(lambda x: analyzer2.polarity_scores(x))
    adf2 = pd.concat([adf2,asentiment2.apply(pd.Series)],1)
    adf2.sort_values(by='created_at', inplace=True)
    adf2.index = pd.to_datetime(adf2['created_at'])
    adf2['mean'] = adf2['compound'].expanding().mean()
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.scatter(adf2['created_at'],adf2['compound'], label='Tweet Sentiment')
    ax.plot(adf2['created_at'],adf2['mean'], color='y', label='Expanding Mean')
    ax.set(title=f'{russ} Tweets over this week', xlabel='Date', ylabel='Sentiment')
    ax.legend(loc='best')
    ax.xaxis.set_major_locator(plt.MaxNLocator(9))
    plt.xticks(rotation=30)
    fig.tight_layout()
    plt.savefig(f'C:/COMPANY/static/images/compound{uni_idx}_weekly.png')
    ####################

    #COMPOUND DISTRIBUTION
    #adf=adf[adf['compound']!=0]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    sns.distplot(adf['compound'], bins=15, ax=ax)
    ax.set(title=f'{russ} Sentiment Scores', xlabel='Compound sentiment', ylabel='Density')
    plt.savefig(f'C:/COMPANY/static/images/overtime{uni_idx}.png')
    ####################
    '''

    #WATSON TONE ANALYZER    
    authenticator=IAMAuthenticator(api_s)
    ta=ToneAnalyzerV3(version='2017-09-21', authenticator=authenticator)
    ta.set_service_url(url_s)
    dff=pd.DataFrame()
    for twe in tweets_no_urls:
        res=ta.tone(twe).get_result()
        try:
            new_row=res['document_tone']['tones'][0]
        except:
            continue
        new_row['tweet']=twe
        dff=dff.append(new_row, ignore_index=True)

    fig1, ax1 = plt.subplots()
    dt=dff.groupby("tone_name")["tone_id"].count()
    labels = dt.keys()
    colors = ['#FBDA08','#FACBF1','#59E816','#E83D16','#66b3ff','#B0ABA0','#AA214C']
    ax1.pie(dt,labels=labels,colors=colors, autopct='%1.1f%%', startangle=90)
    #draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    plt.savefig(f'C:/COMPANY/static/images/watson2_{uni_idx}.png')
    '''

    ####################

def scrapingCOMPANY():

    #####################################

    consumer_key= 'bPoE4mUtF4NkVYvmxaq86OlQ8'
    consumer_secret= 'Y92epzGZRRdOPBSxp1TjoZU5WmBy1AUaxabHaLbZnGBsFNpxz5'
    access_token= '1414537754151686146-OT8mryRlddBcYHgoNn1hUmHubXx10d'
    access_token_secret= 'vRMd4K0hZTpel2NCJJnmhlvXLXCe95mX0wUux4OpSvhNt'

    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api_tw = tw.API(auth, wait_on_rate_limit=True)

    #punctuations = list(string.punctuation)
    punctuations=[]
    punctuations.append('’')
    punctuations.append("'s")
    punctuations.append('“')
    punctuations.append('”')
    punctuations.append('click')
    punctuations.append('bio')
    punctuations.append('link')
    punctuations.append('u')
    punctuations.append('join')
    punctuations.append('help')


    extras_at=["@unibirmingham","@BristolUni","@Cambridge_Uni","@cardiffuni","@durham_uni","@EdinburghUni","@UniofExeter",
            "@UofGlasgow","@imperialcollege","@KingsCollegeLon","@UniversityLeeds","@LivUni","@LSEnews","@OfficialUoM",
            "@UniofNewcastle","@UniofNottingham","@UniofOxford","@QMUL","@QUBelfast","@sheffielduni","@unisouthampton",
            "@ucl","@warwickuni","@UniOfYork"]
    extras_no_at=["unibirmingham","bristoluni","cambridgeuni","cardiffuni","durham_uni","edinburghuni","uniofexeter",
            "uofglasgow","imperialcollege","kingscollegelon","universityleeds","livuni","lsenews","officialuom","uniofnewcastle",
            "uniofnottingham","uniofoxford","qmul","qubelfast","sheffielduni","unisouthampton","university ucl","warwickuni","uniofyork"]
    names_removal=["birmingham","bristol","cambridge","cardiff","durham","edinburgh","exeter","glasgow","london","london","leeds","liverpool",
       "london","manchseter","newcastle","nottingham","oxford","london","belfast","sheffield","southampton","london","warwick","york"]


    ########## SENTIMENT ############
    from COMPANY_watson import ToneAnalyzerV3
    from COMPANY_cloud_sdk_core.authenticators import IAMAuthenticator
    #################################

    try:

        all_tokens={}
        all_tweets={}
        all_full_tweet_object={}
        for russ in russels:
            rus_i=russels.index(russ)
            this_search_tweets=[]
            
            search_term = f" COMPANY {russ} -filter:retweets"
            search_tweets = api_tw.search(search_term,count=100,tweet_mode='extended')
            this_search_tweets+=search_tweets
            search_term = f"COMPANY {extras_at[rus_i]} -filter:retweets"
            search_tweets = api_tw.search(search_term,count=100,tweet_mode='extended')
            this_search_tweets+=search_tweets
            #no tweets found this week
            if len(this_search_tweets)==0:
                flag=1
                return flag,None,None,None

            tweets_no_urls =[tw.full_text for tw in this_search_tweets if not tw.lang!='en']
            tweets_no_urls =[remove_url(tweet) for tweet in tweets_no_urls]
            tweets_no_urls = list(set(tweets_no_urls))
            try:
                tweets_no_urls.remove('')
            except:
                pass
            #store
            all_tweets[russ]=tweets_no_urls
            all_full_tweet_object[russ]=this_search_tweets
            

            tokenized=tokenizer(tweets_no_urls)
            tokenized=stopwords(stop_words,tokenized)
            tokenized=punct(punctuations,tokenized)
            to_remove=[russ,extras_at[rus_i],extras_no_at[rus_i],'university','University','Congratulations','congratulations','done']
            tokenized=punct(to_remove,tokenized)
            tokenized=punct(names_removal,tokenized)
            all_tokens[russ]=tokenized


        for university in all_tweets.keys():
            all_tweets_COMPANY_old[university]+=all_tweets[university]
            all_tweets_COMPANY_old[university]=list(set(all_tweets_COMPANY_old[university]))
            all_tokens_COMPANY_old[university]+=all_tokens[university]

            merged_tokens=[]
            for token in all_tokens_COMPANY_old[university]:
                a=' '.join(token)
                merged_tokens.append(a)

            merged_tokens=list(set(merged_tokens))
            tokens_no_duplicates=[]
            for no_dup in merged_tokens:
                a=no_dup.split(' ')
                tokens_no_duplicates.append(a)

            all_tokens_COMPANY_old[university]=tokens_no_duplicates

            all_full_tweet_object_COMPANY_old[university]+=all_full_tweet_object[university]

            uniques=[]
            idces=[]
            for i,obj in enumerate(all_full_tweet_object_COMPANY_old[university]):
                if obj.full_text in uniques:
                    continue
                else:
                    uniques.append(obj.full_text)
                    idces.append(i)
            objects_no_duplicates=[]
            for ids in idces:
                objects_no_duplicates.append(all_full_tweet_object_COMPANY_old[university][ids])

            all_full_tweet_object_COMPANY_old[university]=objects_no_duplicates

        

        with open('C:/COMPANY/static/pickles/tempt/all_tokens_COMPANY.pickle', 'wb') as handle:
                pickle.dump(all_tokens_COMPANY_old, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('C:/COMPANY/static/pickles/tempt/all_tweets_COMPANY.pickle', 'wb') as handle:
            pickle.dump(all_tweets_COMPANY_old, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('C:/COMPANY/static/pickles/tempt/all_full_tweet_object_COMPANY.pickle', 'wb') as handle:
            pickle.dump(all_full_tweet_object_COMPANY_old, handle, protocol=pickle.HIGHEST_PROTOCOL)

        flag=0
    except:
        flag=1
        return flag,None,None,None
    return flag,all_tokens,all_tweets,all_full_tweet_object

        
    ######### END OF SCRAPING FUNC #####################


def create_graphsCOMPANY(russ,uni_idx):
            #START
    ###############
    #GRAPH ALL_ENTITIES
    '''
    Declare 
    this_entities dict
    tokenized list
    tweets_no_urls list
    retweets array list 
    this_search_tweets list of full tweet object (for the date)
    '''

    api_s='j3iyvaYX6wE5LYltz01xuGk0sCYV8Eti_gzHKudwDc8o'
    url_s='https://api.eu-gb.tone-analyzer.watson.cloud.COMPANY.com/instances/771d6bc4-dda8-4d65-a283-1c2b654604d7'

    
    #this_entities=all_entities[russ]
    tokenized=all_tokens_COMPANY_old[russ]
    tweets_no_urls=all_tweets_COMPANY_old[russ]
    this_search_tweets= all_full_tweet_object_COMPANY_old[russ]
    if len(this_search_tweets)==0:
        return None

    this_entities={}
    for tweet in this_search_tweets:
        aaa=tweet.entities['user_mentions']
        for a in aaa:
            if a['name'] in this_entities:
                this_entities[a['name']]+=1
            else:
                this_entities[a['name']]=1

    this_entities={k: v for k, v in sorted(this_entities.items(), key=lambda item: item[1],reverse=True)}
    entities_list=list(this_entities.keys())
    matchers = ['Uni', 'dr','Dr','University','university','College','college','UC','UK','England','Prof','Scho','scho','Journal',
            'Academ','academic','London','Institute','institute','Sport','Alliance','COMPANY','COMPANY','PhD','MSc','Msc',
            'Bsc','BSc','Museum','@','British','Knowledge','Foundation','Centre','Service','Research','Group','Institution',
            'Associa','Gallery','Art','Politic','Stud','Council','Community','Carreer','Faculty','Department','dept','Alumni','News']
    top_25=[s for s in entities_list if any(xs in s for xs in matchers)][0:20]
    scores_ent=[this_entities[a] for a in top_25]

    try:
        tops={}
        for a in top_25:
            tops[a]=this_entities[a]
        tops=pd.Series(tops)
        plt.figure(figsize=(10,5))
        sns.barplot( tops.values, tops.index,alpha=0.8)
        plt.title('Most mentioned entities',fontsize=12)
        plt.xlabel('Count of the citations', fontsize=12)
        #plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'C:/COMPANY/static/imagesCOMPANY/topent{uni_idx}.png')
    except:
        pass

    #################
    #TOPIC MODELLING
    dictionary = Dictionary(tokenized)
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    #dictionary.filter_extremes(no_below=10, no_above=0.8)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    # Set training parameters.
    num_topics = 50
    chunksize = 2000
    passes = 20
    iterations = 500
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha= 0.00056,
        eta=0.1,
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )


    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')
        plt.savefig(f'C:/COMPANY/static/imagesCOMPANY/cloud{uni_idx}.png')
    ####################
    #GRAPH MOST USED WORDS
    #concatenate tokens in one sentence 
    tt=[]
    tt2=[]
    for t in tokenized:
        tt2+=t
        a=' '.join(t)
        tt.append(a)

    topwords=pd.DataFrame(tt2)
    topwords = topwords[0].value_counts()

    topwords = topwords[0:25]
    plt.figure(figsize=(10,5))
    sns.barplot(topwords.values, topwords.index, alpha=0.8)
    plt.title('Top Words Overall')
    plt.ylabel('Word from Tweet', fontsize=12)
    plt.xlabel('Count of Words', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'C:/COMPANY/static/imagesCOMPANY/topwords{uni_idx}.png')
    ####################

    #GRAPH MOST USED BIGRAMS
    terms_bigram=[list(bigrams(doc)) for doc in tokenized]

    bigrams_l=list(itertools.chain(*terms_bigram))
    bigram_counts=collections.Counter(bigrams_l)
    bigram_top_20=bigram_counts.most_common(10)
    bigrams_w=[w[0] for w in bigram_top_20]
    bigrams_c=[c[1] for c in bigram_top_20]
    ind=[f"{v[0]}_{v[1]}" for v in bigrams_w]
    vals=[v for v in bigrams_c]
    tops=pd.Series(ind,vals)
    plt.figure(figsize=(10,5))
    sns.barplot(tops.values,tops.index,alpha=0.8)
    plt.title("Most frequent bigrams", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(f'C:/COMPANY/static/imagesCOMPANY/topbig{uni_idx}.png')


    ####################
    #SENTIMENT ANALYSIS


    analyzer = SentimentIntensityAnalyzer()
    atweet_list = (t.__dict__ for t in this_search_tweets)
    adf = pd.DataFrame(atweet_list)
    #adf['lang'] = adf['full_text'].map(lambda x: detect(x))
    #adf = adf[adf['lang']=='en']
    asentiment = adf['full_text'].apply(lambda x: analyzer.polarity_scores(x))
    adf = pd.concat([adf,asentiment.apply(pd.Series)],1)

    ####################
    #NEG/POS/NEU PLOT  
    adf['YN']=adf['compound'].apply(aYN2)
    fig,ax=plt.subplots(figsize=(6,4.5))
    ax = sns.countplot(x="YN", data=adf)
    ax.set(xlabel='Mood around the Tweet', ylabel='Number of Tweets')
    plt.savefig(f'C:/COMPANY/static/imagesCOMPANY/pol{uni_idx}.png')
    ####################

    #SENTIMENT OVER TIME
    adf.sort_values(by='created_at', inplace=True)
    adf.index = pd.to_datetime(adf['created_at'])
    adf['mean'] = adf['compound'].expanding().mean()
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.scatter(adf['created_at'],adf['compound'], label='Tweet Sentiment')
    ax.plot(adf['created_at'],adf['mean'], color='y', label='Expanding Mean')
    ax.set(title=f'{russ} Tweets over Time', xlabel='Date', ylabel='Sentiment')
    ax.legend(loc='best')
    ax.xaxis.set_major_locator(plt.MaxNLocator(9))
    plt.xticks(rotation=30)
    fig.tight_layout()
    plt.savefig(f'C:/COMPANY/static/imagesCOMPANY/compound{uni_idx}.png')

    ####################

    #COMPOUND DISTRIBUTION
    #adf=adf[adf['compound']!=0]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    sns.distplot(adf['compound'], bins=15, ax=ax)
    ax.set(title=f'{russ} Sentiment Scores', xlabel='Compound sentiment', ylabel='Density')
    plt.savefig(f'C:/COMPANY/static/imagesCOMPANY/overtime{uni_idx}.png')
    ####################
    '''

    #WATSON TONE ANALYZER    
    authenticator=IAMAuthenticator(api_s)
    ta=ToneAnalyzerV3(version='2017-09-21', authenticator=authenticator)
    ta.set_service_url(url_s)
    dff=pd.DataFrame()
    for twe in tweets_no_urls:
        res=ta.tone(twe).get_result()
        try:
            new_row=res['document_tone']['tones'][0]
        except:
            continue
        new_row['tweet']=twe
        dff=dff.append(new_row, ignore_index=True)

    fig1, ax1 = plt.subplots()
    dt=dff.groupby("tone_name")["tone_id"].count()
    labels = dt.keys()
    colors = ['#FBDA08','#FACBF1','#59E816','#E83D16','#66b3ff','#B0ABA0','#AA214C']
    ax1.pie(dt,labels=labels,colors=colors, autopct='%1.1f%%', startangle=90)
    #draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    plt.savefig(f'C:/COMPANY/static/images/watson2_{uni_idx}.png')
    '''

    ####################


