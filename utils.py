import json
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from gensim import corpora, models
from gensim.models import CoherenceModel

from operator import itemgetter

# ==============================================
# ================ READING DATA ================
# ==============================================


def save_articles(articles, filename):
  """
  Save articles to a file.
  :param articles: articles to be saved
  """

  with open(filename, 'w', encoding='utf8') as fp:
    json.dump(articles, fp)
    

def read_json_file(filepath):
  """
  Read articles from a json file.

  :param filepath: path to a file to read
  :return : json data from file  
  """

  with open(filepath) as infile:
    data = json.load(infile)

  return data


def read_preprocessed_specific_media(media_list, year):
  """
  Read preprocessed articles of selected year and media from media_list.

  :param media_list: list of media to read preprocessed articles
  :param year: int or string of year from which articles are

  :return articles: list of preprocessed aricles
  """

  articles = []

  load_dir = f'/content/drive/MyDrive/Colab Notebooks/preprocessed_articles/{str(year)}/'
  for media in media_list:
    filepath = load_dir + media
    for article in read_json_file(filepath):
      articles.append(article)

  return articles


def read_raw_specific_media(media_list, year):
  """
  Read raw articles of selected year and media from media_list.

  :param media_list: list of media to read preprocessed articles
  :param year: int or string of year from which articles are

  :return data: list of preprocessed aricles
  """
  data = []
  load_dir = f'/content/drive/MyDrive/Colab Notebooks/raw_articles/{str(year)}/'
  for media in media_list:
    filepath = load_dir + media
    data = read_json_file(filepath)

  return data


def prepare_dataframe(media_list, year):
  """
  Prepare a dataframe with information of articles of media from media_list and
  specific year. 
  We remove short articles (shorter than 25 words) from the data.
  We also remove duplicated articles (same title)

  :param media_list: list of media to read preprocessed articles
  :param year: int or string of year from which articles are

  :return df_full: pandas Dataframe object.
  It contains next columns:
    - body: string, raw article
    - title: string, title of article
    - media: string, name of media that wrote the article
    - word_length: int, number of words in raw article
    - preprocessed_body: list(str), preprocessed article
  """

  df_full = pd.DataFrame()
  for media in media_list:
    print(media)
    df = pd.DataFrame.from_dict(read_raw_specific_media([media], year))
    df['media'] = media
      
    df['word_length'] = df.body.apply(lambda x: len(str(x).split()))
    # print(len(df.word_length))

    try:
      print('First short articles...')
      df1 = df.loc[df['word_length'] > 25]
      df1 = df.drop_duplicates(subset='title', keep="last")
      df1['preprocessed_body'] = read_preprocessed_specific_media([media], year)
    except:
      print('First duplicates...')
      df = df.drop_duplicates(subset='title', keep="last")
      df = df.loc[df['word_length'] > 25]
      df['preprocessed_body'] = read_preprocessed_specific_media([media], year)
      df1 = df

    

    
    df_full = df_full.append(df1, ignore_index=True)

  return df_full



# ==============================================
# ================ VIZUALIZATIONS ==============
# ==============================================

def visualize_articles_by_media(media_names, counts):
  """
  Plotting a distribution of articles across top 10 media
  
  :param articles: a dictionary articles to plot
  :param num_of_articles_by_media: a list of counts of articles by each media
  """

  fig, ax = plt.subplots()
  ax.barh(np.arange(len(media_names)), counts)
  ax.set_yticks(np.arange(len(media_names)))
  ax.set_yticklabels(media_names)
  ax.invert_yaxis()  # labels read top-to-bottom

  for i, v in enumerate(counts):
      ax.text(v + 3, i + .25, str(v))




def dataframe_info(df, column, media=""):
  """
  Print detailed information about given column in dataframe df and plotting it.
  """
  print(df.describe())

  plt.figure(figsize=(12,6)) 
  p1=sns.kdeplot(df[column], shade=True, color='r').set_title(f'Distribucija {column} v Ã„Å’lankih ' + media)


def visualize_topic_distribution(df, df_topic_info):

  num_articles_dict = df.media.value_counts().to_dict()
  media_list = ['MMC RTV Slovenija', 'Siol.net Novice', '24ur.com', 'Dnevnik']
  color_list = {
      media_list[0]: 'tab:green',
      media_list[1]: 'tab:blue',
      media_list[2]: 'tab:orange',
      media_list[3]: 'tab:purple',
  }
  if len(list(num_articles_dict.keys()))  > 4:
    media_list.append('Tednik Demokracija')
    media_list.append('Nova24TV')
    media_list.append('PortalPolitikis')
    color_list = {
        media_list[0]: 'tab:green',
        media_list[1]: 'tab:blue',
        media_list[2]: 'tab:orange',
        media_list[3]: 'tab:purple',
        media_list[4]: 'tab:brown',
        media_list[5]: 'tab:red',
        media_list[6]: 'tab:pink',
    }

  topic_stats = df.groupby(['topic_id', 'media']).size()
  topic_names = df_topic_info.loc[:, 'topic_name']

  topic_count_per_media = {}

  for topic in range(len(topic_names)):

    for media in media_list:

      if topic == 0:
        topic_count_per_media[media] = []

      if media not in topic_stats[topic]:
        topic_count_per_media[media].append(0)
        continue
      
      num_articles = num_articles_dict[media]
      num_articles_topic = topic_stats[topic][media]
      pct = num_articles_topic / num_articles
      topic_count_per_media[media].append(pct)


  bars = pd.DataFrame(topic_count_per_media, index=topic_names)

  bar_args = {
      'figsize': (15,10),
      'xlabel': 'Topic name',
      'ylabel': '%',
      'color': color_list
  }

  bars.plot.bar(**bar_args)



# ================================================
# ================ TOPIC MODELING ================
# ================================================



# ================ DATA PREPARATION ==============

def remove_stopwords(article, stopwords):
  for word in article:
    if word in stopwords:
      article.remove(word)

  return article

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def generate_ngrams(text, n_gram=1):
    ngrams = zip(*[text[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]


def LDA_data_preparation(df, n_gram=1, min_count=10, threshold=80, no_below=15, no_above=0.5, keep_n=100000):

  documents = df.preprocessed_body

  if n_gram == 2:
    # Build the bigram and trigram models
    bigram = models.Phrases(documents, min_count=min_count, threshold=threshold)
    bigram_mod = models.phrases.Phraser(bigram)
    documents = make_bigrams(documents.to_list(), bigram_mod)

  elif n_gram == 3:
    # Build the trigram and trigram models
    bigram = models.Phrases(documents, min_count=min_count, threshold=threshold)
    bigram_mod = models.phrases.Phraser(bigram)
    trigram = models.Phrases(documents, min_count=min_count, threshold=threshold)
    trigram_mod = models.phrases.Phraser(trigram)
    documents = make_trigrams(documents.to_list(), bigram_mod, trigram_mod)  
  
  df['preprocessed_body'] = documents
  dictionary = corpora.Dictionary(documents)
  dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
  corpus = [dictionary.doc2bow(doc) for doc in documents]

  print(f'Length of a dictionary (id:word): {len(dictionary)}')
  print(f'Length of corpus (id:count): {len(corpus)}')

  return dictionary, corpus, documents



# ==================== LDA MODEL ==================

def LDA(dictionary, corpus, save_dir, num_topics, chunksize, passes, iterations, eval_every, alpha='auto', eta='auto', random_state=23):

  # Make a index to word dictionary.
  temp = dictionary[0]  # This is only to "load" the dictionary.
  id2word = dictionary.id2token

  lda_model = models.LdaModel(
      corpus=corpus,
      id2word=dictionary,
      chunksize=chunksize,
      alpha=alpha,
      eta=eta,
      iterations=iterations,
      num_topics=num_topics,
      passes=passes,
      random_state=random_state,
      eval_every=eval_every
  )


  for idx, topic in lda_model.print_topics(-1):
      print('Topic: {} \nWords: {}'.format(idx, topic))

  lda_model.save(save_dir)

  return lda_model


def LDA_load(model_path, preprocessed_articles):

  lda_model =  models.LdaModel.load(model_path)
  dictionary = lda_model.id2word
  bow_corpus = [dictionary.doc2bow(article) for article in preprocessed_articles]

  return dictionary, bow_corpus, lda_model


def compute_coherence_score(lda_model, preprocessed_articles, dictionary):

  # Compute Coherence Score
  coherence_model_lda = CoherenceModel(model=lda_model, texts=preprocessed_articles, dictionary=dictionary, coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()

  return coherence_lda


def compute_coherence_values(dictionary, bow_corpus, preprocessed_articles, save_dir, start, limit, step, lda_parameters):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """

    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit, step):
        filepath = f'{save_dir}_{str(num_topics)}'
        print(f'=========== SAVING TO: {filepath} ===========')
        lda_model = LDA(dictionary, bow_corpus, filepath, num_topics=num_topics, **lda_parameters)
        model_list.append(lda_model)
        coherence = compute_coherence_score(lda_model, preprocessed_articles, dictionary)
        print(coherence)
        coherence_values.append(coherence)

    return model_list, coherence_values



# ======================================================
# ================ TOPIC INTERPRETATION ================
# ======================================================


def assign_doc_topics(df, lda_model, dictionary):
  df = doc_topic(df, lda_model, dictionary)
  df = df.sort_values('topic_score', ascending=False)

  return df


def doc_topic(df, lda_model, dictionary):

  topic_ids = []
  topic_scores = []
  # loop over dataframe of articles
  for index, article in df.iterrows():
    # print(article)
    # define corpus of article_body and get the most probable topic for it
    corpus = dictionary.doc2bow(article.preprocessed_body)
    topic_id, score = max(lda_model[corpus],key=itemgetter(1))

    # store topic_id to data frame
    topic_ids.append(topic_id)
    topic_scores.append(score)
    # print(df.loc[df.index == index].topic_id)
    # break

  df['topic_id'] = topic_ids
  df['topic_score'] = topic_scores

  return df


def doc_topic_dist(df):
  
  topic_dist = df.topic_id.value_counts().to_dict()
  
  return topic_dist


def get_topic_info(df, topic_data, lambd=0.6, num_terms=20):

  topic_dist = doc_topic_dist(df)
  df_topic_info = pd.DataFrame(topic_dist.items())
  df_topic_info.columns = ['topic_id', 'count']

  df_topic_info = df_topic_info.sort_values('topic_id')

  topic_titles = []

  for _, row in df_topic_info.iterrows():
    topic_id = row.topic_id

    ixs = df.index[df.topic_id == topic_id].tolist()

    top_titles = df.loc[ixs[:20], 'title'].to_list()
    topic_titles.append(top_titles)


  df_topic_info['topic_titles'] = topic_titles
  df_topic_info.head()


  all_topics = {}

  for i in range(1,len(df_topic_info)+1): #Adjust this to reflect number of topics chosen for final LDA model
      topic = topic_data.topic_info[topic_data.topic_info.Category == 'Topic'+str(i)].copy()
      topic['relevance'] = topic['loglift']*(1-lambd)+topic['logprob']*lambd
      all_topics['Topic '+str(i)] = topic.sort_values(by='relevance', ascending=False).Term[:num_terms].values.tolist()

  df_topic_info['topic_words'] = list(all_topics.values())

  return df_topic_info


def interprete_topics(df_topic_dist):
  topic_names = []

  for index, row in df_topic_dist.iterrows():
    topic_id = row.topic_id
    titles = row.topic_titles
    words = row.topic_words
    print(topic_id, words)
    for title in titles:
      print(title)
    topic_name = input('Vnesi ime teme: ')
    topic_names.append(topic_name)
    print('=========================================')

  df_topic_dist['topic_name'] = topic_names
  df_topic_dist.head()

  return df_topic_dist