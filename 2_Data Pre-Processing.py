import re
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords



def remove_usernames_links(tweet):
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('http[^\s]+','',tweet)
    return tweet
final_df['tweet'] = final_df['tweet'].apply(remove_usernames_links)




def remove_emoji(tweet):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweet)


final_df['tweet'] = final_df['tweet'].apply(remove_emoji)

final_df.to_csv(r'/content/clean_final_df.csv' , index = False)




data['tokenized_tweets'] = data.apply(lambda row: nltk.word_tokenize(row['tweet']), axis=1)
sw = stopwords.words('arabic')
data['stopped_tweets'] = [i for i in data['tokenized_tweets'] if not i in sw]

#iterate over the tokenized_tweets column
len_tokens = []
word_tokens = data['tokenized_tweets']
for i in range(len(word_tokens)):
  len_tokens.append(len(word_tokens[i]))

data['n_tokens'] = len_tokens