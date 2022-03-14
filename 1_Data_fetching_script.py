pip install requests
import requests 
import pandas as pd


def collecting_tweets(id_df):

  total_jsons = {}
  x = 0 
  url = 'https://recruitment.aimtechnologies.co/ai-tasks'
  while (x < len(id_df)):
    sample = id_df[x:x+1000].astype(str).tolist()
    r = requests.post(url , json = sample)
    j = r.json()
    total_jsons.update(j)
    x=x+1000
  tweets_df = pd.DataFrame(total_jsons.items() , columns = ['id' , 'tweet'])
  tweets_df.head()
  print(tweets_df.shape)

  return tweets_df


tweets_df = collecting_tweets(id_df)

dialect_dataset['id'] = dialect_dataset['id'].astype(str)

tweets_df['id'] = tweets_df['id'].astype(str)

final_df = dialect_dataset.merge(tweets_df, on='id', how='left')


final_df.to_csv(r'/content/final_df/final_df.csv' , index = False)


final_df = pd.read_csv('/content/final_df.csv' , lineterminator='\n')
final_df.head()