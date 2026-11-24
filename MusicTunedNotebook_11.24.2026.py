# -*- coding: utf-8 -*-
"""DanielHolloman16.5_Music_Recommendation_System_Reference_Notebook_Low_Code_01.18.2026.ipynb


# **Music Recommendation System**

## **Problem Definition**

### **The Context:**

QUESTION:
 - Why is this problem important to solve?

 **ANSWER:** Recommended products on ecommerce website Amazon make up 35% of all purchases (SOURCE). Other companies such as Netflix and Spotify also rely on recommendation systems to predict which movies, series, and songs will appeal to their large user base. Highly precise recommendation systems can help increase sales for many businesses. If customers are happy with recommendations they tend to stay customers and consume more product.

### **The objective:**

QUESTION:
 - What is the intended goal?

 **ANSWER:** Over the course of this project, we will prepare song and user data and process it in such a way that we can make recommendations to customers. By the end of this notebook we will be able to recommend 10 songs to a user based on their music preferences and their similarity to other users on the platform.

### **The key questions:**
QUESTION:
- What are the key questions that need to be answered?

**ANSWER:** We shall determine which songs are similar and which users are similar. This will give us a basis for the final question, where we will answer what song should be played to the user next.

### **The problem formulation**:

QUESTION:
- What is it that we are trying to solve using data science?

**ANSWER:** We are ultimately trying to solve the boredom problem. Users rely on music to pass the time and enrich moments of relaxation, studying, or partying. By offering a valuable service to customers we can extract more subscription fees from subscribers and play more ads for free accounts. Recommendation systems are a vehicle for keeping users engaged and consuming our content, leading to higher profits for the streaming service company.

## **Data Dictionary**

The core data is the Taste Profile Subset released by the Echo Nest as part of the Million Song Dataset. There are two files in this dataset. The first file contains the details about the song id, titles, release, artist name, and the year of release. The second file contains the user id, song id, and the play count of users.

**song_data**

- song_id - A unique id given to every song
- title - Title of the song
- Release - Name of the released album
- Artist_name - Name of the artist
- year - Year of release

**count_data**

- user _id - A unique id given to the user
- song_id - A unique id given to the song
- play_count - Number of times the song was played

## **Data Source**
http://millionsongdataset.com/

## **Important Notes**

- This notebook can be considered a guide to refer to while solving the problem. The evaluation will be as per the Rubric shared for the Milestone. Unlike previous courses, it does not follow the pattern of the graded questions in different sections. This notebook would give you a direction on what steps need to be taken to get a feasible solution to the problem. Please note that this is just one way of doing this. **There can be other 'creative' ways to solve the problem, and we encourage you to feel free and explore them as an 'optional' exercise**.

- In the notebook, there are markdown cells called Observations and Insights. It is a good practice to provide observations and extract insights from the outputs.

- The naming convention for different variables can vary. **Please consider the code provided in this notebook as a sample code.**

- All the outputs in the notebook are just for reference and can be different if you follow a different approach.

- There are sections called **Think About It** in the notebook that will help you get a better understanding of the reasoning behind a particular technique/step. Interested learners can take alternative approaches if they want to explore different techniques.

### **Importing Libraries and the Dataset**

**Note**: After running this commands, you’ll be prompted to restart your notebook/runtime. Please go ahead and do that. Once restarted, simply proceed to the next step of your workflow
"""

!pip install numpy==1.26.0

# Mounting the drive
from google.colab import drive
drive.mount('/content/drive')

# Used to ignore the warning given as output of the code
import warnings
warnings.filterwarnings('ignore')

# Basic libraries of python for numeric and dataframe computations
import numpy as np
import pandas as pd

# Basic library for data visualization
import matplotlib.pyplot as plt

# Slightly advanced library for data visualization
import seaborn as sns

# To compute the cosine similarity between two vectors
from sklearn.metrics.pairwise import cosine_similarity

# A dictionary output that does not raise a key error
from collections import defaultdict

# A performance metrics in sklearn
from sklearn.metrics import mean_squared_error

"""### **Load the dataset**"""

#importing the datasets
count_df = pd.read_csv('/content/drive/MyDrive/MIT_AAIDSP_GL/Recommendation_Systems/Capstone_Music_Rec_Systems/count_data.csv')
song_df = pd.read_csv('/content/drive/MyDrive/MIT_AAIDSP_GL/Recommendation_Systems/Capstone_Music_Rec_Systems/song_data.csv')

"""### **Understanding the data by viewing a few observations**"""

# Display the first 10 records of count_df
count_df.head(10)

# Display the first 10 records of song_df
song_df.head(10)

"""### **Let us check the data types and and missing values of each column**"""

# Display the info of count_df
count_df.info()

# Display the info of song_df
song_df.info()

"""#### **Observations and Insights:**
**ANSWER:** 2,000,000 entries of how often a song is played paired with the user that listened to the song. 1,000,000 songs paired with artist names. There are 17 null values in the title column and 7 missing values in the release column.

"""

# Left merge count_df and song_df on "song_id". Drop duplicates from song_df data simultaneously
df = pd.merge(count_df, song_df.drop_duplicates(['song_id']), on="song_id", how="left")

# Drop the column 'Unnamed: 0'
df = df.drop(['Unnamed: 0'],axis=1)
df

"""**Think About It:** As the user_id and song_id are encrypted. Can they be encoded to numeric features?

**ANSWER:** Yes, it can be encoded to numeric form and that will result in easier analysis, since algorithms work well with numbered data such as integers. This technique is shown in the below cell.
"""

# Apply label encoding for "user_id" and "song_id"
# Label encoding code
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['user_id'] = le.fit_transform(df['user_id'])

df['song_id'] = le.fit_transform(df['song_id'])

"""### **New CELL**"""

df['user_id']

"""### **New CELL**"""

df['song_id'].head()

"""### **New CELL**"""

df

"""**Think About It:** As the data also contains users who have listened to very few songs and vice versa, is it required to filter the data so that it contains users who have listened to a good count of songs and vice versa?

**ANSWER:** Filtering out the users who have low play counts helps us visualize a more complete dataset of actual users and increases our prediction capabilities.

"""

# Get the column containing the users
users = df.user_id

# Create a dictionary that maps users(listeners) to the number of songs that they have listened to
playing_count = dict()

for user in users:
    # If we already have the user, just add 1 to their playing count
    if user in playing_count:
        playing_count[user] += 1

    # Otherwise, set their playing count to 1
    else:
        playing_count[user] = 1

# We want our users to have listened at least 90 songs
SONG_COUNT_CUTOFF = 90

# Create a list of users who need to be removed
remove_users = []

for user, num_songs in playing_count.items():

    if num_songs < SONG_COUNT_CUTOFF:
        remove_users.append(user)

df = df.loc[ ~ df.user_id.isin(remove_users)]

# Get the column containing the songs
songs = df.song_id

# Create a dictionary that maps songs to its number of users(listeners)
playing_count = dict()

for song in songs:
    # If we already have the song, just add 1 to their playing count
    if song in playing_count:
        playing_count[song] += 1

    # Otherwise, set their playing count to 1
    else:
        playing_count[song] = 1

# We want our song to be listened by atleast 120 users to be considred
LISTENER_COUNT_CUTOFF = 120

remove_songs = []

for song, num_users in playing_count.items():
    if num_users < LISTENER_COUNT_CUTOFF:
        remove_songs.append(song)

df_final= df.loc[ ~ df.song_id.isin(remove_songs)]

"""Out of all the songs available, songs with play_count less than or equal to 5 are in almost 90% abundance. So for building the recommendation system let us consider only those songs.

**COLAB ANSWERED:** This statement suggests that a large majority (around 90%) of the songs in the dataset have been played 5 times or less. In the context of building a recommendation system, this observation is significant for a few reasons:

1. Sparsity Handling: Recommendation systems often deal with very sparse data, meaning most users have only interacted with a small fraction of available items. Songs with very low play counts contribute to this sparsity. Including them all could make the recommendation process less efficient and potentially less accurate, as there's less information to learn from for these rarely played songs.

2. Focus on meaningful interactions: A play_count of 1 or 2 might indicate a casual listen or even an accidental play, rather than a strong preference. By focusing on songs with slightly higher play_count (or, in this case, by excluding songs with very high play counts and focusing on the 'long tail' of less popular items if the intention is to recommend more niche content, or by excluding items with extremely low play counts if the goal is to focus on items with at least some proven interaction), the system can prioritize interactions that are more likely to reflect genuine user interest.

3. Computational Efficiency: Including a massive number of very rarely played songs can significantly increase the computational resources needed to build and train models, especially for matrix factorization or collaborative filtering algorithms, without necessarily adding much value.

4. Specific Goal of the Recommendation System: While the statement says to consider only those songs with play_count <= 5, this wording might imply a specific strategy. If the goal is to recommend popular songs, then focusing on songs with higher play_count would make sense. However, if the goal is to discover and recommend less popular but potentially relevant songs (addressing the 'long tail' problem), then considering songs with lower, but not extremely low, play counts could be valuable.

In essence, the notebook is making a decision to filter the data to retain a subset of songs that are considered more relevant or manageable for the specific recommendation task at hand, given the distribution of play counts.
"""

# Keep only records of songs with play_count less than or equal to (<=) 5
df_final = df[df['play_count'] <= 5]

# Check the shape of the data
df_final.shape

"""## **Exploratory Data Analysis**

### **Let's check the total number of unique users, songs, artists in the data**

Total number of unique user id
"""

# Display total number of unique user_id
users.nunique()

"""Total number of unique song id"""

# Display total number of unique song_id
songs.nunique()

"""Total number of unique artists"""

# Display total number of unique artists
df['artist_name'].nunique()

"""#### **Observations and Insights:_________**

**MY ANSWER:** There are only about 10K unique songs in the whole database, even though it is called the million song db. That is a misnomer in my opinion. There are approximately 76K unique users and about 3K unique artists in the database. Since unique artists is much lower than unique songs, the data implies that many artists appear in the database multiple times for different songs they have released.

### **Let's find out about the most interacted songs and interacted users**

Most interacted songs
"""

df_final.groupby('title')['play_count'].count().sort_values(ascending = False).head(10)

"""Most interacted users"""

df_final.groupby('user_id')['play_count'].count().sort_values(ascending = False).head(10)

"""#### **Observations and Insights:_______**
#### **MY ANSWER:** The max interacted song was "Use Somebody" by Kings of Leon with 751 plays and max user interactions was from user 75144 with a total of 621 songs played.

Songs released on yearly basis
"""

# count of the songs in an year using the title count
count_songs = song_df.groupby('year').count()['title']
count = pd.DataFrame(count_songs)
count.drop(count.index[0], inplace = True)
count.tail()

plt.figure(figsize = (30,10))
ax = sns.barplot(x = count.index,
            y = 'title',
            data = count,
            estimator = np.median,)
for item in ax.get_xticklabels(): item.set_rotation(90)
plt.ylabel('number of songs released')
# Show the plot
plt.show()

"""####**Extra Code Cell**"""

import matplotlib.ticker as ticker

plt.figure(figsize = (30,10))
ax = sns.barplot(x = count.index,
            y = 'title',
            data = count,
            estimator = np.median,
            palette = 'viridis') # Added a color palette for visual appeal

# Set the maximum number of x-axis ticks for better readability
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15))
plt.tick_params(axis='x', labelsize=14)

# Set y-axis tick label size
plt.tick_params(axis='y', labelsize=14)

# Set the title and labels
plt.title('Number of Songs Released Per Year', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Number of Songs Released', fontsize=16)

for item in ax.get_xticklabels():
    item.set_rotation(90)

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

import matplotlib.ticker as ticker

plt.figure(figsize = (30,10))
ax = sns.barplot(x = count.index,
            y = 'title',
            data = count,
            estimator = np.median,
            palette = 'viridis') # Added a color palette for visual appeal

# Set the maximum number of x-axis ticks for better readability
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=19))
plt.tick_params(axis='x', labelsize=20)

# Set y-axis tick label size
plt.tick_params(axis='y', labelsize=20)

# Set the title and labels
plt.title('Number of Songs Released Per Year', fontsize=25)
plt.xlabel('Year', fontsize=24)
plt.ylabel('Number of Songs Released', fontsize=24)

for item in ax.get_xticklabels():
    item.set_rotation(45)

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# Define the bin edges for decades (including the last year present in the data)
decade_bins = [1919, 1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2011]

# Define labels for the decades
decade_labels = ['1920s', '1930s', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s']

# Create a new DataFrame with years binned by decade
# Ensure 'year' is a column to use pd.cut
count_reset = count.reset_index()
count_reset['decade'] = pd.cut(count_reset['year'], bins=decade_bins, labels=decade_labels, right=True)

# Group by decade and sum the number of songs
decade_counts = count_reset.groupby('decade')['title'].sum().reset_index()

# Display the aggregated data
display(decade_counts.head())

import matplotlib.ticker as ticker

plt.figure(figsize = (15, 8))
ax = sns.barplot(x = 'decade',
            y = 'title',
            data = decade_counts,
            palette = 'viridis') # Added a color palette for visual appeal

# Set x-axis tick label size
plt.tick_params(axis='x', labelsize=12)

# Set y-axis tick label size
plt.tick_params(axis='y', labelsize=12)

# Set the title and labels
plt.title('Number of Songs Released Per Decade', fontsize=16)
plt.xlabel('Decade', fontsize=14)
plt.ylabel('Number of Songs Released', fontsize=14)

for item in ax.get_xticklabels():
    item.set_rotation(45)

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout() # Adjust layout to prevent labels from being cut off

# Show the plot
plt.show()

"""####**PARETO CHART TEST**"""

import matplotlib.ticker as ticker

# Ensure 'title' is recognized as the column for song counts
decade_counts_for_plot = decade_counts.copy()
decade_counts_for_plot = decade_counts_for_plot.sort_values(by='decade', ascending=True)

# Calculate cumulative sum of songs
decade_counts_for_plot['cumulative_sum'] = decade_counts_for_plot['title'].cumsum()

# Calculate total number of songs to get cumulative percentage
total_songs = decade_counts_for_plot['title'].sum()
decade_counts_for_plot['cumulative_percentage'] = (decade_counts_for_plot['cumulative_sum']/total_songs)*100

# Create a figure and a set of subplots
fig, ax1 = plt.subplots(figsize=(15, 8))

# Plot the bar chart for number of songs released per decade on the first y-axis (ax1)
sns.barplot(x='decade', y='title', data=decade_counts_for_plot, palette='viridis', ax=ax1)

# Set labels and title for ax1
ax1.set_title('Number of Songs Released Per Decade with Cumulative Percentage', fontsize=20)
ax1.set_xlabel('Decade', fontsize=16)
ax1.set_ylabel('Number of Songs Released', fontsize=16)
ax1.tick_params(axis='x', labelsize=14, rotation=45)
ax1.tick_params(axis='y', labelsize=14)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Create a second y-axis (ax2) that shares the same x-axis as ax1
ax2 = ax1.twinx()

# Plot the cumulative percentage line on the second y-axis (ax2)
sns.lineplot(x='decade', y='cumulative_percentage', data=decade_counts_for_plot,
             marker='o', sort=False, color='red', linewidth=2, ax=ax2)

# Set labels and limits for ax2
ax2.set_ylabel('Cumulative Percentage (%)', color='red', fontsize=16)
ax2.tick_params(axis='y', labelcolor='red', labelsize=14)
ax2.set_ylim(0, 100) # Ensure the cumulative percentage goes from 0 to 100

# Add percentage labels on the line plot points
for i, txt in enumerate(decade_counts_for_plot['cumulative_percentage']):
    ax2.annotate(f'{txt:.1f}%', (i, decade_counts_for_plot['cumulative_percentage'].iloc[i]),
                 textcoords="offset points", xytext=(0,10), ha='center', color='red', fontsize=12)

plt.tight_layout() # Adjust layout to prevent labels from being cut off
plt.show()

# Group by year and sum the play_count, excluding year 0
yearly_play_counts = df_final[df_final['year'] != 0].groupby('year')['play_count'].sum()

# Find the maximum play count and the corresponding year
max_play_count_year = yearly_play_counts.idxmax()
max_play_count = yearly_play_counts.max()

print(f"The year with the maximum number of song plays is {max_play_count_year} with a total of {max_play_count} plays.")

"""import matplotlib.ticker as ticker

plt.figure(figsize = (15, 8))
ax2 = sns.barplot(x = 'decade',
            y = 'title',
            data = decade_counts,
            palette = 'viridis') # Added a color palette for visual appeal

plt.figure(figsize = (15, 8))
ax3 = sns.lineplot(x = 'decade',
            y = 'title',
            data = decade_counts,
            palette = 'viridis') # Added a color palette for visual appeal

# Set x-axis tick label size
plt.tick_params(axis='x', labelsize=12)

# Set y-axis tick label size
plt.tick_params(axis='y', labelsize=12)

# Set the title and labels
plt.title('Number of Songs Released Per Decade', fontsize=16)
plt.xlabel('Decade', fontsize=14)
plt.ylabel('Number of Songs Released', fontsize=14)

for item in ax2.get_xticklabels():
    item.set_rotation(45)

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout() # Adjust layout to prevent labels from being cut off

# Show the plot
plt.show()

#### **Observations and Insights:__________**
####**ANSWER:** The count of song releases by year starts small in the 1920s, when artists relied on pressing companies and agents to record their music. Fast forward to 1979, the advent of the Sony Walkman and similar music playing devices spurred growth of song releases as well as sales of albums on cassette and eventually on compact disc. In 1981 MTV took off, raising awareness of musicians as bands scrambled to make music videos, spurring growth of audio recordings. Beginning in the early to mid 90s digital music began to take over. And much of all physical media was digitized across the world. Then around 2001 the first iPod was released, and the online music business exploded. As millions of legitimate purchases were loaded onto iPods, so too did bootleg copies of albums. In 2005, the advent of Youtube meant that artists could self-publish their musical works and get more exposure than ever before. Then in early 2007 the iPhone was released, putting a bare bones recording studio in the pockets of many musicians around the world. The music business has flourished since then, in part because of music platforms like Bandcamp and Soundcloud in 2008. Although creative expression through music was thriving, many consumers and musicians criticized big business for stealing profits from small time musicians around the world.

**Think About It:** What other insights can be drawn using exploratory data analysis?

## **Important Insights from EDA**

What are the the most important observations and insights from the data based on the EDA performed?

Now that we have explored the data, let's apply different algorithms to build recommendation systems.

## Building various models

### **Popularity-Based Recommendation Systems**

Let's take the count and sum of play counts of the songs and build the popularity recommendation systems based on the sum of play counts.
"""

# Calculating average play_count
average_count = df_final.groupby('song_id')['play_count'].mean()    # Hint: Use groupby function on the song_id column

# Calculating the frequency a song is played
play_freq = df_final.groupby('song_id')['play_count'].count()    # Hint: Use groupby function on the song_id column

# Making a dataframe with the average_count and play_freq
final_play = pd.DataFrame({'avg_count':average_count, 'play_freq':play_freq})

# Let us see the first five records of the final_play dataset
final_play.head()

"""Now, let's create a function to find the top n songs for a recommendation based on the average play count of song. We can also add a threshold for a minimum number of playcounts for a song to be considered for recommendation.

****NEEDS REVISION:**** This appears to be the top 10 songs based on item-item similarity
"""

# Build the function to find top n songs
def top_n_songs(data, n, min_interaction = 1):
  result = data[data['play_freq'] >= min_interaction]
  return result

# Recommend top 10 songs using the function defined above
top_n_songs(final_play, 10)

"""### **User User Similarity-Based Collaborative Filtering**

To build the user-user-similarity-based and subsequent models we will use the "surprise" library.
"""

# Install the surprise package using pip.
!pip install surprise

# Import necessary libraries

# To compute the accuracy of models
from surprise import accuracy

# This class is used to parse a file containing play_counts, data should be in structure - user; item; play_count
from surprise.reader import Reader

# Class for loading datasets
from surprise.dataset import Dataset

# For tuning model hyperparameters
from surprise.model_selection import GridSearchCV

# For splitting the data in train and test dataset
from surprise.model_selection import train_test_split

# For implementing similarity-based recommendation system
from surprise.prediction_algorithms.knns import KNNBasic

# For implementing matrix factorization based recommendation system
from surprise.prediction_algorithms.matrix_factorization import SVD

# For implementing KFold cross-validation
from surprise.model_selection import KFold

# For implementing clustering-based recommendation system
from surprise import CoClustering

"""### Some useful functions

Below is the function to calculate precision@k and recall@k, RMSE and F1_Score@k to evaluate the model performance.

**Think About It:** Which metric should be used for this problem to compare different models?

####**REVISE CODE**
"""

def precision_recall_at_k(model, k = 30, threshold = 1.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)

    #Making predictions on the test data
    predictions = model.test(testset)

    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, playing_count in user_est_true.items():

        # Sort play count by estimated value
        playing_count.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in playing_count)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in playing_count[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in playing_count[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set Precision to 0 when n_rec_k is 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set Recall to 0 when n_rel is 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    #Mean of all the predicted precisions are calculated.
    precision = round((sum(prec for prec in precisions.values()) / len(precisions)),3)
    #Mean of all the predicted recalls are calculated.
    recall = round((sum(rec for rec in recalls.values()) / len(recalls)),3)

    accuracy.rmse(predictions)
    print('Precision: ', precision) #Command to print the overall precision
    print('Recall: ', recall) #Command to print the overall recall
    print('F_1 score: ', round((2*precision*recall)/(precision+recall),3)) # Formula to compute the F-1 score.

"""**Think About It:** In the function precision_recall_at_k above the threshold value used is 1.5. How precision and recall are affected by changing the threshold? What is the intuition behind using the threshold value of 1.5?

#___________________________________________
####**COLAB ANSWER:** The intuition behind using a threshold value of 1.5 for play_count in the precision_recall_at_k function is to differentiate between casual interactions and more significant indications of user preference.

####In this recommendation system, play_count serves as a proxy for a user's 'rating' of a song. While we don't have explicit ratings (like 1-5 stars), we assume that the more a user plays a song, the more they like it. The Reader was instantiated with rating_scale=(0, 5), implying play_count values are interpreted within this range.

####play_count = 1: A single play might be a casual listen, an accidental click, or someone trying out a song and not necessarily enjoying it enough to repeat it. It's a very weak signal of preference.

####play_count >= 2: A play_count of 2 or more indicates that the user returned to the song, suggesting a stronger, more intentional engagement and likely enjoyment.
By setting the threshold at 1.5:

####Any song with a play_count of 1 (which is less than 1.5) will not be considered 'relevant' when calculating precision and recall. These are treated as non-preferred items.

####Any song with a play_count of 2 or higher (which is greater than or equal to 1.5) will be considered 'relevant'. These are the songs we want our system to recommend.

####This threshold helps us focus on evaluating the model's ability to recommend songs that users genuinely prefer and engage with multiple times, rather than songs they might have only listened to once without much commitment.

#______________________________________________

Below we are loading the **dataset**, which is a **pandas dataframe**, into a **different format called `surprise.dataset.DatasetAutoFolds`** which is required by this library. To do this we will be **using the classes `Reader` and `Dataset`**

You will also notice here that we read the dataset by providing a scale of ratings. However, as you would know, we do not have ratings data of the songs. In this case, we are going to use play_count as a proxy for ratings with the assumption that the more the user listens to a song, the higher the chance that they like the song
"""

# Instantiating Reader scale with expected rating scale
reader = Reader(rating_scale=(0, 5)) #use rating scale (0, 5)

# Loading the dataset
data = Dataset.load_from_df(df_final[['user_id', 'song_id', 'play_count']], reader) # Take only "user_id","song_id", and "play_count"

# Splitting the data into train and test dataset
trainset, testset = train_test_split(data, test_size=0.4, random_state = 42) # Take test_size = 0.4

"""**Think About It:** How changing the test size would change the results and outputs?

####**REVISE**
"""

# Build the default user-user-similarity model
sim_options = {'name': 'pearson',
               'user_based':True}

# KNN algorithm is used to find desired similar items
sim_user_user = KNNBasic(sim_options=sim_options, random_state=1) # Use random_state = 1

# Train the algorithm on the trainset, and predict play_count for the testset
sim_user_user.fit(trainset)

# Let us compute precision@k, recall@k, and f_1 score with k = 30
precision_recall_at_k(sim_user_user) # Use sim_user_user model

"""**Observations and Insights:_________**

#### **COLAB: The default user-user similarity model has been successfully trained and evaluated. Here are the performance metrics:**

**COLAB: RMSE: 1.1854 - This is the Root Mean Squared Error, indicating the average magnitude of the errors in predicting play counts.
Precision@K: 0.395 - This means that 39.5% of the recommended songs are relevant to the user.
Recall@K: 0.621 - This indicates that the model is able to recommend 62.1% of all relevant songs to a user.
F1-Score: 0.483 - This is the harmonic mean of precision and recall, providing a balanced measure of the model's accuracy.
These metrics establish a baseline for our user-user collaborative filtering model. We will aim to improve these values through hyperparameter tuning.**
"""

# Predicting play_count for a sample user with a listened song
sim_user_user.predict(6958, 1671, r_ui = 2, verbose = True) # Use user id 6958 and song_id 1671

# Predicting play_count for a sample user with a song not-listened by the user
sim_user_user.predict(6958, 3232, verbose = True) # Use user_id 6958 and song_id 3232

"""**Observations and Insights:_________**

####**COLAB ANSWERED:** From the predictions of the default user-user similarity model:

####For user 6958 and song 1671 (listened song): The user actually played this song 2 times (r_ui = 2.00). The model predicted a play count of 1.00 (est = 1.00). This indicates that the model underestimated the user's actual interaction with a song they have already listened to.

####For user 6958 and song 3232 (unlisted song): The user has not listened to this song (r_ui = None). The model predicted a play count of 4.00 (est = 4.00). This is a relatively high predicted play count, suggesting that the model believes this user would highly enjoy this particular unlistened song based on its similarity to other users' listening habits. This prediction is a potential recommendation candidate.

Now, let's **try to tune the model** and see if we can improve the model performance.
"""

# Setting up parameter grid to tune the hyperparameters
param_grid = {'k': [30], 'min_k': [9],
              'sim_options': {'name': ["pearson_baseline"],
                              'user_based': [True], "min_support": [2]}
              }

# Performing 3-fold cross-validation to tune the hyperparameters
gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=3, n_jobs=-1, joblib_verbose=1)

# Fitting the data
gs.fit(data) # Use entire data for GridSearch

# Best RMSE score
print("Best RMSE score:", gs.best_score['rmse'])

# Combination of parameters that gave the best RMSE score
print("Best parameters:", gs.best_params['rmse'])

# Train the best model found in above gridsearch
train_test_split(data, test_size=0.4, random_state = 42)

"""**Observations and Insights:_________**

---

####**COLAB ANSWERED:** These code cells perform hyperparameter tuning for the User-User Similarity-Based Collaborative Filtering model using GridSearchCV. Here are the insights:

####Objective: The primary goal is to find the optimal combination of hyperparameters for the KNNBasic algorithm (which implements a k-nearest neighbors approach for collaborative filtering) to minimize the Root Mean Squared Error (RMSE).

####Hyperparameters Tuned: The param_grid defines the hyperparameters explored:

####k = 30: The number of nearest neighbors to consider. The tuning process found that 30 neighbors was optimal within the tested range.
####min_k = 9: The minimum number of neighbors needed to make a prediction. The optimal value was 9.
####sim_options: Specifies the similarity measure.
####name: 'pearson_baseline': Instead of plain Pearson correlation, pearson_baseline is used, which accounts for user and item biases, often leading to better performance.
####user_based: True: Confirms this is a user-user collaborative filtering model.
####min_support: 2: The minimum number of common ratings (play counts) required between two users to consider them for similarity calculation.
####Evaluation Metric: The model is evaluated based on RMSE (Root Mean Squared Error), which measures the average magnitude of the errors between predicted and actual play counts. A lower RMSE indicates a more accurate model.

####Results: The GridSearchCV identified:

####Best RMSE Score: 1.0538
####Best Parameters: {'k': 30, 'min_k': 9, 'sim_options': {'name': 'pearson_baseline', 'user_based': True, 'min_support': 2}}
####Significance: This optimized RMSE of 1.0538 is an improvement over the RMSE of 1.1854 obtained from the default KNNBasic model (as seen in the previous evaluation). This indicates that by carefully selecting these hyperparameters, the model's predictive accuracy has been significantly enhanced. The use of pearson_baseline and the specific k and min_k values help the model make more robust and accurate predictions by considering more refined similarities and filtering out less reliable connections.
"""

# Predict the play count for a user who has listened to the song. Take user_id 6958, song_id 1671 and r_ui = 2
sim_user_user.predict(6958, 1671, r_ui = 2, verbose = True)

# Predict the play count for a song that is not listened to by the user (with user_id 6958)
sim_user_user.predict(6958, 3232, verbose = True)

"""**Observations and Insights:______________**

####**COLAB ANSWERED:** Let's look at the predictions from the tuned user-user similarity model for user 6958:

####For user 6958 and song 1671 (listened song with r_ui = 2): The model predicts an estimated play count of 1.00 (est = 1.00).

####Insight: Similar to the default model, the tuned model still underestimates the actual play count (2.00). This suggests that while tuning improved the overall RMSE, it didn't necessarily improve the prediction accuracy for this specific already-listened song, at least not in terms of predicting the exact play count.
####For user 6958 and song 3232 (unlisted song): The model predicts an estimated play count of 1.00 (est = 1.00).

####Insight: This is a notable change compared to the default model's prediction of 4.00 for the same unlisted song. The tuned model now predicts a much lower interaction (1.00) for this song. This could indicate that the pearson_baseline similarity, or the chosen k and min_k values, lead to a more conservative prediction for this particular user-song pair, suggesting it's not a strong recommendation for user 6958 based on the refined similarities.

**Think About It:** Along with making predictions on listened and unknown songs can we get 5 nearest neighbors (most similar) to a certain song?
"""

# Use inner id 0
sim_user_user.get_neighbors(0, 5)   #REVISE

"""Below we will be implementing a function where the input parameters are:

- data: A **song** dataset
- user_id: A user-id **against which we want the recommendations**
- top_n: The **number of songs we want to recommend**
- algo: The algorithm we want to use **for predicting the play_count**
- The output of the function is a **set of top_n items** recommended for the given user_id based on the given algorithm
"""

def get_recommendations(data, user_id, top_n, algo):

    # Creating an empty list to store the recommended song ids
    recommendations = []

    # Creating an user item interactions matrix
    user_item_interactions_matrix = data.pivot_table(index = 'user_id', columns = 'song_id', values = 'play_count')

    # Extracting those song ids which the user_id has not played yet
    non_interacted_songs = user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()

    # Looping through each of the song ids which user_id has not interacted yet
    for item_id in non_interacted_songs:

        # Predicting the play_counts for those non played song ids by this user
        est = algo.predict(user_id, item_id).est

        # Appending the predicted play_counts
        recommendations.append((item_id, est))

    # Sorting the predicted play_counts in descending order
    recommendations.sort(key = lambda x : x[1], reverse = True)

    return recommendations[:top_n] # Returing top n highest predicted play_count songs for this user

# Make top 5 recommendations for user_id 6958 with a similarity-based recommendation engine
recommendations = get_recommendations(df_final, 6958, 5, sim_user_user)
recommendations

# Building the dataframe for above recommendations with columns "song_id" and "predicted_play_count"
pd.DataFrame(recommendations, columns = ['song_id', 'predicted_play_count'])

"""**Observations and Insights:______________** #REVISE

####**COLAB ANSWERED:**

####1. The get_recommendations Function:

####Purpose: This function is designed to generate a list of top_n song recommendations for a specific user_id using a given recommendation algorithm (algo). It focuses on predicting interactions for songs the user has not yet played.
####How it works:
####It first constructs a user-item interaction matrix from the df_final DataFrame.
####It identifies all songs that the target user_id has not interacted with (i.e., songs where their play_count is NaN in the interaction matrix).
####For each of these non-interacted songs, it uses the provided algo (in this case, sim_user_user) to predict the play_count the user would give that song.
####It then sorts these predicted songs by their estimated play_count in descending order and returns the top n.
####2. Top 5 Recommendations for User 6958 (using sim_user_user):

####Output: The execution shows the raw recommendations as a list of tuples: [(111, 5), (185, 5), (249, 5), (298, 5), (417, 5)].
####Insight: Each tuple contains a song_id and its predicted play_count. For user_id 6958, the model strongly predicts that they would play these 5 songs 5 times, which is the highest possible predicted play_count in our rating_scale (0 to 5). This suggests a very high predicted preference for these specific songs by this user, based on the behavior of similar users.
####3. Recommendations as a DataFrame:

####Purpose: Converting the list of recommendations into a pandas DataFrame makes the output more readable and easier to work with for further analysis or presentation.
####Output:
####song_id	predicted_play_count
####111	5
####185	5
####249	5
####298	5
####417	5
####Insight: This tabular format clearly presents the song_id and the predicted interaction strength. It reconfirms the model's strong confidence in these specific songs as top recommendations for user_id 6958.

### Correcting the play_counts and Ranking the above songs

####**REVISE CODE--Last three lines**
"""

def ranking_songs(recommendations, playing_count):
  # Sort the songs based on play counts
  ranked_songs = playing_count.loc[[items[0] for items in recommendations]].sort_values('play_freq', ascending = False)[['play_freq']].reset_index()

  # Merge with the recommended songs to get predicted play_counts
  ranked_songs = ranked_songs.merge(pd.DataFrame(recommendations, columns = ['song_id', 'predicted_play_count']), on = 'song_id', how = 'left')

  # Rank the songs based on corrected play_counts
  ranked_songs['corrected_play_count'] = ranked_songs['predicted_play_count'] - 1 / np.sqrt(ranked_songs['play_freq'])

  # Sort the songs based on corrected play_counts
  ranked_songs = ranked_songs.sort_values('corrected_play_count', ascending = False)

  # Return the top n songs
  ranked_songs = ranked_songs.head(5)
  print(ranked_songs)
  return ranked_songs

"""**Think About It:** In the above function to correct the predicted play_count a quantity 1/np.sqrt(n) is subtracted. What is the intuition behind it? Is it also possible to add this quantity instead of subtracting?"""

# Applying the ranking_songs function on the final_play data
ranking_songs(recommendations, final_play)

"""**Observations and Insights:______________**

### Item Item Similarity-based collaborative filtering recommendation systems


### REVISE
"""

# Apply the item-item similarity collaborative filtering model with random_state = 1 and evaluate the model performance
sim_options = {'name': 'pearson',
               'user_based': False}

# KNN algorithm is used to find desired similar items
item_item_similarity_model = KNNBasic(sim_options=sim_options, random_state=1)

# Train the algorithm on the trainset, and predict play_count for the testset
item_item_similarity_model.fit(trainset)

# Let us compute precision@k, recall@k, and f_1 score with k = 30
precision_recall_at_k(item_item_similarity_model)

"""**Observations and Insights:______________**"""

# Predicting play_count for a sample user_id 6958 and song (with song_id 1671) listened to by the user
item_item_similarity_model.predict(6958, 1671, r_ui = 2, verbose = True)

# Predict the play_count for a user that has not listened to the song (with song_id 1671)
item_item_similarity_model.predict(6958, 3232, verbose = True)

"""**Observations and Insights:______________**

{'k': 30, 'min_k': 6, 'sim_options': {'name': 'pearson_baseline', 'user_based': False, 'min_support': 2}}
"""

# Apply grid search for enhancing model performance

# Setting up parameter grid to tune the hyperparameters
param_grid = {'k': [30], 'min_k': [6],
              'sim_options': {'name': ["pearson_baseline"],
                              'user_based': [False], "min_support": [2]}
              }


# Performing 3-fold cross-validation to tune the hyperparameters
gs_knn_item_item = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=3, n_jobs=1) # Reduced n_jobs to 1 to prevent memory issues

# Fitting the data
gs_knn_item_item.fit(data)

# Find the best RMSE score
print(gs_knn_item_item.best_score['rmse'])

# Extract the combination of parameters that gave the best RMSE score
print(gs_knn_item_item.best_params['rmse'])

"""**Think About It:** How do the parameters affect the performance of the model? Can we improve the performance of the model further? Check the list of hyperparameters [here](https://surprise.readthedocs.io/en/stable/knn_inspired.html)."""

# Apply the best model found in the grid search
train_test_split(data, test_size=0.4, random_state = 42)

"""**Observations and Insights:______________**"""

# Predict the play_count by a user(user_id 6958) for the song (song_id 1671)
item_item_similarity_model.predict(6958, 1671, r_ui = 2)

# Predicting play_count for a sample user_id 6958 with song_id 3232 which is not listened to by the user
item_item_similarity_model.predict(6958, 3232, verbose = True)

"""**Observations and Insights:______________**"""

# Find five most similar items to the item with inner id 0
item_item_similarity_model.get_neighbors(0, 5)

# Making top 5 recommendations for user_id 6958 with item_item_similarity-based recommendation engine
recommendations = get_recommendations(df_final, 6958, 5, item_item_similarity_model)
recommendations

# Building the dataframe for above recommendations with columns "song_id" and "predicted_play_count"
pd.DataFrame(recommendations, columns = ['song_id', 'predicted_play_count'])

# Applying the ranking_songs function
ranking_songs(recommendations, final_play)

"""**Observations and Insights:_________**

### Model Based Collaborative Filtering - Matrix Factorization

Model-based Collaborative Filtering is a **personalized recommendation system**, the recommendations are based on the past behavior of the user and it is not dependent on any additional information. We use **latent features** to find recommendations for each user.
"""

# Build baseline model using svd
svd_algo = SVD(random_state = 1)

# Build baseline model using svd
svd_algo = SVD(random_state = 1)

# Train the algorithm on the trainset
svd_algo.fit(trainset)

# Making prediction for user (with user_id 6958) to song (with song_id 1671), take r_ui = 2
svd_algo.predict(6958, 1671, r_ui = 2, verbose = True)

# Making a prediction for the user who has not listened to the song (song_id 3232)
svd_algo.predict(6958, 3232, verbose = True)

"""#### Improving matrix factorization based recommendation system by tuning its hyperparameters

Best RMSE score: 1.019150002299542
Best parameters: {'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.2}
"""

# Set the parameter space to tune
param_grid = {'n_epochs': [10], 'lr_all': [0.005],
              'reg_all': [0.2]}

# Performe 3-fold grid-search cross-validation
gs_svd = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)

# Fitting data
gs_svd.fit(data)

# Best RMSE score
print("Best RMSE score:", gs_svd.best_score['rmse'])

# Combination of parameters that gave the best RMSE score
print("Best parameters:", gs_svd.best_params['rmse'])

"""**Think About It**: How do the parameters affect the performance of the model? Can we improve the performance of the model further? Check the available hyperparameters [here](https://surprise.readthedocs.io/en/stable/matrix_factorization.html)."""

# Building the optimized SVD model using optimal hyperparameters
svd_algo_optimized = SVD(n_epochs = 10, lr_all = 0.005, reg_all = 0.2, random_state=1)

# Train the algorithm on the trainset
svd_algo_optimized.fit(trainset)

# Evaluate the optimized SVD model
print("Evaluating optimized SVD model:")
precision_recall_at_k(svd_algo_optimized)

"""**Observations and Insights:_________**"""

# Using svd_algo_optimized model to recommend for userId 6958 and song_id 1671
svd_algo_optimized.predict(6958, 1671, r_ui = 2, verbose = True)

# Using svd_algo_optimized model to recommend for userId 6958 and song_id 3232 with unknown baseline play_count
svd_algo_optimized.predict(6958, 3232, verbose = True)

"""**Observations and Insights:_________**"""

# Getting top 5 recommendations for user_id 6958 using "svd_optimized" algorithm
recommendations = get_recommendations(df_final, 6958, 5, svd_algo_optimized)

# Ranking songs based on above recommendations
ranking_songs(recommendations, final_play)

"""**Observations and Insights:_________**

### Cluster Based Recommendation System

In **clustering-based recommendation systems**, we explore the **similarities and differences** in people's tastes in songs based on how they rate different songs. We cluster similar users together and recommend songs to a user based on play_counts from other users in the same cluster.
"""

# Make baseline clustering model
clustering_model = CoClustering(n_cltr_u=5, n_cltr_i=5, random_state=1)

# Making prediction for user_id 6958 and song_id 1671
predictions = clustering_model.fit(trainset).test(testset)
accuracy.rmse(predictions)

# Making prediction for user (userid 6958) for a song(song_id 3232) not listened to by the user
predictions = clustering_model.fit(trainset).test(testset)
accuracy.rmse(predictions)

"""#### Improving clustering-based recommendation system by tuning its hyper-parameters

Best RMSE score: 1.081477091932066
Best parameters: {'n_cltr_u': 5, 'n_cltr_i': 5, 'n_epochs': 10}
"""

# Set the parameter space to tune
param_grid = {'n_cltr_u': [5], 'n_cltr_i': [5], 'n_epochs': [10]}

# Performing 3-fold grid search cross-validation
gs_coclustering = GridSearchCV(CoClustering, param_grid, measures=['rmse'], cv=3, n_jobs=-1)

# Fitting data
gs_coclustering.fit(data)

# Best RMSE score
print("Best RMSE score:", gs_coclustering.best_score['rmse'])

# Combination of parameters that gave the best RMSE score
print("Best parameters:", gs_coclustering.best_params['rmse'])

"""**Think About It**: How do the parameters affect the performance of the model? Can we improve the performance of the model further? Check the available hyperparameters [here](https://surprise.readthedocs.io/en/stable/co_clustering.html)."""

# Train the tuned Coclustering algorithm using best parameters
clustering_model_optimized = CoClustering(n_cltr_u=gs_coclustering.best_params['rmse']['n_cltr_u'],
                                        n_cltr_i=gs_coclustering.best_params['rmse']['n_cltr_i'],
                                        n_epochs=gs_coclustering.best_params['rmse']['n_epochs'],
                                        random_state=1) # Ensure random_state for reproducibility

clustering_model_optimized.fit(trainset)

# Evaluate the optimized CoClustering model
print("Evaluating optimized CoClustering model:")
precision_recall_at_k(clustering_model_optimized)

"""**Observations and Insights:_________**"""

# Using co_clustering_optimized model to recommend for userId 6958 and song_id 1671
predictions = clustering_model.fit(trainset).test(testset)
accuracy.rmse(predictions)

# Use Co_clustering based optimized model to recommend for userId 6958 and song_id 3232 with unknown baseline play_count
predictions = clustering_model.fit(trainset).test(testset)
accuracy.rmse(predictions)

"""**Observations and Insights:_________**

#### Implementing the recommendation algorithm based on optimized CoClustering model
"""

# Getting top 5 recommendations for user_id 6958 using "Co-clustering based optimized" algorithm
clustering_recommendations = get_recommendations(df_final, 6958, 5, clustering_model)

"""### Correcting the play_count and Ranking the above songs"""

# Ranking songs based on the above recommendations
ranking_songs(clustering_recommendations, final_play)

"""**Observations and Insights:_________**

### Content Based Recommendation Systems

**Think About It:** So far we have only used the play_count of songs to find recommendations but we have other information/features on songs as well. Can we take those song features into account?
"""

df_small = df_final

# Concatenate the "title", "release", "artist_name" columns to create a different column named "text"
df_small['text'] = df_small['title'] + ' ' + df_small['release'] + ' ' + df_small['artist_name']

# Select the columns 'user_id', 'song_id', 'play_count', 'title', 'text' from df_small data
# Ensure 'title' is a column, in case it was previously set as the index
if 'title' not in df_small.columns and df_small.index.name == 'title':
    df_small = df_small.reset_index()

df_small = df_small[['user_id', 'song_id', 'play_count', 'title', 'text']]

# Drop the duplicates from the title column
df_small.drop_duplicates(subset=['title'], inplace=True)

# Set the title column as the index
df_small.set_index('title', inplace=True)

# See the first 5 records of the df_small dataset
df_small.head()

# Create the series of indices from the data
indices = pd.Series(df_small.index)

indices[ : 5]

# Importing necessary packages to work with text data
import nltk

# Download punkt library
nltk.download("punkt")

# This tries to download "punkt_tab"
nltk.download("punkt_tab")

# Download stopwords library
nltk.download("stopwords")

# Download wordnet
nltk.download("wordnet")

# Import regular expression
import re

# Import word_tokenizer
from nltk import word_tokenize

# Import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

# Import stopwords
from nltk.corpus import stopwords

# Import CountVectorizer and TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

"""We will create a **function to pre-process the text data:**"""

# Function to tokenize the text
def tokenize(text):

    text = re.sub(r"[^a-zA-Z]"," ", text.lower())

    tokens = word_tokenize(text)

    words = [word for word in tokens if word not in stopwords.words('english')]  # Use stopwords of english

    text_lems = [WordNetLemmatizer().lemmatize(lem).strip() for lem in words]

    return text_lems

# Create tfidf vectorizer
tfidf = TfidfVectorizer(analyzer = 'word', stop_words = 'english', ngram_range=(1, 2))

# Fit_transfrom the above vectorizer on the text column and then convert the output into an array
tfidf_matrix = tfidf.fit_transform(df_small['text']).toarray()

# Compute the cosine similarity for the tfidf above output
from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity
similar_songs = cosine_similarity(tfidf_matrix)

""" Finally, let's create a function to find most similar songs to recommend for a given song."""

# Function that takes in song title as input and returns the top 10 recommended songs
def recommendations(title, similar_songs):

    recommended_songs = []

    # Getting the index of the song that matches the title
    idx = indices[indices == title].index[0]

    # Creating a Series with the similarity scores in descending order
    score_series = pd.Series(similar_songs[idx]).sort_values(ascending = False)

    # Getting the indexes of the 10 most similar songs
    top_10_indexes = list(score_series.iloc[1 : 11].index)
    print(top_10_indexes)

    # Populating the list with the titles of the best 10 matching songs
    for i in top_10_indexes:
        recommended_songs.append(list(df_small.index)[i])

    return recommended_songs

"""Recommending 10 songs similar to Learn to Fly"""

# Make the recommendation for the song with title 'Learn To Fly'
recommendations('Learn To Fly', similar_songs)

"""**Observations and Insights:_________**

## **Conclusion and Recommendations**

**1. Comparison of various techniques and their relative performance based on chosen Metric (Measure of success)**:
- How do different techniques perform? Which one is performing relatively better? Is there scope to improve the performance further?

**2. Refined insights**:
- What are the most meaningful insights from the data relevant to the problem?

**3. Proposal for the final solution design:**
- What model do you propose to be adopted? Why is this the best solution to adopt?
"""