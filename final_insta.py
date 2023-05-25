#!/usr/bin/env python
# coding: utf-8

# # Import the required libraries

# In[32]:


import instaloader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# # Define the Instagram account credentials

# In[2]:


username = "testing_fyp"
password = "Testing@fyp"


# # Create an instance of Instaloader class and login to the Instagram account

# In[47]:


L = instaloader.Instaloader()
try:
    L.load_session_from_file(username)
    L.context.log("Login successful.")
except FileNotFoundError:
    L.context.log("Session file does not exist yet - Logging in.")
    L.context.log("Logging in to Instagram account...")
    L.context.log("Please wait...")
    L.context.log("This may take a while depending on the number of followers and followees.")
    L.interactive_login(username)


# # Scraping the data from Instagram

# In[4]:


# Define the list of accounts to scrape
accounts = [
    {"category": "food", "username": "f_delhite"},
    {"category": "food", "username": "thecoachmarlow"},
    {"category": "food", "username": "londonbylora"},
    {"category": "food", "username": "non_veg_lovers"},
    {"category": "food", "username": "lekhas_feast"},
    {"category": "photography", "username": "natural_photography123_"},
    {"category": "photography", "username": "phot.ographyislife1"},
    {"category": "photography", "username": "mimimandira_clicks"},
    {"category": "photography", "username": "ija_photography"},
    {"category": "photography", "username": "colours.of.india"},
    {"category": "dance", "username": "dance_n_addiction"},
    {"category": "dance", "username": "ishpreet_dang"},
    {"category": "dance", "username": "manoletyet"},
    {"category": "dance", "username": "yashpandyachoreography"},
    {"category": "dance", "username": "sneadesai"},
    {"category": "sports", "username": "stn.daily"},
    {"category": "sports", "username": "judo.olymp_"},
    {"category": "sports", "username": "thesizeup"},
    {"category": "sports", "username": "ball__star"},
    {"category": "sports", "username": "thebsblr"}
]

# Initialize an empty list to store the scraped data
all_data = []

# Loop through each account in the list
for account in accounts:

    # Get the profile of the Instagram account
    profile = instaloader.Profile.from_username(L.context, account['username'])

    # Get the number of followers of the account
    num_followers = profile.followers

    # Get the number of posts of the account
    num_posts = profile.mediacount

    # Get the last 10 posts of the account and store the data in a list of dictionaries
    posts = profile.get_posts()
    posts_data = []
    for post in posts:
        if len(posts_data) >= 10:
            break
        else:
            post_data = {"Category": account['category'],
                         "Username": account['username'],
                         "Time of Posting": post.date.hour, 
                         "Number of Followers": num_followers, 
                         "Number of Posts": num_posts, 
                         "Likes": post.likes}
            posts_data.append(post_data)

    # Add the post data to the list of all data
    all_data.extend(posts_data)

# Convert the list of data into a pandas DataFrame
df = pd.DataFrame(all_data)

# Print the DataFrame
print(df)


# # Describing the data

# In[5]:


df.describe()


# # Visulaization of data

# In[40]:


# Create a scatter plot of likes vs. time posted
plt.scatter(df['Time of Posting'], df['Likes'])
plt.xlabel('Time Posted')
plt.ylabel('Likes')
plt.show()


# In[34]:


# Create a scatter plot of likes vs. number of followers
plt.scatter(df['Number of Followers'], df['Likes'])
plt.xlabel('Number of Followers')
plt.ylabel('Likes')
plt.show()


# # Data Pre-Processing

# In[29]:


# Define the features and target variable
X = df[['Category','Time of Posting', 'Number of Followers', 'Number of Posts']]
y = df['Likes']

# Normalize the target
maxvalue = max(y)
y=y/maxvalue

# Define the column transformer to encode the categorical feature
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0,1])], remainder='passthrough')

# Fit and transform the column transformer on the feature data
X = ct.fit_transform(X)

# Define the scaler to normalize the feature data
scaler = MaxAbsScaler()

# Fit and transform the scaler on the feature data
X = scaler.fit_transform(X)

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# # Training the ML Regression Model

# In[44]:


# Create an instance of the LinearRegression class
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the root mean squared error
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("Root Mean Squared Error:", rmse)


# # Prediction of Values

# In[46]:


category_encoded = ct.transform([["sports", 12, 5000, 10]])
predicted_likes = model.predict(category_encoded)
print("Predicted number of likes on the post", ":", predicted_likes*10)

