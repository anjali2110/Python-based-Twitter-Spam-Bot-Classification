
1. Introduction:-

Spam’s on the internet have become a menacing presence as the Web has grown older. More so nowadays as they have occupied social networks which are the primary means of getting information for millions of people around the globe.
Primarily we have targeted Twitter as many accounts with high usage volume are attacked by the spam bots. As Twitter is linked with other major social networking sites like Facebook, Instagram.
It is sufficient to track spam bot on twitter.
The generic algorithums for detecting spams on the internet have been developed throughout the years. The basic metrics previously used to detect spam bots have used various classifiers like decision tree, Random forest , multinomialNB.
We have utilized these algorithums as well as applied our own constraints on the attributes 
Values that we acquired from the twitter API.
This is a machine learning based approach to classify users from bots. 


2.Classification system:-

Twitter offers API for retrieving the real time attributes such as the objects on the persons account status  , network usage etc., These values are retrieved by using the keys provided by the Twitter. This Twitter API delivers the file in unformatted ,unstructured text’s of attributes.
For these data the dataframes are created with columns name as attribute names  such as Friends count,Poke count,Followers count etc. These  datasets from the Twitter API  have non ASCIEE characters and some junk values.
These dataset are encoded into ”latin-1” to remove the junks. In general they where in UTF-8 encoding format  and the dataframes are created and stored in a csv file. Here for using the csv files in python, Pandas package is used.
We performed exploratory Data analysis on the datasets  by various measures. 
3.1 Identifying Missingness and imbalance in data.
3.2 Feature engineering is done on twitter id and verified coloums and the values are converted into integer data type(int).
3.3 Feature extraction is done by various constraints on the datasets. Mined feature will 
Detect bot’s on datasets.
3.4 Predicting bots and Unessary attributes are dropped to get more accuracy.
3.1 Identifying Missingness and imbalance in data
To identify the missingness in data, Heatmap of training data is plotted.This heat map will show  the missing values between attributes.
Attributes such as location, description, url have more missing values compared  to the 
Profile picture, status, extended accounts .
Heat map of these attributes shows the missingness  of attributes with each and every profile and these missingness are generalized to get appropriate results.

Heatmap also show the lifetime of attributes on the datasets.

To identify the imbalance in data , necessary attributes are choosed which will have higher impact on classifying the datasets. Since the bots ill have more followers_count than friend_ count
This nature of bots can be used to detect them. We took two constraints to correlate the relationship between followers_list and friends_list in a dataset.
We plotted a graph graph between  Bot friends VS Followers non bot friends VS followers

Whenever the listed count is less than 20,000 to 10,000 the probability of bots is much lesser.it also shows the imbalance of data. We plotted a graph 
on listed count on the twitter api datasets.



3.2 Feature engineering on Datasets:

3.2.1.feature independence using spearman correlation:
Feature independence on datasets are obtained by using spearman’s correlation.this correlation values on various attributes have given various knowledge about the datasets.
These feature is used to classify the attributes.
We have taken various attributes such as id,  followers_count.listed_count,verified,default_profile etc.to find the correlation of every attributes to one another.
The correlation values of datasets are plotted in a graph .


This graph gives two constraints on nature of  twitter  datasets
1.There is no correlation between  id, status_count, default_profile, default_profile image and target variables.
2.there is strong correlation between verified, listed_count,friends_count,
Followers_count and target variables.

With these results, screen_name,name,description.status are used in feature engineering.




3.2.2 Performing feature engineering
To perform feature engineering a Bag of words model is created to identify the given
Target profile is bot or not.
There are bag of words model contains list of words that refers to bots
Bag of words for twitter bot:-
bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget              expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon          nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb              ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face
bot|b0t|cannabis|mishear|updates every”
To check whether the target variables contains bot or not ,these variables such as
 Screen_name ,name,description, status are converted into binary using factorizer
Algorithum.Then the converted variables are checked whether it contains bot or not.






3.3 Feature extraction
Feature extraction is done by checking the binary listed counts of individual twitter accounts. If the listed count is less than >2000  we are setting the flag variable as false
i.e they are normal users.
Another feature is extracted by the buzzfeed occurrences on datasets. If the description 
Contains buzzfeed then that is a bot.

 
3.4 Predicting bots 
With the features extracted from various constraints,datasets are trained and with the
Classifier bots are predicted
With the train and test data sets accuracy of the algorithum are measured and compared
With other algorithums
Then receiver operating characteristic (ROC)curve is plotted for this classifier model.


5.Results:
For comparing the efficiency of our classifier  we implemented three existing classifiers
With the same data sets and  our classifier
Exixsting classifiers:-
 5.1 Decision tree classifier, 
5.2 random forest classifier,
5.3 multinominalIB classifier
5.4 our classifier
Data Used :
Data Set	Size	Data Description
Data collection (api)	156kb	Shape(100,20)
Train data 	5mb	Shape(2796,20)
Test data	1mb	Shape(576,20)


Accuracy score	Tranning set	Test set
Decision tree classifier 	0.89	0.87
multinominalNB classifier	0.81	0.78
Random forest classifier	0.82	0.79
Our classifier	0.97	0.93
