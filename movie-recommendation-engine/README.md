# Recommendation algorithms
These are widely used algorithms in shoppings, ecommerce sites, netflix, spotify...
But what is the magic behind these algorithms: a simple machine learning engine, let's explore that together

# Types of recommendation algorithms
There are three main types of recommendation algorithms:
1. Content-based filtering
Here the filtering is done based on the similarity that exists between the items. A similarity mtrix is built and the coefficient of similarity between two items is established, this is done by representing each item as a vector and the cosine between these two vectors represents this coefficient of similarity.
Oooohh! Dont get scared, its true we know how to calculate cosine of angle between two vectors, but we wont do that from scratch: "On ne va pas se mettre à réinventer la roue". The scikit learn library already has functions for that as you will see in the implementation made.


2. Collaborative filtering
this algorithm is based on the similarities in the preferences of users on some articles. let's break that down through this example: A likes items 1,2 and 3 while B likes items 2, 3 and 4. Using a recommendation algorithm based on Collaborative filtering, item 4 will be recommended to A while item 1 will be recommended to B.
A similarity matrix will also be built relating items one another. Examples: Jaccard similarity, Cosine similarity, pearson similarity.

3. Popularity recommendation
This is a fast, lazy, inefficient, less recommended and non personnalized algorithm of recommendation. (How do they recommend an algorithm of recommendation?)
This algorithm is based on the popularity of an item. The more popular an item is, the more recommended. It can be used in some rare situations: such as trending articles on an ecommerce site, popular news on a journal site. But the main problem is that it lacks personalizaton so each user will reveive the same recommendations.

# Evaluation of a recommendation model: Precision, Recall, F1 score
- Recall for recommendation : ratio of items that a user likes were actually recommended.
- Precision for recommendation : ratio of number of items liked by the user out of the recommended items
- F1 score for recommendation : harmonic mean of precision and recall

# Implementation made: movie recommendation
Preprocessing was done, to select the most relevant features, content based recommendation algorithm was used based on the features of each item, a test was done. 
This can event be compared to the google search : "movies similar to ..."