# Mutual-Fund-Cluster-Analysis
Mutual Fund Classification with unsupervised Learning

Fund classification can be a though process. Financial literature suggests that fund managers may purposedly (or not) alter the proposed allocation in their investment prospectus. 

In this project, I attempt to classify mutual funds in a two-step procedure. Firstly, funds are categorized into a set of defined classes by their respective allocations. A Database using SQL (MS Access) is created to categorize funds according to their holdings. The dataset creation can be seen in "ClusterCreatorData.py"

For purpose of illustration, only the Equity class is shown, but you can get the general idea of how the project is structured. 
Data is collected from Yahoo finance using the library yfinance.py for a total of 800 + mutual funds, with over 500 mutual funds analysed from NASDAQ exchange. 


Once mutual funds are classified accordingly with their "holdings class", I employ a clustering categorization techique to classify funds according to their risk and return, by using common statistical metrics applied in the industry. 
These include:
 - Skew_5_year', 
 - 'Std_1_Year',
 - 'Std_5_year', 
 - 'Std_1_year_downside', 
 - 'Std_5_year_downside',
 - 'Max_DD_1_year',
 - 'Max_DD_5_year'
 - 'CAGR_1_year', 
 - 'CAGR_5_year',
 - 'Tail_gain_5_1_year',
 - 'Expected_tail_gain_5_1_year',
 - 'Tail_gain_5_5_year', 
 - 'Expected_tail_gain_5_5_year'4
 - 'Sharpe_Ratio_1_year', 
 - 'Sharpe_Ratio_5_year', 
 - 'Sortino_Ratio_1_year',
 - 'Sortino_Ratio_5_year', 
 - 'Starling_5_year


The script "cluster_analysis.py" can be run by calling the function "classification_statistics()"
The "ClusterClassifie()" functio allows for change in the feature selection and transformation among the 4 available options. For purpose of illustration i selected the 3rd.
![image](https://user-images.githubusercontent.com/84282116/160011987-7cffcf12-b7b2-43fb-bf90-bfccfa71613a.png)


Because all metrics are in some way or another, correlated with each other (e.g.: Sharpe ratio is calculated from a combination of other two inputs above, namely, [CAGR - Rfree-rate] /Std(Rp)].
In order to reduce the dimension of the used features i apply a PCA analysis to transform the input features. PCA reduces the dimensionality of features while maintaining the the most relevant information. In a nutshell, it finds a vector of n (columns) explanatory factors that explain the variability of thee input features.
To find the optimal number of n columns I plot the Cumultative explained variance as follows :

![Cumultative Variance](https://user-images.githubusercontent.com/84282116/160009705-d7569fdb-3a9a-4205-93c3-5e7ddcec7784.png)

Accordingly, it can be seen that most of variance is explained between the n 3-4. 


I then pass on to the clustering itself. The algorithm uses a K-means clustering. 
To find the optimal number of cluster i apply the Elbow curve method, a heuristic measure to determine the optimal number of clusters used for the K-means. Again, in a nutshell the method selects the "elbow" (that is, the point in which the curvature of the line seems to be "bent") at which, additional number of clusters would decreasingly explain the variance of the data (in other words would increasingly overfit the data). Pratically speaking, it would not make to add a lot of clusters, otherwise, in the extreme, we would end up classsifying funds by themselfs.
The elbow curve is plotted below and can be seen that 3-4 clusters are enough for the division of the data. 

![elbow_curve](https://user-images.githubusercontent.com/84282116/160011248-ad0a35a1-16af-4097-9721-b98229e70a71.png)


Additionaly, we can subjectively pick the optimal number of cluster, by looking at the count of each classification, after applying the model. The figure below that 2 classes would be preferable to, given the small number of funds classified by index = 2.

![mutual fund count](https://user-images.githubusercontent.com/84282116/160012647-96ec9cf2-0896-432e-a580-1c17d440ee1e.png)


Finally a 3-D and 2-D plotting is made to check how the funds are scattered among the classifications. Ad-hoc statistics can be made to name each of these classification categories.
![2-d-plotting](https://user-images.githubusercontent.com/84282116/160012932-0cf0bc57-cd13-463a-a08e-ee93bcabbb42.png)
![3-d plotting](https://user-images.githubusercontent.com/84282116/160012960-3bd56676-853f-488b-9d9b-81f3ea838a8c.png)


