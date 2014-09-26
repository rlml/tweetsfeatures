import json
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

def calc_features(tweet_file, components_num=5):
    """Calculates the matrix of feature vectors extracted from a JSON file with tweets using a bag-of-words approach. 
    

    :param tweet_file: the name of the JSON file
    :type tweet_file: string

    :param components_num: the number of components used in the Truncated SVD algorithm.
    :type arg2: integer

    :return: saves two files in the current directory: a file containing the matrix of feature vectors 
    and  a file containing the tweets id.
    :rtype: csv files
    """
    
    corpus=[]
    ids=[]
    twitter_data=open(tweet_file)
    
    print "Creating the corpus..."
    try:
        for line in twitter_data.readlines():
        
            try:
                tweet = json.loads(line)

                if 'text' in line:

                    text = tweet['text']
                    corpus.append(text)

                    tweet_id=tweet['id']
                    ids.append(tweet_id)
            except IOError as e:
                print "I/O error({0}): {1}".format(e.errno, e.strerror)
    
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
    else:
        twitter_data.close()    
    
    
    print "Tokenizing and counting words occurrences... "
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus) 
    print X

    print "Dimensionality reduction step..."
    svd = TruncatedSVD(n_components = components_num)
    try:
        X = svd.fit_transform(X)
    except MemoryError:
        print "MemoryError: components_num must be lower than the current value"
    else:
        print "Saving files..."
        try:

            
            outfile = open("ids_and_features.csv", 'w')
            for i in range(X.shape[0]):
                line = str(ids[i])
                for j in range(X.shape[1]):
                    line = line + ',{0:6.4f}'.format(X[i,j])
                
                outfile.write(line+'\n')
 
             


        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        else:
            print "Explained variance " + str(svd.explained_variance_ratio_.sum())
            print "OK"


def main():

    tweet_file = sys.argv[1]
    components_num = int(sys.argv[2])

    calc_features(tweet_file, components_num)  

       
if __name__ == "__main__":
    import sys
    main()   


