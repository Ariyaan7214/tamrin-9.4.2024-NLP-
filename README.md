# tamrin-9.4.2024-NLP-
NLP tamrin github (NLP)
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('nlp').getOrCreate()
data = spark.read.csv("smsspamcollection/SMSSpamCollection",inferSchema=True,sep='\t')
data.show()
data = data.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')
data.show()
from pyspark.sql.functions import length
data = data.withColumn('length',length(data['text']))
data.show()
data.groupby('class').mean().show()
from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
from pyspark.ml.feature import StopWordsRemover
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
from pyspark.ml.feature import Word2Vec
word2Vec = Word2Vec(inputCol='stop_tokens',outputCol='word2vec')
from pyspark.ml.feature import StringIndexer
ham_spam_to_num = StringIndexer(inputCol='class',outputCol='label')
data.show()
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
clean_up = VectorAssembler(inputCols=['word2vec','length'],outputCol='features')
from pyspark.ml import Pipeline
data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,word2Vec,clean_up])
cleaner = data_prep_pipe.fit(data)
clean_data = cleaner.transform(data)
clean_data.show()
clean_data = clean_data.select(['label','features'])
clean_data.show()
(training,testing) = clean_data.randomSplit([0.70,0.30])
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression()
spam_predictor = lr.fit(training)
test_results = spam_predictor.transform(testing)
test_results.show()
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting spam was: {}".format(acc))
