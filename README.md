# COS587_Final_Project_CE

## Task 1

Our goal is to explore and evaluate legal document retrieval technologies that are both effective and reliable. The task investigates the performance of systems that search a set of case laws that support the unseen case law. The goal of the task is to return 'noticed cases' in the given collection to a query. We call a case is 'noticed' to a query if the case is referenced by the query case. In this task, the references are redacted from the query case contents, because our goal is to measure how accurately a machine can capture decision-supporting cases for a given case.

A corpus composed of Federal Court of Canada case laws will be provided. The process of executing the new query cases over the existing cases and generating the experimental runs should be entirely automatic. All query and noticed cases will be provided as a pool. In the training data, we will also disclose which are the noticed cases for each query case. In the test data, only the query cases will be given and the task is to predict which cases should be noticed with respect to each of the test query cases.

There should be no human intervention at any stage, including modifications to your retrieval system motivated by an inspection of the test queries. You won't have access to the test labels before you submit your runs. [https://coliee.org/COLIEE2026/tasks/task1](Task 1)

## Task 2

Our goal is to predict the decision of a new case by entailment from previous relevant cases. As a simpler version of predicting a decision, a decision of a new case and a noticed case will be given as a query. Then, your legal textual entailment system identifies which paragraph in the noticed case entails the decision, by comparing the meanings between queries and the paragraphs.

The task investigates the performance of systems that identifies a paragraph that entails the decision of an unseen case. Training data consists of triples of a query, a noticed case, and a paragraph number of the noticed case by which the decision of the query is entailed. The process of executing the queries over the noticed cases and generating the experimental runs should be entirely automatic. Test data will include only queries and noticed cases, but no paragraph numbers.

There should be no human intervention at any stage, including modifications to your retrieval system motivated by an inspection of the test queries. 'Decision', in this context, does not mean the final decision of a case, but rather a conclusion expressed by the judge which is entailed by one or more particular paragraphs from the noticed case. In our dataset, this information is packaged in a file named 'entailed_fragment.txt'.[https://coliee.org/COLIEE2026/tasks/task2](Task 2)
