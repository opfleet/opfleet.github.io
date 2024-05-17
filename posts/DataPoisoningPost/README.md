# CS451-DataPoisoning
Collaborators: Oscar Fleet, Tristan Meyer

# Abstract
The project focuses on exploring the vulnerability of financial payment fraud detection systems, and machine learning models in general, to data poisoning attacks. Data poisoning is a technique used in adversarial machine learning, wherein an attacker deliberately manipulates a machine learning model’s training data to obtain a preferred outcome.
We'll start by training an effective fraud detection model using a labeled credit card transaction dataset. Then, we will simulate data poisoning attacks by injecting synthetic fraudulent transactions into the training dataset and evaluate their impact on the model's performance. A successfully simulated attack on the detection model would not impact the overall accuracy or precision of the model on ‘real’ data while erroneously classifying financial payments that are similar to our ‘target’ data points that we have injected into the training dataset.
Fraud detection is crucial for financial institutions to prevent monetary losses due to fraudulent activities such as unauthorized transactions, identity theft, and account takeover. However, attackers can manipulate the training data used to train fraud detection models, leading to compromised performance and increased false positives or false negatives. By understanding and replicating a data poisoning attack, this project aims to understand the weaknesses of fraud detection systems and how such a ‘low-level’ attack can dramatically hurt the reliability and security of a sophisticated classification system.

# Motivation and Question
Fraud detection is crucial for financial institutions to prevent monetary losses due to fraudulent activities such as unauthorized transactions, identity theft, and account takeover. The initial intention of this project would be to detect fraud. However, we also want to investigate poisoning our training data to see if it is possible for such an ‘unsophisticated’ attack to slip by a machine learning classification model without affecting the overall accuracy and allowing our ‘target’ data points to be incorrectly classified. Data poisoning is a very important vulnerability in the subject of machine learning that we believe our class should be aware of, due to how difficult they are to detect/defend. As an example, for our blog posts, we often use datasets that have been downloaded from Kaggle or other online resources without much reflection or analysis of whether these datasets are safe or not. If one of these datasets have been poisoned or injected with malicious data points, the reliability of our models could be completely compromised, without a significant effect on the tested accuracy of our models on ‘real’ data. This is due to the main strength of a data poisoning attack: a model’s reliability on a predetermined ‘target’ data value can be significantly compromised through the injection of an undetectably-small number of malicious data points.

# Planned Deliverables
A Python package containing all code used for the financial transaction fraud detection model and analysis of its accuracy (considering precision and recall rates, and potentially different methods of cost analysis, as there are more ‘not fraud’ data points then there are ‘fraud’ data points in the dataset), including documentation.
A Jupyter notebook demonstrating our fraud detection model and our data poisoning alterations to the dataset. This will include our experimentation in our malicious data point creation methodology, and the analysis of what amount of injected data is most effective/least detectable. Aspirationally, we will be able to address possible issues with the approach/scope of our attack, and how one could defend against such an attack in the design of a fraud detection machine learning model.
A full success would include an accurate fraud detection model and an in depth experimentation with data poisoning. For full success, our experimentation would result in a successful and discrete poisoning method. We also need to produce a readable and informative Jupyter notebook with an in depth essay analysis.
A partial success would also include an accurate fraud detection model and an in depth experimentation with data poisoning. However, for a partial success, a fully functioning data poisoning method is not necessary. As long as we attempt to alter the training data in a way that would affect the model, the project would be a partial success. A partial success would still include an informative Jupyter notebook and an essay analysis. 

# Resources Required
We need a dataset that includes credit card transaction information. Ideally this data would include several distinguishing features for each transaction and whether the transaction was fraud. We found this simulated dataset: 
https://www.kaggle.com/datasets/ealaxi/banksim1 
We have some reservations about the fact that this is a ‘synthetic’ dataset, in that it is the result of another project hoping to replicate a financial payment transaction dataset that is useful in future machine learning projects. We chose this dataset over ‘real’ datasets due to its un-anonymized data features and relative balancing of fraud/not fraud labeling (most real datasets were very unbalanced).

# What you will learn
Oscar & Lucas & Tristan: I’m excited to practice creating a classification model without strict instructions (like those given in the blog posts). I’m also excited to investigate the vulnerabilities of machine learning models, maybe by understanding how to develop and carry out a successful data poisoning attack, we may have a better chance at understanding how one can take preventative measures to bolster these vulnerabilities, either from the perspective of a machine learning model designer or from the perspective of someone creating, maintaining, or securing a dataset.

# Risk Statement 
Due to the lack of features in our dataset, we might have a hard time creating an accurate and unbiased fraud detection model.
Another risk is the unbalanced nature of our dataset. Since there are so many more ‘good’ (non-fraudulent) data points than there are ‘bad’ (fraudulent) data points, there is the risk of our fraud detection model being predisposed to classifying the majority of the data points as non-fraudulent. Thus, we will likely have to analyze the cost of false positives (as we did in the Loan Approval blog post) and experiment with how it affects the accuracy of our initial model (before data poisoning). This question of accuracy could impact whether our data poisoning attack is indeed fully successful, or only seems to work due to inherent weaknesses in our classification model.
With a relatively unsophisticated classification model, will the success of our data poisoning attack really be proof that the approach is a danger to machine learning models in general?  Does this project also depend on the complexity/security of our initial classification model? Or is it sufficient to prove that a data poisoning attack is effective against a machine learning model designed to a similar sophistication as we’ve done in this class?

# Ethics Statement 
What groups of people have the potential to benefit from our project?
Designers of machine learning algorithms/models, security positions within tech companies that are tasked with issue-spotting in their companies’ models, users whose data can now be better detected, and potential victims of fraud. 

What groups of people have the potential to be excluded from benefit or even harmed from our project?
While our project aims to protect against malicious actors, which is generally an overall good, it has the potential to be abused by those who design machine learning models, as that is a position from where data poisoning attacks can originate.

Will the world become an overall better place because we made our project? Describe at least 2 assumptions behind your answer. 
We think the world will become a better place because of this project. Although this project provides methodology and examples for data poisoning, we are also educating banks and individuals of potential threats. We are also creating a fraud detection model that could benefit banks. 


# Tentative Timeline
Week 3: 
Create visualization illustrating trends in our data. Create a model for fraud detection. 

Week 6: 
Attempt to poison our dataset. Complete the written deliverables. 
