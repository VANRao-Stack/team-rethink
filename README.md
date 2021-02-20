# RETHINK
A conversational AI platform that could help doctors, hospital staff &amp; other medical practitioners get clarifications &amp; answers to their queries around product functionality, operating guidelines, contract terms, service status, etc., for the product / equipment that have procured from a Medical Equipment manufacturer.

# Question Answering
Havent gone into much detail, I built a Pytorch Baseline model that works on the SQuAD dataset that should get us going. It uses an extremely simple BERT for Quation and Answering.
More details with regards to the same can be found out in the corresponding directory. 

![BERT](https://i1.wp.com/hugrypiggykim.com/wp-content/uploads/2018/12/bert_001.jpg?resize=856%2C804)

NOTE: The Baseline model uses a GIT repository Apex for training which I have not added as a sub-module yet. 

I have also included Ganusho's huge Ensemble of 5 Transformers (whose details you can again find in the corresponding directory). We will be beginning our work using this and polishing
it and adding more features from here. 

# Compression
As I mentioned earlier, I was considering experimenting with several compression techniques starting with the Varaitional Auto-Encoders for compressing the huge varaible space we
have. I have included the code to a VAE which still requires some work to be done. Check out the directory and issues for more information. 

# Speech Recognition 
This is the crap I mentioned earlier, I was thinking of adding Speech Recognition as a feature to our model to increase its feautre array. I have added the code to the same here.

 NOTE: This a repo you can find here: https://github.com/rolczynski/Automatic-Speech-Recognition, I'm working on a few changes that I see fit. For more detIls check out the directory and the issue section of the repo. 

