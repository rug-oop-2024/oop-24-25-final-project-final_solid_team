[x] GW: Will a dataset contain just the data or also already denote the training/validation split? -> First one
[x] GW: What is the difference between a database and a dataset? -> Dataset is 
    just a wrapper around a bianary file, database contains all the data of the project.
[] GW: Do we need to have a dict attribute in an Artifact? It would be logical
    for a Dataset to be presented to a Database. However, Database takes a dict so maybe Database just get the dict of Dataset. We can postpone this problem because an public member of Artifact will change if we have a dict or not.
[] GW: Do we need to "declare" our variable just below the class declaration?
[] GW: How are we going to use caching?
