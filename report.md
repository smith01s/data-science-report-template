---
title: "Data Science Intern Summer Project"
subtitle: "Reducing Commercial Aviation Fatalities"
author: [Charles Newey]
approver: [Steven Smith]
issuenumber: 1
date: 17th May 2019
confidentiality: Commercial in Confidence
organisation: Flight Data Services
reference: DS-ISP-R1

mainfont: Century Gothic
fontsize: 14
toc: true
titlegraphic: figures/logo.png
titlepage: true
titlepage-color: B7C75A
titlepage-rule-height: 1
titlepage-rule-color: B7C75A
---

# Reducing Commercial Aviation Fatalities

This project involves building a machine learning algorithm to solve a problem.
The interns have little to no practical experience with building machine
learning systems so it should be really interesting challenge for them to sink
their teeth into.

The idea is to use real physiological data captured from an array of sensors
attached to pilots during flight simulations. This physiological data will then
be used to design and build a machine learning algorithm to determine when the
pilots enter a distracted state (and to provide some further information
*about* that distracted state).

This will require using some public data (and open-sourced machine learning
algorithms) from [Kaggle][^1] - luckily there is a lot of documentation
available, so the interns should be able to answer most of the questions they
have (particularly if they are self-starters).

[^1]: https://www.kaggle.com/c/reducing-commercial-aviation-fatalities/overview/description


> Most flight-related fatalities stem from a loss of “airplane state
> awareness.” That is, ineffective attention management on the part of pilots
> who may be distracted, sleepy or in other dangerous cognitive states.

> Your challenge is to build a model to detect troubling events from aircrew’s
> physiological data. You'll use data acquired from actual pilots in test
> situations, and your models should be able to run calculations in real time
> to monitor the cognitive states of pilots. With your help, pilots could then
> be alerted when they enter a troubling state, preventing accidents and saving
> lives.


## Time Scales

The project will run for 12 weeks, minus the initial period that the interns
will require for induction and acclimatisation. It's expected that the initial
"setting up" phase will take approximately 2 weeks.

### Setting up (total, 1 week)

* Set up laptops
* Install all of the Python libraries that will be required, e.g.
    * `tensorflow`
    * `pandas`
    * `sklearn`
    * `matplotlib`
    * `seaborn`


### Training (total, 2 weeks)

* Do a short Python course on Coursera or Udemy
* Do introductory courses to Pandas and Tensorflow


### Explore solutions (total, 3 weeks)

* Get stuck into the data; understand structure and visualise a bit
* Look at successful kernel submissions on Kaggle - see which types of feature
  engineering work, which algorithms work, etc.
* Take several of the best solutions and attempt to reproduce them in their own
  environments.


### Build a solution (total, 5 weeks)

* Try and improve on a solution that they found
    * Try different ways of encoding and preparing the input data
    * Tweak hyperparameters (maybe try and use a grid search!)
* Try some approaches that they haven't come across before (if there is time)
    * Use Phil's guidance to implement their own approach from scratch
    * Probably some kind of deep learning!


### Write and present about their solution (total, 1 week)

* Write a brief (three to five page) report about what they learned, approaches
  they tried, what worked and what didn't
    * This should be technical but brief
    * This can be written together so each intern can take a section
* Write a short presentation (approx. 15 minutes) about their internship
    * Keep this non-technical and as exciting as possible
    * We want people to be jazzed up about data science!!


## Resource Requirements

### Staff Checkpoints

* Philip will be the interns' daily point of contact and mentor.
* Charlie will hold weekly checkpoint meetings to guide the long-term direction
  of the project and to ensure that the interns aren't blocked.
* Steven will occasionally be present at these regular checkpoint meetings,
  subject to availability.


### Machine Learning Workshops

Several machine learning workshops will be held throughout the duration of
  the internship - run by various members of the data science team. These
  sessions will be open-door - that is, any other interested staff at FDS are
  welcome to join! Here is a rough plan:

1. Introduction to machine learning (Charlie)
    * Supervised/unsupervised
    * Types (regression, classification, dimensionality reduction)
    * How it actually works (training phase to learn some function)
    * Key components (task, loss function, optimiser or closed-form)
    * Training/testing/validation (why we split up datasets)
    * Bias and variance (overfitting and underfitting)
    * Hyperparameter tuning (which knobs to twiddle to improve performance)
2. ML algorithm basics (Phil)
    * Start out with a nice easy algorithm (probably linear regression)
    * Decision trees
    * Bayes' theorem
    * Feedforward neural networks (basic overview)
3. Advanced ML algorithms (Charlie)
    * Ensembling: bagging (e.g. random forest) and boosting (e.g. XGBoost)
    * Advanced NN architectures: CNN, LSTM/GRU
    * Dimensionality reduction: PCA, SVD, T-SNE (?)
    * The weird stuff: Extreme Learning Machines, random projection, etc
4. Preparing a dataset (Phil)
    * Splitting into train/test/validation sets (why?)
    * Cleaning data (imputing, denoising, resampling, etc)
    * Scaling numerical data (min-max, standardisation)
    * Encoding categorical data (one-hot, embeddings)
5. Evaluating your algorithm (Charlie)
    * Metrics for regression
        * Differentiable (MSE, MAE, MSLE)
        * Non-differentiable
    * Metrics for classification
        * Differentiable (hinge loss, huber loss)
        * Non-differentiable (F1, AUC, AUPRC)
        * Multi-class (cross-entropy)
5. Putting it all together (Phil and Charlie)
    * Let's build something simple (a linear regression) from scratch
        * Solve *iteratively* rather than closed-form - more general technique
    * Choose a dataset (Boston Housing)
    * Prepare the data
    * Run a couple of rounds of training
    * Compare performance at each round
    * Evaluate after training some more

Each session will take between 2 and 3 hours, and will take between 5 and 10
hours to write. This will be highly valuable for both the interns and the
company as a whole - we will increase skills among the technical staff (chiefly
the software developers), but also hopefully increase general interest among
the other staff. These sessions will start quite basic and then build up to
something fairly technical - hopefully we can design these sessions in such a
way that everyone can engage with them at a certain level (or that they can
attend the first two or three and then only the most dedicated stay for the
rest).
