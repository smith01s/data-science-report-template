---
title: "FDX Event Auto-Validation"
subtitle: "A response to IATA's concerns"
author: [Charles Newey]
approver: [Steven Smith]
issuenumber: 1
date: 18th January 2019
confidentiality: Commercial in Confidence
organisation: Flight Data Services
reference: DS-AEV-S2

mainfont: Century Gothic
fontsize: 14
toc: true
titlegraphic: figures/logo.png
titlepage: true
titlepage-color: B7C75A
titlepage-rule-height: 1
titlepage-rule-color: B7C75A
---

# Target model accuracy

One of the first comments in IATA's response to the auto event validation
project spec is from Anna, and we have included it below;

> \textit{``AQL results will supercede any model generated percentage.''}

It's not entirely clear what Anna is trying to say here, but in the interests
of thoroughness, we will provide a little bit more information throughout this
document about how we measure the performance of our classifiers. Jung has also
raised some concerns (namely about class imbalance) which we shall address here
too. Please be aware that this document contains a smattering of techinical
material and will assume a level of competence in mathematics and statistics.
Feel free to contact any of us with questions and we will be more than happy to
provide further clarification.


## How we measure performance

The simplest metric to measure our classifiers' performance is simply
*accuracy*. This means that we add up all of the events that our models
predicted correctly and divide that number by the number of total events. This
will give a number between zero and one that will tell us the *correct
classification rate* of our classifiers.

However, as Jung has suggested, this metric falls down in cases where there is
a *class imbalance* (for example, for events where there are significantly more
*invalid* events than valid ones). There are other similar metrics that take
this class imbalance into account --- and these are the metrics that we use to
evaluate the performance of our classifiers; namely $F_{1}$ score and balanced
accuracy ($Acc_{B}$). Refer to the appendices for a brief overview of these
metrics and their definitions.


## Proposed thresholds

We propose that an $F_{1}$ and $Acc_{B}$ of $0.90$ or higher is an acceptable
threshold for classification performance. We are certainly open to discussing
other thresholds --- however, bear in mind that as we *increase* the
performance requirements of our classifiers, the number of models that we can
build will necessarily *decrease* --- we would have to invest significantly
more time in ensuring that our models meet these more stringent quality
requirements. Thus, there is a trade-off between classification performance and
development time --- and we feel that this balance has been appropriately
reached with thresholds of $0.90$.


\clearpage

# Involvement with model development and feature selection

We are happy to share some general information about the models and features
used for event classification. There are currently some restrictions (both
technical and computational) which limit our ability to integrate more
"interesting" data sources such as time-series data and weather information.
While these sources are definitely worth investigating in greater detail in the
long-term, we are confident that our current KPV-based models will perform
adequately for the time being.

We are also open to suggestions for possible features to include in our model,
as we are always happy to work with people who have more domain experience! Be
aware that we won't always be able to include these features --- sometimes
these may be too difficult or slow to compute, or require too much development
overhead to be practical. We will, however, discuss any ideas that IATA may
suggest.

Please refer to the appendices for an example of the feature set and feature
importances for one of our trained models (in this case, a random forest --- a
common ensemble method involving bagging decision trees).


# Model interpretability

To help our models' intended users understand how their classifications are
made, we use a model interpretation library called [`shap`
(https://github.com/slundberg/shap)][shap]. This library uses a concept from
game theory (Shapley values) to determine the importance of each feature and
its contribution to a particular classification result. This helps users answer
questions such as "what impact does feature $x$ have on the validity of event
$y$?".

However, this is quite separate from the "why" --- we do not expect to provide
a step-by-step causal analysis for *why* a classifier came to a particular
decision. We have a similar response to Jung's comments on correlated
parameters in the feature engineering process (c.f. below);

> \textit{``...when validating unstable approach, during the flight, the human
> pilot might be estimating the altitude is too high than normal (which was the
> precursor), the pilot decides to reduce engine power, or change the flap
> configuration intentionally (human actions), however, during the final
> approach phase, the sudden tailwind (external factors) made the sink rate
> abnormal (results from external factors) and led to the unstable approach.''}

This kind of causal analysis is entirely out of scope for this project. We are
simply trying to classify whether a particular event occurrence was valid or
not (i.e. due to faulty aircraft, sensor errors, inaccuracies in data
extraction, numerical precision errors, and so forth). We do not expect to
provide a qualitative explanation of *why* something happened. Of course, some
parameters are correlated with others, but at the moment we include a very
limited set of parameters as features (and from our current investigation, this
is sufficient to validate the bulk of event occurrences in the vast majority of
cases).

[shap]: https://github.com/slundberg/shap


# Which algorithms do we use?

The real answer to this question is "it depends". Different algorithms perform
better or worse in different circumstances, and we use the most appropriate
tool for the job. During the model selection process, we test a range of
different algorithms (and hyperparameter combinations) for each event, and then
select the highest-performing combination. Typically so far we have had success
with ensemble tree models (e.g. random forests and gradient-boosted decision
trees) and naive Bayes classifiers, but if we discovered an algorithm that
yielded better performance, then we would use that preferentially.

The only factor that limits the diversity of models that we can use is IATA's
requirements on model interpretability --- because interpretability appears to
be a high priority, we've expressed a preference for simpler and more
intelligible models --- such that their decisions will be easier to reason
about.


# Comparison of human validation results

We agree completely that the classifiers should be compared to human-level
performance. There are a few different ways that we can do this --- which will
be roughly outlined below. In these sections we discuss some mathematical
metrics that we can use for this purpose --- and for clarity, their definitions
have been provided at the end of the document. We are happy to explain these in
more detail if desired.


## Inter-rater agreement

We can compare the *similarity* of the outputs of two classifiers (i.e. we
would treat a human analyst as one "classifier", and one of our ML models as
another) using a variety of statistical methods for measuring "inter-rater
agreement". This could take the form of a Jaccard similarity score or a more
statistically robust metric (e.g. Cohen's $\kappa$ --- which takes into account
the probability of an "agreement" occurring by chance). This approach simply
measures the *similarity* of two different classifiers --- and makes no
assumptions about the veracity of the human analyst's classifications. See the
appendices for a brief overview of these metrics.


## Industry-standard classification metrics

Alternatively, if we were to treat a human analyst's classifications as "ground
truth" (i.e. we make the assumption that their work is *absolutely* correct),
we can compare our classifiers' performance using industry-standard classifier
metrics (e.g. F1 score, balanced accuracy score, or something else). See the
appendices for a brief overview of this metric.


## Estimating the *error rate* of a classifier

A final approach we can take is to simply compare the *proportion* of events
that our classifiers will get wrong (i.e. the *error rate*). This approach has
the advantage of allowing us to express *uncertainty* in our measurements ---
that is, if we measure the *error rate* by performing statistical experiments
using small batches of randomly-selected events, we can give a *margin of
error* and a *confidence interval* on our measurements. See the appendices for
a brief overview of this metric (and some experiment sample sizes that we've
calculated to demonstrate the trade-off between accuracy and human workload).


# Sharing models with IATA

We are happy to share with IATA various types of metadata relating to each of
our models --- including such things as Shapley value feature importances,
performance information, and high-level information about the algorithms that
we use for each event type --- and obviously, the features that we use to train
the models with.

However --- we have not used *any* FDX data for training any of our models. The
data that we used to construct our training, test, and validation sets is
entirely proprietary to Flight Data Services, and is sensitive to Flight Data
Connect customers --- there is no question of making these datasets available
to IATA. Similarly, as our models have been trained on this sensitive data, any
model artifacts (e.g. weight matrices, trained model binaries, feature
encoders, etc) will not be shared with IATA.

In summary, we are able to share *metadata* regarding the models, but will not
be sharing data or model *artifacts* due to their sensitive nature.


# Data distribution

Jung raised concerns regarding the distribution of the data (i.e.  classifying
the validity on particularly rare event types), and we felt this was worth
addressing. The models that we are developing for event classification are not
meant to be a completely automated replacement for a human analyst. The
software that we are building is primarily intended to be an *assistive
technology*.

Machine learning algorithms require a considerable volume of data to work
effectively. Because of this, we are only building models for the *most common
events* --- those where we have the *most data*. We aren't building models for
particularly uncommon events (or realistically, events where we have fewer than
tens of thousands of occurrences), because there simply won't be enough
training data for a machine learning algorithm to achieve sufficient accuracy.

It's worth reiterating at this point that we're not aiming to build a framework
for causal analysis --- we aren't looking to explain *why* a pilot may have
taken a particular action, or to notice that a particular control input was as
a result of a particular external condition, or anything of that variety. Our
system is intended to operate at a far more abstract level than that --- simply
using KPVs and other metadata about the flight to infer the probability of an
event being invalid.

These models are designed to catch only *the most obviously invalid* events ---
typically those where we receive an absurd or impossible sensor reading, or
events trigger during the wrong phase of flight (for example, certain events
can trigger tens of thousands of time per flight due to a faulty squat sensor
in the landing gear of certain aircraft). Likewise, we aren't going to build
anything that will claim to be capable of discerning whether a control input
was due to external factors (e.g. ATC instructions, etc) --- these situations
are far more nuanced and we expect human involvement to be required at this
stage.


# What happens when prediction confidence is low?

We have explicitly designed our models to operate within these constraints, and
recognise cases where prediction confidence is low. We have achieved this by
selecting *probabilistic classifiers* for our event classification tasks. The
output from these algorithms is simply a *probability* --- a number between $0$
and $1$ that indicates how likely it is that a particular event is invalid.
Clearly, if this value is very high (e.g. $0.95$), then this means our
classifier has a particularly high confidence that the event is invalid
(conversely, if the value is $0.05$, then the classifier has a high confidence
that the event is valid).

However, if a predicted probability is closer to $0.5$, we take this to mean
that the classifier has a *low* confidence in that prediction. In cases where
the classifier has a low confidence (typically where the predicted probability
is between $0.25$ and $0.75$), we mark this classification as "uncertain" ---
and these events are then earmarked for human validation.


# Additional features

Jung raised some salient points about additional features that we may want to
include --- particularly regarding time-sequential data. This is something we
are already actively looking into for future iterations of this project ---
however, due to time constraints and limitations on computing power, we are
currently only using *instantaneous* data (that is, KPVs and other metadata
surrounding a particular event occurrence). Rest assured that this is a
possibility that we are considering and actively working towards --- just not
yet.

Likewise, we are interested in including operational, regional, and local
characteristics of various areas (and operators) into our models. We would also
be interested in including some primitive forms of weather data into our models
--- but again, this is currently out of scope for the first iteration of this
project --- we aren't even sure where we would begin to collect some of this
data! Again, we're happy to listen to suggestions and ideas if they are likely
to significantly improve the performance of our models --- as scientists, we're
always open to new ideas that will help make our work better!


# In closing

We believe that it might be useful and instructive for both parties to
participate in a video conference to discuss the project and exactly what it is
that we have done --- and what we plan to do. We would like to discuss each of
the points mentioned above in more detail --- and it would be useful if IATA
were able to draw up a list of their questions beforehand (so that we can
prepare responses and other materials --- and then we can use it as an agenda).


# Appendices

## Classification metrics

### Balanced accuracy (BACC)

Balanced accuracy is the average of the *true positive rate* and the *true
negative rate*. This is the closest metric to standard accuracy, but takes
different class distributions into account.

$$
Acc_{B} = \frac{1}{2} (\frac{TP}{P} + \frac{TN}{N})
$$


### F1 score

This is the *harmonic* mean of two classification metrics; *precision* and
*recall*. This takes into account class imbalance and weights true positives
and false positives equally.

$$
F_{1} = \frac{2{TP}}{2{TP} + {FP} + {FN}}
$$


## Inter-rater agreement metrics

### Jaccard similarity score

The Jaccard similarity coefficient is defined below. This measures similarity
between two sets (in this case, two sets of event classifications --- one by a
human, and another by a model).

$$
J(A, B) = \frac{| A \cap B |}{|A| + |B| - |A \cap B|}
$$


### Cohen's $\kappa$

The definition for Cohen's $\kappa$ is below. $p_{o}$ is simply the *observed*
accuracy of a given classifier, while $p_{e}$ is the probability of an
agreement due to chance.

$$
\begin{aligned}
\kappa &= 1 - \frac{1 - p_{o}}{1 - p_{e}} \\
p_{e} &= \frac{1}{N^{2}} \sum_{k} n_{k1}n_{k2}
\end{aligned}
$$


## Binomial proportion confidence interval

One of the key quantities that we need to estimate to verify the performance of
our machine learning models is the *error rate*. We estimate the error rate on
*unseen* data by sampling an analyst's manual validations and computing the
proportion of results that our classifiers get *incorrect*.

We use a Gaussian approximation of the binomial distribution to estimate this
proportion (to within a given margin of error and statistical confidence). For
clarity, I have included the formulae required to compute the approximate
number of samples required, as well as some Python code that we use for
computing it.


### Formula for binomial proportion confidence interval

$$
\hat{p} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}
$$

We can rearrange the above formula to yield $n$ --- the number of samples
required to estimate a given proportion - hence;

$$
n = \hat{p}(1 - \hat{p}) \cdot \frac{z^{2}}{m^{2}}
$$


\clearpage
### Example

We need to take $385$ samples if wish to estimate our algorithm's error rate to
within a $\pm 5\%$ margin of error with a 95\% statistical confidence.

As we increase our desired statistical confidence (and decrease our desired
margin of error), the number of samples required increases drastically. For
example, to estimate our algorithm's error rate to within a $\pm 1\%$ margin of
error with a $99\%$ statistical confidence, we would need to take $16588$
samples.

A few more sample sizes for different margins of error and confidence intervals
are included below.

| Margin of Error  | Confidence    | Samples |
|------------------|---------------|---------|
| 0.01 ($\pm 1\%$) | 0.95 ($95\%$) | 9603    |
| 0.025            | 0.99          | 2654    |
| 0.05             | 0.99          | 664     |
| 0.05             | 0.95          | 385     |
| 0.10             | 0.99          | 166     |

At this stage, we are recommending adopting a $95\%$ confidence interval on a
$\pm 5\%$ margin of error. For example, if our model is correct $92\%$ of the
time, this will mean that we have a $95\%$ confidence that our model's *actual*
accuracy lies in the interval $[87\%, 97\%]$.


\clearpage

### Python code for computing sample sizes

```python
def binom_prop_sample_size(
        margin_of_error=0.05,
        confidence=0.95,
        p_prior=0.50):
    """
    Calculate the sample size required to estimate a
    binomial proportion to within a particular margin
    of error at a particular level of statistical
    confidence. This uses a Gaussian approximation to
    the binomial distribution; the two distributions
    are similar enough in the limit of n.

    The p_prior argument is difficult to explain. It
    is essentially the prior estimate of the
    proportion. Setting this to 0.50 maximises the
    entropy in the prior distribution and thus
    maximises the number of samples required to
    arrive at the desired level of confidence. This
    ensures that the results of the experiment are as
    trustworthy as possible by assuming an
    uninformative prior.

    Resources:
        - https://goo.gl/xhgPbS
        - https://goo.gl/p9BZda
        - https://goo.gl/CXvra6

    Arguments:
        margin_of_error: Two-sided interval (default: 0.05)
        confidence: Confidence in measurement. (default: 0.95)
        p_prior: Prior probability. (default: 0.50)
    """
    import numpy as np
    import scipy.stats as st

    z_score = st.norm.ppf( 1 - ((1 - confidence) * 0.5) )
    return np.ceil(
        (p_prior * (1 - p_prior)) *
            ((z_score ** 2) / (margin_of_error ** 2))
    )
```


## Case study; A random forest model for "Engine N1 Low" events

### Model performance analysis

As we discussed above, we take a variety of different summary metrics to
describe the high-level performance characteristics of our model. This
particular model comfortably meets our designated $0.90$ threshold, and thus we
can be confident that it will work well.

| **Metric**        | **Value** |
|-------------------|-----------|
| Balanced accuracy | 0.957     |
| Recall            | 0.916     |
| Precision         | 0.979     |
| AUC               | 0.957     |
| F1                | 0.946     |


### Feature importances

This is a small sample of the feature set that we're using for our models. Bear
in mind that the feature space for our models is quite high-dimensional and if
all features were included below... it would be a very long list! The "feature
name" column should hopefully be fairly self-explanatory --- but the "feature
importance" column may not be. The "feature importance" column shows a value
between $0$ and $1$ to describe how *significant* that particular feature is
--- that is, how useful it is in determining the validity of the event.

In this instance, we see that `time_before_landing` and `time_after_takeoff`
are the two most important features, explaining approximately $35\%$ of the
variance in our model. This suggests that one of the primary causes of event
invalidity is events triggering in the incorrect phase of flight (e.g. "N1 low"
events generated outside of the landing phase). Another significant feature
relates to engine types --- a little investigation reveals that this is tightly
linked to whether the aircraft in question has a FADEC (full authority digital
engine controller) or not.


| **Feature Name**                  | **Feature Importance** |
|-----------------------------------|------------------------|
| `time_before_landing`             | 0.206                  |
| `time_after_takeoff`              | 0.149                  |
| `exceedance`                      | 0.139                  |
| `similar_events_on_aircraft`      | 0.128                  |
| `similar_events_on_flight`        | 0.069                  |
| `engine_type_name="Trent 556"`    | 0.052                  |
| `engine_type_name="AS907-1-1A"`   | 0.032                  |
| `family="BD-100"`                 | 0.031                  |
| `family="A340"`                   | 0.029                  |
| `family="A321"`                   | 0.012                  |
| `family="ERJ-170/175"`            | 0.008                  |
| `engine_type_name="CF34-8E5"`     | 0.006                  |
| `family="CL-600"`                 | 0.006                  |
| `engine_type_name="CFM56-5C4"`    | 0.005                  |
| `engine_type_name="CFM56-5B6/2P"` | 0.005                  |
| ...                               | ...                    |
| ~2000 more features...            | ...                    |


### Shapley value interpretation

The charts that are output by the `shap` package need a little concentration to
understand --- certainly, this is true the first time you see them! We've tried
to add a brief explanation below to each of the following plots, so that
hopefully we can pass on some intuition as to the information they're showing.


#### Force plot

This chart shows a similar sort of information to the feature importance table
above --- just in a slightly different format. This chart shows a list of
features and feature values at the bottom, with the blue and pink bars
describing their overall contribution to the overall decision made by the
model. Features in **blue** mean that they *reduce* the probability of an event
being classified as "invalid" --- whereas the features in **pink** *increase*
the probability of an event being classified as invalid.

![A "force plot" output by the `shap` package.](./figures/shap_force_plot.png)


#### Summary plot

The summary plot shows similar information again, but with a slightly different
perspective. A list of the most informative features is on the left-hand side
--- and the points on the chart represent the contribution these features have
towards the overall decision of the classifier. Points in pink indicate that
*high values* of these features have an impact on classifications, and points
in blue indicate that *low values* have an impact. To take an example, if we
look at `similar_events_on_aircraft` (second from bottom), we can see that high
values (i.e. lots of similar events on aircraft) slightly increases the
likelihood that an event will be classified as *invalid*.


![A "summary plot" from the `shap` package.](./figures/shap_summary_plot.png)
