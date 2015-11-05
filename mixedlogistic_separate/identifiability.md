It is a concern that a mixture of multi-class Bernoulli model, i.e. mixed logistic with binary or multi-class Bernoulli response, is not identifiable. 

There are a few reference listed below:

- "As will be shown below repeated measurements for some individuals are valuable information in order to ensure the identifiability of the model." [^1]

- "... class of all finite mixtures of binomial distributions is not identifiable. Information as to the nature of those sub-families of binomial distributions that are identifiable under finite mixture may be gleaned from ..." [^2] indicates that finite mixture of binomial models are not identifiable in general. In the paper, the author didn't consider covariates. 

- "For binary data we can rewrite equation (3.7) as ... The above equation implies that we only modify the link function with the probability ... In this case, no matter whether the binary responses are heterogeneous, the responses always have Bernoulli distributions."

    "Although an unlimited class of finite binomial mixture may not be identifiable, class of finite mixture of some subfamilies of binomials may be identifiable." [^3]


[^1]: 2008, Grun, et al. Identifiability of Finite Mixtures of Multinomial Logit Models with Varying and Fixed Effects

[^2]: 1963, Teicher, Identifiability of Finite Mixtures

[^3]: 1994, Peiming Huang, Mixed Regression Models for Discrete Data, pp146-147
