---
layout: post
title: "Probing Multilingual BERT for Ergativity in Basque"
---

Multilingual language models like mBERT are trained on data from dozens of languages, most of which follow familiar nominative–accusative patterns such as English or Spanish. A natural question is whether these models genuinely acquire language-specific syntactic structure, or whether they project minority languages into the mold imposed by dominant training data.

In this post, I summarize a probing study that asks whether mBERT internally represents **ergative–absolutive alignment** in Basque, a typologically distinct language isolate.

## Why Basque is a hard test

In nominative–accusative languages, the subject of an intransitive verb and the agent of a transitive verb are treated alike. Basque instead aligns the subject of an intransitive verb with the object of a transitive verb (the absolutive), while marking the transitive agent with an ergative suffix. This distinction is morphological rather than positional, and Basque word order is relatively free.

For a model trained largely on Indo-European languages, there is a real risk that it might rely on linear heuristics or majority-language subject biases, rather than encoding Basque morphosyntax.

## Probing setup

To test this, I used a probing framework based on linear classifiers trained on frozen internal representations from each layer of mBERT. The task was to distinguish ergative from absolutive nouns in the Basque-BDT Universal Dependencies treebank.

Several design choices were important:

- I restricted the data to common nouns with explicit `Case=Erg` or `Case=Abs` annotations.
- Because Basque is agglutinative, I aligned UD labels with mBERT tokens using a last-subtoken strategy that targets the suffix position where case information is realized.
- I trained a separate probe for each layer to analyze how syntactic information evolves across depth.

The goal was not downstream performance, but to test whether ergativity is **linearly recoverable** from the model’s representations.

## Surface transfer vs. deep acquisition

The experiment was guided by two competing hypotheses.

**Surface transfer:** mBERT treats Basque case as a shallow morphological feature and clusters intransitive subjects with transitive agents, reflecting a nominative bias.

**Deep acquisition:** mBERT encodes ergative–absolutive alignment structurally, clustering intransitive subjects with transitive objects.

To directly test for majority-language interference, I introduced a *Nominative Bias Score*, measuring how often intransitive subjects are misclassified as ergative.

## What the probes reveal

The results favor the deep acquisition hypothesis.

Probe accuracy was high across layers, peaking at **95 percent in Layer 9**, consistent with prior work showing that mid-to-upper layers encode syntactic structure. More importantly, the Nominative Bias Score at the peak layer was **0.0366**, meaning that fewer than four percent of intransitive subjects were incorrectly treated as agents.

In other words, mBERT does not appear to force Basque into a nominative template. Instead, it maintains a distinct representational geometry where absolutive arguments cluster together, regardless of grammatical role.

Layer-wise analysis also revealed a meaningful trajectory. Early layers exhibit slightly more bias, which is progressively reduced as representations become more abstract.

## Why this matters

These findings suggest that multilingual pretraining can support language-specific syntactic representations, even for low-resource languages with typological features absent from most of the training data.

More broadly, the results argue against the view that multilingual models rely only on surface heuristics. At least in this case, mBERT appears capable of encoding non–Indo-European morphosyntax in a structurally faithful way.

The Nominative Bias Score also provides a general diagnostic tool for studying cross-lingual interference, and could be extended to other typological phenomena such as split ergativity, word order variation, or agreement systems.


The complete paper is available here:

[PDF](/assets/papers/probing-mbert-ergativity-basque.pdf)
