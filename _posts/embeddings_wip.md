# The Mean(ing)s to an End: An Exploration through Word Embeddings

> "You shall know a word by the company it keeps"
>
> — J.R. Firth

I had just finished a lecture on Word2vec and was trying to talk about it with a friend of mine. I thought back to the lecture and came back with "the idea is that you can do math with words! You can take something like king - man + woman and it will be pretty close to queen." They asked me how it worked. I tried to look back on skipgrams and CBOW and nothing came to mind.

That gap between what is pratically a fun fact and how the mechanism works is what inspired me to learn how embedding algorithms really work, down to the nitty-gritty. I have decided to take this learning to a blog post, both to hold myself accountable to getting through it and to help consolidate what I find so that if another student like me gets stuck, they don't have to go digging for resources through other blog posts that cover on algorithms and technical papers that get very long and very tiring.

However, there are _many_ embedding algorithms so for convenience, and to allow myself to get these posts out in a reasonable time frame, they will be split into three parts. This first post will cover some technical background and will go over the most important algorithms that generate static embeddings. The second post will cover algorithms for context-based/dynamic embeddings and the third post will be more of an appendix that takes these text embeddings and combines them with other types of input like images and audio.

## Technical Background

Before we get into the meat of it, we're going to cover the important background information that will help ease the transitions between sections and also cover some things you might have forgetten that or might have slipped through the cracks.

### The Distributional Hypothesis

This is the fundamental building block for all word embeddings and for a lot of NLP. It was first formalized in 1954 by Zellig Harris where he defines the distribution of a linguistic element is "the sum of all its environments." He also gives us the canonical example that the words "eye-doctor" and "oculist" appear in all of the same contexts and therefore would be synonyms while "oculist" and "lawyer" do not share the same contexts and therefore would be different words[^1]. I guess the word oculist was a lot more common then.

The distributional hypothesis leads us directly to the co-occurence matrix. The hypothesis tells us that words that appear together will mean similar things but the matrix will show that to us. The co-occurence matrix is made by counting how many times two words appear together within some defined context window. Now imagine we have a three sentence dataset of the following:
1. The oculist examined the eye.
2. The eye-doctor examined the eye.
3. The lawyer examined the contract.

This dataset will create the following co-occurence table:

||the|oculist|examined|eye|eye-doctor|lawyer|contract
|---|---|---|---|---|---|---|---|
|the|0	|1|	2|	2|	1|	1|	1|
|oculist|1	|0	|1|	0|	0|	0|	0|
|examined|2	|1|	0|	0|	1|	1|	0|
|eye|2	|0|	0|	0|	0|	0|	0|
|eye-doctor|1	|0|	1|	0|	0|	0|	0|
|lawyer|1	|0|	1|	0|	0|	0|	0|
|contract|1|	0|	0|	0|	0|	0	|0|

You will notice that "eye-doctor" and "oculist" have the same vector embeddings, which would say they are the same word. Of course this is a very small and pretty unrepresentative dataset, but it goes to show the principle. Also, we refer to our vocabulary, the set of all words, as V. That would make this a V x V matrix. As our vocabulary grows in size, this will become more and more unwieldy, but it is still an important concept to keep in mind as we continue for things like Word2vec which build on the foundation of this concept.

### One-Hot Encodings

This is a super simple one, but I figured it needs mentioning as they show up absolutely _everywhere_ in NLP and in word embeddings. One-hot encodings are pretty much the simplest word embeddings you can create. Take your vocabulary and lay them out so that each one of them is now a dimension of a vector. Now, you put a 1 for the word that you have and leave all of the other numbers as a 0. For our prior dataset, this would be the embedding for the word "examined":

|the|oculist|examined|eye|eye-doctor|lawyer|contract
|---|---|---|---|---|---|---|
|0	|0	|1|	0|	0|	0|	0|

One important thing to note here is that you will often see one-hot encodings used in things like Cross-Entropy Loss, which is usually between two probability distributions. The key thing to note is that the encoding here is a probability distribution, specfically a delta function where one value shoots straight up and everything else is 0. This idea of the encoding as a distribution will be really important later on when we need to compare our generated probability distributions to real ones (the encoding) when training neural based embedding models.

### The Manifold Hypothesis

The average person uses between 1,500 and 3,000 words in regular conversation and the Oxford dictionary defines roughly 170,000 words that are still in use. That would make for a gigantic co-occurence matrix and you can imagine the headache that dealing with a 170,000 x 170,000 dimension matrix would be.

Instead, the manifold hypothesis tells us that we can typically model data that appears in high-dimensional space on a lower-dimensional manifold, called a latent space. This means that we can dramatically reduce the number of variables at play. In our case, we can reduce the number of dimensions needed to model words down from every single word in the English language and instead use fewer dimensions. This process also has the neat affect of forcing the model to be smarter in how it encodes information as it has less space to work with, similar to how an Autoencoder works.

### Distance Metrics

The curse of dimensionality says that in very high dimensional spaces, everything gets really, really far away from one another. This means that your data is extremely sparse and very hard to compare. The amount of data we have would need to grow exponentially with the number of dimensions, which would mean a massive amount of data added for each dimension.

Because of the curse of dimensionality, even our "lower" dimensional vector spaces will struggle when we use something like a Euclidean distance. Everything is just so far apart that it doesn't tell us much about the things that we are comparing.

For this reason, cosine distance is typically used. It is defined as:

$$\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

This value ranges from -1, meaning complete opposites, to +1, meaning the same word. This metric is the standard for measuring similarity between two words embeddings.

# The Statisical Models

Now that we have covered the background, we can move to the precussors of the embedding algorithms. That does mean that not all of these are technically embedding algorithms but they do provide valuable background for the true embedding algorithms to come. These being statisical algorithm also does mean that none of them use neural networks, usually becaues they predate the backpropagation algorithm. This gives us valuable insight as to how researchers were thinking about numerical representations of words before they could train a model to do it.

## TF-IDF

The idea of inverse document frequency was first invented in 1972 by Karen Spärck Jones, in order to measure the importance of word to a particular document. The goal here is to adjust for for the fact that some words will appear more frequently than others, so the number of documents they appear in can help us determine if they carry importance. Unlike something lie

## PMI

# Static Embeddings

## Word2Vec

## GloVe

## fastText

## StarSpace

## References

[^1]: Harris, Z. (1954). [*Distributional Structure*](https://www.its.caltech.edu/~matilde/ZelligHarrisDistributionalStructure1954.pdf). Word, volume 10, issues 2-3, pages 146–162