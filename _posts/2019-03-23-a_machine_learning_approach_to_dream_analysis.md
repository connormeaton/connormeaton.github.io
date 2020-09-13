---
title: 'A Machine Learning Approach to Dream Analysis'
date: 2019-03-23
permalink: /posts/2019/03/23/a-machine-learning-approach-to-dream-analysis/
tags:
  -
---
[![Screen-Shot-2019-03-01-at-8-45-26-AM.png](https://i.postimg.cc/yNtQ2FQ6/Screen-Shot-2019-03-01-at-8-45-26-AM.png)](https://postimg.cc/06G0wJ1h)
  *a t-SNE representation of the dream space of reddit users.*

Humans dream. Sometimes our dreams are as touching as visiting a deceased loved one. Sometimes they are wild subconscious utterances drenched in symbolism. Sometimes they are as mundane as eating a sandwich. Regardless, as mysterious as the content of our dreams are, even more puzzling is how across cultures, humans often dream similar dreams.

As our modern culture has made large efforts to sever itself from our symbolic roots, dreams are not treated with the same respect as the more objective facets of our lives. From a lack of time and interest or from embarrassment and shame, many of us ignore our dreams, casting aside a valuable tool for self-understanding. More so, it not only takes a scholar of comparative religion to build a body of knowledge to wade through dream content, but a rebel willing to look empirically at mystical content. This is rare.

Fortunately, we may be able to use modern machine learning techniques to conveniently dive into the ancient depths of our dreams and bring actionable insights back to our day-to-day life. This is where my project comes in, which was inspired by Carl Jung’s theory of the collective unconscious, where he proposed all humans share subconscious psychic content and as a result, may partly explain why we dream similar dreams.

Before going further, this article is not meant to be technical, but rather a practical exploration of human-centered machine learning. However, if you’re interested in the nitty-gritty, I’ve placed links to my code (all hosted on Github) where relevant. The project is still very much in development and much room for improvement exists in the code.

My goal was to use a machine learning technique called ‘natural language processing’ to build a program that helps a user understand their dreams by categorizing them and showing them other dreams similar to their own.

To start, I built a massive corpus of self-reported dreams. I did this through a process called ‘web scraping’, where I essentially built a little robot to dig through every self-reporting dream forum on Reddit and capture the text. In total, I gathered about 5,000 dreams. Web scraping code here.

As with anything that mirrors the human soul, the internet is a messy place, so the dreams were messy as well. Us humans can read them just fine, but machines need things a bit more precise. To make the dreams more machine-readable, I built a tool to clean up the text in a variety of ways. For example, I removed words like ‘it’ and ‘the’, and made sure similar words, or misspelled words, were all treated as one (ex., ‘waters’, watr’, ‘Water’, ‘watering’ were all converted to ‘water’.) Text processing code here.

The dreams were rendered machine readable, but they were just a pile of text files with no real structure. Fortunately, brilliant minds have invented a class of modeling techniques know as ‘unsupervised learning’, where a machine finds patterns that the deployer did not have the time, resources, or ability to find themselves. In my case, I programmed my machine to create a set number of categories of dreams based off of things like word similarity and sort each dream accordingly into a group. The algorithm I used is called ‘Latent Dirichlet Allocation’ and the code for this can be found here.

The result of this grouping was fascinating. Based off of reading the most relevant words per category, several interesting themes occurred from the corpus of dreams. For example, there were shared categories of dreams centered around air/water, exploring a structure, school, sickness/health, and violence. I doubt anyone is surprised to see these categories emerge, but what was fascinating is that these groupings allow us to leave to world of anecdote and observe these shared dream categories empirically. See the word clouds representing some of the categories below and please feel free to contribute to their interpretation.

[![Screen-Shot-2019-03-02-at-11-11-59-AM.png](https://i.postimg.cc/6Qq6MNgJ/Screen-Shot-2019-03-02-at-11-11-59-AM.png)](https://postimg.cc/XZM0vtnQ)

Much work can be done to tweak the model to make these categories even more interpretable and relevant. However, with the time scope of my project, I wanted to move on to next phase, making this useful for the average person.

I wrote a program that allows the user to write their dream as text and pass it into the model with the click of a button. The program then processes your dream and predicts which category it falls into. Then, from within that category, it shows you the three most similar dreams to yours. It does this using a mathematical concept called ‘term frequency–inverse document frequency’ and is essentially what Google uses to find relevant pages for you based on your search terms.

I designed this functionality because it was the most helpful guide to dream interpretation I could think of. Let me explain with an example.

In the past year, I’ve been dreaming a lot about tidal waves. I’ve never experienced tidal waves, have never seen one, and frankly, rarely think about them. This leads me to question whether my subconscious is actually concerned with tidal waves, or if they are a symbol of some other meaning, something not yet conscious.

I passed my dream into my program and with no surprise, it categorized it into the water/air group. This is obvious in my case, but if a user had a more ambiguous dream, categorizing it may provide a contextual framework to approach the dream and perhaps glean new insight.

What was even more exciting was that my program then presented me with the three most similar dreams from the original corpus. They were all about tidal waves and they all exhibited a similar mysterious quality as in my own. Reading through them gave me fresh insight to aid the very personal interpretation of my own dream, but even more, it was comforting to know that I was not alone with my obscure dreams.

The phenomena of dreams are, and will likely remain, a great mystery. However, just because they appear mystical, immeasurable, and bizarre does not mean they should be considered beyond the pale of intellectual pursuit. Anything we can do to take our dreams seriously, to observe them with both empirical skepticism and open-minded curiosity, will be tremendously useful the pursuit of understanding ourselves.
