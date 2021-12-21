--- 
title: "Classical approaches to Machine Learning"
author: "Chris Penfold"
date: "2021-12-21"
site: bookdown::bookdown_site
output: bookdown::gitbook
documentclass: book
bibliography: [book.bib, packages.bib]
biblio-style: apalike
link-citations: yes
description: "Course materials for An Introduction to Machine Learning"
cover-image: figures/cover_image.png
---

# About the course 

Machine learning describes a series of data-driven algorithmic approaches that simulate the “learning without being explicitly programmed” paradigm. These methods are particularly useful when limited information is available about the structure or properties of a dataset; also, real-world data rarely follows a well-defined mathematical distribution (due to technical variation in measurements, noise, etc) – assumption-free models offer flexibility for this type of input with the side effects of underlying characteristics of the dataset (e.g. through feature selection). The term “Machine Learning” encompasses a broad range of approaches in data analysis with wide applicability across biological sciences. Lectures will introduce commonly used approaches, provide insight into their theoretical underpinnings and illustrate their applicability and limitations through examples and exercises. During the practical sessions students will apply the algorithms to real biological data-sets using the R language and RStudio environment. All code utilised during the course will be available to participants.

## Prerequisites

* Some familiarity with R would be helpful.

## Schedule

Time | Data | Module
--- | --- | ---
14:00 – 15:00 | 30/11/21 | Linear regression / linear models
15:00 – 16:30 | 30/11/21 | Logistic regression
16:30 – 17:00 | 30/11/21 | Review and questions
14:00 – 15:00 | 2/12/21 | Artificial Neural Networks
15:00 – 16:30 | 2/12/21 | Convolutional neural nets and beyond
16:30 – 17:00 | 2/12/21 | Review and questions

## Github
The github reposotory for Classical approaches to Machine Learning containing code, datasets and lectures is availabile [here](https://github.com/cap76/AZCourse). The html textbook is found in docs/index.html. Individual chapters (files ending .Rmd) can be opened in RStudio as interactive markdown files.

## Google docs interactive Q&A

Clicking the link [here]{https://docs.google.com/document/d/1fDiVihZWsSiFKllsANFGd_jV3SK6jlq4IOj5dEPtQTM/edit?usp=sharing} takes you to the interactive Q&A document, where you can ask any questions you might have.

## License
[GPL-3](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Contact
If you have any **comments**, **questions** or **suggestions** about the material, please contact <a href="mailto:cap76@cam.ac.uk">Chris Penfold</a>.

## Colophon

This book was produced using the **bookdown** package [@R-bookdown], which was built on top of R Markdown and **knitr** [@xie2015].
