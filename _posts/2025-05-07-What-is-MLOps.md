---
layout: default
title:  "What is MLOps"
date:   2025-05-07 09:00:00 +02:00
---

The newest member of Ops family of names is MLOps. Like with every other position name in software engineering, it does not correspond to **actual** actions performed by theese specialists, but to the **purpose** of those actions. This article is an attempt at claryfying *why* companies need to MLOps.

## Prediction quality > other KPI's

More often than not, software is used to answear a "what is going to hapen next" based on the histry collected so far. That might be predicting the next tsunami, recession, word or body movement. A logical consequence of our deep care for prediction quality is that we want the process to be as scientific and standarized as possible. We want to reduce human error, isolate variables that influence performance, have control groups, a full package. 

## Software vs. AI

In conventional software, the reaction of a program to external information is described in code. When we modify code, we have a good intuition how that change is going to affect the entire system. On the other hand, in Ai systems the reaction of the program to external information is described in data. That has quite profound consequences, as it shifts attention from writing code to collecting as much data as possible and definining goals, while program logic is generated by a specific algorithm. We then compare the results of that generation, that were trained under slightly different starting conditions.

## MLOps is DevOps coulture in Ai and Big Data

Long ago, software had a release cycle of hardware products. Requirements collection, architecture design, coding, testing, defect correction and release. From an outside perspective, release looked like throwing a ready made solutions through a fence, and hoping there is someone else on the other side to catch it and care for it. That someone else were admins, who were responsible for somehow makeing the program they've never seen work. DevOps is shifting that reponsibility onto people that wrote the damn app. Developers did what developers do and automated the hell out of the processes necessary to keep the app running. The Ai world is in a similar place - data scientists build some binary file and send it over to developers, who are responsible for ensuring the model is running correctly, despite not knowing what that actually means. The responsibility to ensure the (and ability to react to unsatisfactory) model performance should be as much in the hands of a data scientist as possible. MLOps is really automations that enable the data scientists to do just that. It comes with several challanges, four of which we will discuss further

## Data storage

As we have discussed, data is what decides program behaviour. As we (hopefully) know, we should keep track of changes that influence program behaviour. Unfortunetelly, versioning data is more complex than versioning code. There are 2 reasons for that. First is that code takes much less space. A git repo is not going to exceed 5gb of code, while corporations count the training data in petabytes. The second is data governance - code leak means a rewrite. Data leak means a lawsuit. 

## Scaling computations

Scaling compute has always been hard. We wouldn't have work without it. It comes without saying that scaling compute comes with technical and legal challanges. Scaling to cloud is a legal nightmare for large corporations, scaling internal infra brings challenges in its own right. 

## Artifact versioning
The result of training is a binary file, just as an .exe file is. The difference between them is their stability - handwritten changes to a program are more prodictable and trustoworthy than outputs of a brainless algorithm. We need to much more deeply analize and track the changes in the results we achive over time. 

## Monitoring

The lack of trust to the resulting artifact extends to constantly monitoring queries it receives and results it returns to compare them to queries and results we've seen during training phase. The algorithm should only return answears to questions that fit into a range of a training set. If we trained our algorithm on spanish, we should'nt allow queries in russian. If we never return a prediction of sugar level in blood above 200, we should porobably flag a result of 2000. If we see a lot of queries that do not correspond to a training set or anomalies in responses, it might be a sign to retrain our model.

## Summary

The point of this article is to demostrate what is MLOps and why it is hard. In a sense it is hard for the same reasons why software is hard - just in its own unique way. As always, why comes before what and how


