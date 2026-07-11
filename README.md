# Diagnosing

## Run the baseline

After running the baseline, it shows that the baseline does not support unanswerable questions about docs and also retrieves the whole document even when only a specific answer is needed. The first step is to add support for unanswerable questions for that just add some treshholds first, and then improve chunking so the system can return answers that are more specifically relevant to the question. also its require to add eval dataset i use glm-5.2 to create eval dataset.

After adding threshold 5 for the baseline and evaluating on data, it shows that the main problem is when the question is unanswerable but the model returns an answer to it, resulting in false positives. As shown in the score distribution plot, there is no threshold that can well separate the answer.

![score distribution](results/baseline_tresh/score_distribution.png)

![confusion matrix](results/baseline_tresh/confusion_matrix.png)

i think the problem is that the chunk contexts arent specific and are very general so every query that point a little to that even unrelated can have answer.