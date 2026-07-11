# Diagnosing

## Run the baseline

After running the baseline, it shows that the baseline does not support unanswerable questions about docs and also retrieves the whole document even when only a specific answer is needed. The first step is to add support for unanswerable questions for that just add some treshholds first, and then improve chunking so the system can return answers that are more specifically relevant to the question. also its require to add eval dataset i use glm-5.2 to create eval dataset.
