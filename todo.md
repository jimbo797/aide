TO DO
===

- Look at biggest discrepencies between actual and AIDE scores, and plan how to decrease these differences
    - Video quality
    - Sections A-D
        - Major issue with section C

- Fix video quality on android player clients. 
    - Try to use "tv" instead
    - Pass to Parker if can't fix

- Pivot to using the new rubric from the spring
    - make sure to divide the students between "forecasting" and "car loan", since these are different rubrics

- Choose 10 videos randomly to watch and know really well
    - Make sure all the important frames are extracted
    - Cross check each video with the true grade and the autograder grade

- Finish cleaning up codebase 
    - deleting old files
    - define the correct model to be used everywhere (preprocess and eval)

- Re-implement automatic rubric tree construction
    - Attach assignment instructions to the tree
    - Attach eval examples to the tree


DONE
===

- Rubric re-structure
    - tree
    - Categories are the smallest unit that can be assigned a score for
    - Criteria are marked as "met" or "not met" and are used to determine the score for the category.

- Add evidence for criteria being met

- Refactor codebase for preprocessing and evaluating student assignments

- Add a "thinking mode" to the eval agent that plans, gets evidence, talks about what score to give, and then after that makes the assessment
    - The idea is to establish more context to guide the LLM in generating the eval

- Multrithreading for annotating frames and evaluating leaves
    - Using asyncio

- Analysis tools for visualizing AIDE's performance

- Lower frame difference threshold for choosing key frames to make sure all important frames are captured
    - Already lowered from 0.32 to 0.1


OPTIONS FOR IMPROVEMENT
===
- Add analysis for runtime and cost

- Add recalculating and averaging category scores using Olivia's strategy for consistency and confidence scores

- Add frame random sampling
- For frame deltas frame selection, try to select a nearby frame neighbor that is the least blurry
    - using perceived frame quality

- Try making the angel/devil (positive/negative) evaluation (idea from Parker, described in my notes)

- Create a Docker container
- Add requirements.txt

- Parallelize/multithread for speedup
    - Also look into using AsyncOpenAI for higher concurrency
    - Test OpenAI rate limiting to find out how many open queries I can have at one time