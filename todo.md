TO DO
===

- Fix video quality on android player clients. 
    - Try to use "tv" instead

- Re-implement automatic rubric tree construction
    - Attach assignment instructions to the tree
    - Attach eval examples to the tree

- Strategize about how to make the autograder better on videos without spreadsheet tool


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

- Add additional frequent frame sampling

- Manually make the spring rubrics for car loan and forecast

- Create a student metadata file, which includes all the artifacts and metadata-level information about them.
- Make a tool to access this metadata

- Implement spreadsheet reading
    - Initial design made
    - Fix the autograder not being about to see all of the charts, like trendlines and r^2 values. Maybe pass the entire metadata into the llm

- Add requirements.txt

- Add cost analysis for each run
    - which models are being used
    - how many tokens are being used
    - how much time it takes


OPTIONS FOR IMPROVEMENT
===

- Add recalculating and averaging category scores using Olivia's strategy for consistency and confidence scores

- For frame deltas frame selection, try to select a nearby frame neighbor that is the least blurry
    - using perceived frame quality

- Try making the angel/devil (positive/negative) evaluation (idea from Parker, described in my notes)

- Create a Docker container

- Parallelize/multithread for speedup
    - Also look into using AsyncOpenAI for higher concurrency
    - Test OpenAI rate limiting to find out how many open queries I can have at one time

- Use "smarter" models that are better at complex reasoning
    - OpenAI o1/o3, DeepSeek-R1, and Claude 3.7 Sonnet
- Use cheaper models for less critical tasks