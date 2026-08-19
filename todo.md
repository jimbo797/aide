TO DO
===

- Re-implement automatic rubric tree construction
    - Attach assignment instructions to the tree
    - Attach eval examples to the tree

- remove the sheet name param from evaluate_class and design a better, less intrusive method, since this is not in the spirit of generalization

- add option to use models from other model providers


DONE
===

- Add tools to access sources

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

- tool that outputs points lost per student and a concise reason why

- tool for TAs to go back and modify scores easily



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

- Model specialization
    - Use "smarter" models that are better at complex reasoning for evaluating 
    - Use cheaper models for less critical tasks

- Automatically query for model pricing instead of hard-coding values. 
    - Some but not all model prices are on https://www.llm-prices.com/, and the raw data is at https://www.llm-prices.com/current-v1.json 