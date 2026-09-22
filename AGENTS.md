# Chess AI development guideline

## Dev environment tips
- Local environment is MacOS with 16GB memory, it is used for maintaining code, compilation, lightweight test.
- Most training and inferences happen on salad.com cloud instances. They usually come with RTX 3090/4090/5090.
- Cloud instances are not reliable and can become unavilable at any moment. Therefore, test scripts have to consider checkpoint, backup plan (coolaborate with the user).
- Create and maintain a gitgub repository "chess" to store all the source repo.

## Planning and developement guideline
- High level development plan is in PLAN.md file.
- When present new code or code changes, add descriptions in front for user to uderstand.
- The project is for educational purpose, so be verbose, explain/discuss design decisions with user until both sides are aligned.
- After current subtask is completed, don't rush to the next goal. Pause and ask.
