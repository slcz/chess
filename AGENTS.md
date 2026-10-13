# Chess AI development guideline

## Dev environment tips
- Local environment is MacOS with 16GB memory, it is used for maintaining code, compilation, lightweight test.
- Most training and inferences happen on salad.com cloud instances. They usually come with RTX 3090/4090/5090.
- Cloud instances are not reliable and can become unavilable at any moment. Therefore, test scripts have to consider checkpoint, backup plan (coolaborate with the user).
- Create and maintain a github repository "chess" to store all the source repo.
- To transfer data to salad.com, use scp, not rsync.
- Salad.com instance generated checkpoint can be backed up on huggingface, the Key is in .env file. (HF_TOKEN)

## Planning and developement guideline
- High-level roadmap is in README.md; pipeline ops skills (container
  build, self-play rounds, train/eval) are in SKILL.md.
- When present new code or code changes, add descriptions in front for user to uderstand.
- The project is for educational purpose, so be verbose, explain/discuss design decisions with user until both sides are aligned.
- After current subtask is completed, don't rush to the next goal. Pause and ask.
- AI generated code is expected to be simple and clean. Reduce bloated code.
