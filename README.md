## How It Runs

The workflow triggers on every `pull_request` event:

1. GitHub Actions checks out the code and installs dependencies
2. Runs migrations and pytest with coverage reports
3. Executes `agent.py`, passing the PR number and credentials as environment variables
4. The agent analyzes the PR and posts a review

The review is posted as the `github-actions[bot]` user, making it look like an automated reviewer on every PR.

## Setup

1. Clone the repo
2. Create a `.env` file with your `GITHUB_TOKEN` and `OPENAI_API_KEY`
3. Install dependencies: `poetry install`
4. Run locally: `poetry run python agent.py`

For GitHub Actions, add `OPENAI_API_KEY` as a repository secret. The `GITHUB_TOKEN` is provided automatically by GitHub.

## What I Learned

- Designing agent architectures: experimented with multi-agent handoff patterns (Context → Commentor → Reviewer) before simplifying to a single agent with focused tools for reliability
- Handling production edge cases: managing pending reviews, deduplicating commit data, and gracefully recovering from API errors
- LLM prompt engineering: crafting system prompts that reliably trigger tool calls instead of hallucinating responses
- CI/CD integration: orchestrating LLM agents inside GitHub Actions workflows with secure secret management

## Demo

See [PR #4](https://github.com/jushi819/recipes-api/pull/4) for an example of the agent in action!

---

Built as part of Hyperskill's AI Engineering course. Open to feedback and ideas. Feel free to open an issue or reach out!
