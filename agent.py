import asyncio
import os
from typing import Any
from github import Github, Auth
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import (
    FunctionAgent,
    AgentOutput,
    ToolCall,
    ToolCallResult
)
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPOSITORY = os.getenv("REPOSITORY")
PR_NUMBER = os.getenv("PR_NUMBER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

git = Github(auth=Auth.Token(GITHUB_TOKEN))
repo = git.get_repo(REPOSITORY)


def get_pr_details(pr_number: int) -> dict:
    """Fetch pull request details given a PR number."""
    pr = repo.get_pull(pr_number)
    commit_SHAs = []
    changed_files = []
    commits = pr.get_commits()
    for c in commits:
        commit_SHAs.append(c.sha)
        commit = repo.get_commit(c.sha)
        for f in commit.files:
            if f.filename not in changed_files:
                changed_files.append(f.filename)
    return {
        "author": pr.user.login,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "commit_SHAs": commit_SHAs,
        "changed_files": changed_files
    }

def get_commit_details(head_sha: str) -> list:
    """Fetch commit details given a commit SHA."""
    commit = repo.get_commit(head_sha)
    changed_files = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch if f.patch else "",
        })
    return changed_files

def get_file_contents(file_path: str) -> str:
    """Fetch the contents of a file from the repository."""
    contents = repo.get_contents(file_path)
    if isinstance(contents, list):
        return "\n".join([c.decoded_content.decode("utf-8") for c in contents])
    return contents.decoded_content.decode("utf-8")

def post_review_to_github(pr_number: int, comment: str) -> str:
    """Post a review comment to a GitHub pull request."""
    pr = repo.get_pull(pr_number)
    # Delete any pending reviews from current user to avoid 422 error
    for r in pr.get_reviews():
        if r.state == "PENDING":
            try:
                r.delete()
            except Exception:
                pass
    pr.create_review(body=comment, event="COMMENT")
    return f"Review posted to PR #{pr_number} successfully."


pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
commit_details_tool = FunctionTool.from_defaults(fn=get_commit_details)
file_contents_tool = FunctionTool.from_defaults(fn=get_file_contents)
post_review_tool = FunctionTool.from_defaults(fn=post_review_to_github)

llm = OpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

system_prompt = """You are a PR review agent. Your job is to:
1. Call get_pr_details with the PR number to get PR info and changed files.
2. Call get_commit_details on the latest commit SHA to see what changed.
3. Write a ~200-300 word review in markdown covering: what's good, contribution rules, tests, documentation, suggestions with quoted code.
4. Call post_review_to_github with the PR number and your review.

You MUST complete all 4 steps. Always end by posting the review."""

agent = FunctionAgent(
    llm=llm,
    name="ReviewAgent",
    description="Reviews a PR and posts the review to GitHub.",
    tools=[pr_details_tool, commit_details_tool, file_contents_tool, post_review_tool],
    system_prompt=system_prompt,
)


async def main():
    pr_number = os.getenv("PR_NUMBER")
    query = f"Write and post a review for PR number {pr_number}."

    handler = agent.run(query)

    async for event in handler.stream_events():
        if isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling tool: {event.tool_name}, args: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
