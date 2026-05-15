import asyncio
import os
import sys
from typing import Any
from github import Github, Auth
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import (
    FunctionAgent,
    AgentWorkflow,
    AgentOutput,
    ToolCall,
    ToolCallResult
)
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPOSITORY = os.getenv("REPOSITORY")
PR_NUMBER = os.getenv("PR_NUMBER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

git = Github(auth=Auth.Token(GITHUB_TOKEN))
repo = git.get_repo(REPOSITORY)


# --- GitHub tools ---

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

def get_commit_details(head_sha: str) -> list[dict[str, Any]]:
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
    pr.create_review(body=comment, event="COMMENT")
    return f"Review posted to PR #{pr_number} successfully."

# --- State management tools ---

async def add_context_to_state(ctx: Context, context: str) -> str:
    """Save the gathered context to the workflow state."""
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = context
    await ctx.store.set("state", current_state)
    return "Context saved to state."

async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Save the draft comment to the workflow state."""
    current_state = await ctx.store.get("state")
    current_state["review_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "Draft comment saved to state."

async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Save the final review to the workflow state."""
    current_state = await ctx.store.get("state")
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "Final review saved to state."


# --- Tools ---

pr_details_tool = FunctionTool.from_defaults(fn=get_pr_details)
commit_details_tool = FunctionTool.from_defaults(fn=get_commit_details)
file_contents_tool = FunctionTool.from_defaults(fn=get_file_contents)
post_review_tool = FunctionTool.from_defaults(fn=post_review_to_github)
add_context_tool = FunctionTool.from_defaults(async_fn=add_context_to_state)
add_comment_tool = FunctionTool.from_defaults(async_fn=add_comment_to_state)
add_final_review_tool = FunctionTool.from_defaults(async_fn=add_final_review_to_state)

llm = OpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
)


# --- ContextAgent ---

context_system_prompt = """You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent."""

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context from the GitHub repository including PR details, changed files, and file contents.",
    tools=[pr_details_tool, commit_details_tool, file_contents_tool, add_context_tool],
    system_prompt=context_system_prompt,
    can_handoff_to=["CommentorAgent"]
)


# --- CommentorAgent ---

commentor_system_prompt = """You write review comments for pull requests as a human reviewer would.

You have already received PR context with details and changed files. Write your review NOW with what you have - do NOT ask for more information.

INSTRUCTIONS (follow in order):
1. Write a ~200-300 word review in markdown covering: what's good, contribution rules, tests, documentation, suggestions.
2. Call add_comment_to_state with your review.
3. Hand off to ReviewAndPostingAgent.

DO NOT ask for diffs or more info. DO NOT return a final response. ALWAYS hand off after saving the comment."""

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment.",
    tools=[add_comment_tool],
    system_prompt=commentor_system_prompt,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)


# --- ReviewAndPostingAgent ---

review_and_posting_system_prompt = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub."""

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft comment, checks if it meets quality criteria, and posts the final review to GitHub.",
    tools=[add_final_review_tool, post_review_tool],
    system_prompt=review_and_posting_system_prompt,
    can_handoff_to=["CommentorAgent"]
)


# --- Workflow ---

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review": "",
    },
)


# --- Main ---

async def main():
    pr_number = os.getenv("PR_NUMBER")
    query = f"Write a review for PR number {pr_number}"

    handler = workflow_agent.run(query)

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
