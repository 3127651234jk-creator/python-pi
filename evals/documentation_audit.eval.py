"""Documentation audit host eval.

Port of ``packages/evals/evals/documentation-audit.eval.ts``. Every Markdown
page of the package's documentation becomes one case: the model reads the page
plus the implementation and must submit its verdict through the
``submit_documentation_audit`` tool, exactly once, as the final action.

A ``match`` verdict means no clear, user-visible contradiction was found;
non-match verdicts flag pages that need human review.
"""

from pathlib import Path

from pi.agent.types import AgentTool, AgentToolResult
from pi.ai.types import TextContent
from pi.evals.framework import describe_eval, tool_calls
from pi.evals.harness import PiCodingAgentHarnessOptions, create_pi_coding_agent_harness

TOOL_NAME = "submit_documentation_audit"

# The Python port's documentation: the Markdown pages at the package root.
package_root = Path(__file__).resolve().parents[1]
repository_root = str(package_root)
pages = sorted(entry.name for entry in package_root.glob("*.md") if entry.is_file())


def create_submit_audit_tool() -> AgentTool:
    """The final-submission tool; ``terminate`` stops the agent right after it."""

    async def execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(
            content=[TextContent(text="Documentation audit submitted.")],
            details=params,
            terminate=True,
        )

    return AgentTool(
        name=TOOL_NAME,
        label="Submit documentation audit",
        description="Submit the final verdict after completing the documentation investigation.",
        parameters={
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["match", "mismatch", "inconclusive"]},
                "explanation": {"type": "string", "minLength": 1, "maxLength": 2000},
            },
            "additionalProperties": False,
            "required": ["verdict", "explanation"],
        },
        execute=execute,
    )


harness = create_pi_coding_agent_harness(PiCodingAgentHarnessOptions(
    name="documentation-page-audit",
    tools=["read", "grep", "find", "ls", TOOL_NAME],
    custom_tools=[create_submit_audit_tool()],
))

AUDIT_PROMPT = """Audit this Pi documentation page against the repository implementation.

Documentation page: {documentation_path}
Repository root: {repository_root}

Read the complete page and the relevant implementation. Report a mismatch only for a clear, user-visible contradiction between an explicit documentation claim and actual runtime behavior. Follow the runtime path; names, comments, types, isolated helpers, and tests are not sufficient evidence by themselves.

Missing internal detail, ambiguous wording, hypothetical misuse, and undocumented edge cases are not mismatches. If the evidence is not decisive, report inconclusive. Otherwise report match.

For a mismatch, quote the claim, cite the implementation path and symbol, and state the concrete behavior a user would observe.

Call {tool_name} exactly once as your final action. Do not return prose."""


def register_audit_case(page: str) -> None:
    documentation_path = str(package_root / page)

    @describe_eval(
        "Audit documentation against implementation",
        harness=harness,
        title=f"{page} matches the implementation",
    )
    async def audit_page(run, _documentation_path=documentation_path):
        result = await run(AUDIT_PROMPT.format(
            documentation_path=_documentation_path,
            repository_root=repository_root,
            tool_name=TOOL_NAME,
        ))
        audit_calls = [call for call in tool_calls(result) if call["name"] == TOOL_NAME]
        assert len(audit_calls) == 1
        audit_call = audit_calls[0]
        assert audit_call["status"] == "ok"
        explanation = (audit_call.get("arguments") or {}).get("explanation")
        assert audit_call["arguments"]["verdict"] == "match", explanation


for _page in pages:
    register_audit_case(_page)
