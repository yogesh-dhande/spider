from __future__ import annotations

QA_SYSTEM_PROMPT = (
    "You answer questions using only the visible browser screenshot. "
    "Give a concise answer and do not describe your reasoning."
)

GROUNDING_SYSTEM_PROMPT = (
    "You precisely locate requested interface elements in screenshots. "
    "Coordinates use a 1000 by 1000 normalized grid while the screenshot keeps its aspect ratio."
)

ACTION_SYSTEM_PROMPT = (
    "You operate a web browser from its screenshot and state. Select exactly one next action. "
    "Coordinates and scroll deltas use a 0 to 100 normalized grid. Return one JSON object with "
    'a concise "thought" and an "action" object. The action object must have a "name" and only '
    "the arguments needed by that action. Do not use Markdown."
)


def qa_prompt(question: str) -> str:
    return f"{question.strip()}\nAnswer using only the screenshot."


def grounding_prompt(description: str) -> str:
    return (
        f"Locate the interface element described as: {description.strip()}\n"
        "Return only a JSON list in this format: "
        '[{"point_2d":[x,y],"label":"element description"}]. '
        "Use integer coordinates from 0 to 1000."
    )


def action_prompt(
    goal: str,
    past_actions: list[dict[str, object]] | None = None,
    page_index: int = 0,
    page_title: str = "Unknown",
    page_url: str = "Unknown",
) -> str:
    lines = ["# GOAL", goal.strip(), "", "# PREVIOUS STEPS"]
    for fallback_index, previous in enumerate(past_actions or [], start=1):
        index = previous.get("index", fallback_index)
        lines.extend(
            [
                f"## Step {index}",
                f"THOUGHT: {str(previous.get('thought') or '').strip()}",
                f"ACTION: {previous.get('action')}",
            ]
        )
    lines.extend(
        [
            "# CURRENTLY ACTIVE PAGE",
            f"Page {page_index}: {page_title} | {page_url}",
            "",
            "# NEXT STEP",
        ]
    )
    return "\n".join(lines).strip()


def inference_messages(task: str, prompt: str, image: object) -> list[dict[str, object]]:
    systems = {
        "qa": QA_SYSTEM_PROMPT,
        "grounding": GROUNDING_SYSTEM_PROMPT,
        "action": ACTION_SYSTEM_PROMPT,
    }
    if task not in systems:
        raise ValueError(f"Unsupported task: {task}")
    system = systems[task]
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def training_conversation(task: str, prompt: str, answer: str) -> tuple[list[dict], list[dict]]:
    systems = {
        "qa": QA_SYSTEM_PROMPT,
        "grounding": GROUNDING_SYSTEM_PROMPT,
        "action": ACTION_SYSTEM_PROMPT,
    }
    if task not in systems:
        raise ValueError(f"Unsupported task: {task}")
    system = systems[task]
    prompt_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    completion = [{"role": "assistant", "content": answer}]
    return prompt_messages, completion
