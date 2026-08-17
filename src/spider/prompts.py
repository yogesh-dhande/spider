from __future__ import annotations

QA_SYSTEM_PROMPT = (
    "You answer questions using only the visible browser screenshot. "
    "Give a concise answer and do not describe your reasoning."
)

GROUNDING_SYSTEM_PROMPT = (
    "You precisely locate requested interface elements in screenshots. "
    "Coordinates use a 1000 by 1000 normalized grid while the screenshot keeps its aspect ratio."
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


def inference_messages(task: str, prompt: str, image: object) -> list[dict[str, object]]:
    system = QA_SYSTEM_PROMPT if task == "qa" else GROUNDING_SYSTEM_PROMPT
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
    system = QA_SYSTEM_PROMPT if task == "qa" else GROUNDING_SYSTEM_PROMPT
    prompt_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    completion = [{"role": "assistant", "content": answer}]
    return prompt_messages, completion
