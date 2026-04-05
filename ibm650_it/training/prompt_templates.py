SYSTEM_PROMPT = (
    "Compile the following IBM 650 IT program to canonical PIT deck output.\n"
    "Return only a <PIT>...</PIT> block containing the PIT deck, one card per line, with no explanation."
)


def build_it_block(it_text: str) -> str:
    return f"<IT>\n{it_text.rstrip()}\n</IT>"


def wrap_pit_completion(pit_text: str) -> str:
    body = pit_text.strip("\n")
    return f"<PIT>\n{body}\n</PIT>"


def ensure_pit_wrapped(pit_text: str) -> str:
    stripped = pit_text.strip()
    if stripped.startswith("<PIT>") and stripped.endswith("</PIT>"):
        return stripped
    return wrap_pit_completion(pit_text)


def build_prompt(it_text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser:\n{build_it_block(it_text)}\n\nAssistant:\n"


def build_chat_messages(it_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_it_block(it_text)},
    ]


def build_few_shot_prompt(
    it_text: str,
    examples: list[dict[str, str]],
) -> str:
    sections = [SYSTEM_PROMPT, ""]
    for example in examples:
        sections.append(f"User:\n{build_it_block(example['source_text'])}\n")
        sections.append(f"Assistant:\n{ensure_pit_wrapped(example['completion'])}\n")
    sections.append(f"User:\n{build_it_block(it_text)}\n")
    sections.append("Assistant:\n")
    return "\n".join(sections)


def build_few_shot_chat_messages(
    it_text: str,
    examples: list[dict[str, str]],
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in examples:
        messages.append({"role": "user", "content": build_it_block(example["source_text"])})
        messages.append({"role": "assistant", "content": ensure_pit_wrapped(example["completion"])})
    messages.append({"role": "user", "content": build_it_block(it_text)})
    return messages
