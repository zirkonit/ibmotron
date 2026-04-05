SYSTEM_PROMPT = (
    "Compile the following IBM 650 IT program to canonical PIT deck output.\n"
    "Return only a <PIT>...</PIT> block containing the PIT deck, one card per line, with no explanation."
)


def wrap_pit_completion(pit_text: str) -> str:
    body = pit_text.strip("\n")
    return f"<PIT>\n{body}\n</PIT>"


def ensure_pit_wrapped(pit_text: str) -> str:
    stripped = pit_text.strip()
    if stripped.startswith("<PIT>") and stripped.endswith("</PIT>"):
        return stripped
    return wrap_pit_completion(pit_text)


def build_prompt(it_text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser:\n<IT>\n{it_text.rstrip()}\n</IT>\n\nAssistant:\n"


def build_few_shot_prompt(
    it_text: str,
    examples: list[dict[str, str]],
) -> str:
    sections = [SYSTEM_PROMPT, ""]
    for example in examples:
        sections.append(f"User:\n<IT>\n{example['source_text'].rstrip()}\n</IT>\n")
        sections.append(f"Assistant:\n{ensure_pit_wrapped(example['completion'])}\n")
    sections.append(f"User:\n<IT>\n{it_text.rstrip()}\n</IT>\n")
    sections.append("Assistant:\n")
    return "\n".join(sections)
