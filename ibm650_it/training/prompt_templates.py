SYSTEM_PROMPT = (
    "Compile the following IBM 650 IT program to canonical PIT deck output.\n"
    "Return only the PIT deck, one card per line, with no explanation."
)


def build_prompt(it_text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nUser:\n<IT>\n{it_text.rstrip()}\n</IT>\n\nAssistant:\n<PIT>\n"


def build_few_shot_prompt(
    it_text: str,
    examples: list[dict[str, str]],
) -> str:
    sections = [SYSTEM_PROMPT, ""]
    for example in examples:
        sections.append(f"User:\n<IT>\n{example['source_text'].rstrip()}\n</IT>\n")
        sections.append(f"Assistant:\n<PIT>\n{example['completion'].rstrip()}\n")
    sections.append(f"User:\n<IT>\n{it_text.rstrip()}\n</IT>\n")
    sections.append("Assistant:\n<PIT>\n")
    return "\n".join(sections)
