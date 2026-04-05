from __future__ import annotations

from pathlib import Path

from ibm650_it.simh.runner import SimhRunner


def assemble_candidate_pit(
    *,
    runner: SimhRunner,
    reservation_cards: Path,
    translation_body: Path,
    output_dir: Path,
) -> dict[str, object]:
    pit_phase2 = runner.build_pit_phase2_input_p1(
        reservation_cards,
        translation_body,
        output_dir / "pit_phase2_input_p1.dck",
    )
    result = runner.assemble_pit(pit_phase2, output_dir)
    return result.to_dict()
