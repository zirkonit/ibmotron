from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from ibm650_it import REPO_ROOT
from ibm650_it.pit.normalize_pit import canonicalize_pit_file
from ibm650_it.simh.deckio import join_decks, split_tail_cards
from ibm650_it.simh.workdir import JobWorkdir, create_job_workdir, stage_file, stage_text

P1_FOOTPRINT = 265


@dataclass(slots=True)
class TranslationOnlyResult:
    status: str
    upper_acc: int
    workdir: Path
    pit_raw: Path | None
    pit_raw_canonical: Path | None
    console_log: Path
    stdout_log: Path
    print_log: Path

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ReservationSplitResult:
    reservation_cards: Path
    translation_body: Path

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class AssemblyResult:
    status: str
    workdir: Path
    pit_phase2_input_p1: Path
    soap_output: Path | None
    console_log: Path
    stdout_log: Path
    print_log: Path

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SpitBuildResult:
    spit_p1: Path

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class RunResult:
    status: str
    workdir: Path
    spit_p1: Path
    output_deck: Path | None
    console_log: Path
    stdout_log: Path
    print_log: Path

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PipelineResult:
    translate: TranslationOnlyResult
    split: ReservationSplitResult
    assemble: AssemblyResult
    spit: SpitBuildResult
    run: RunResult

    def to_dict(self) -> dict[str, object]:
        return {
            "translate": self.translate.to_dict(),
            "split": self.split.to_dict(),
            "assemble": self.assemble.to_dict(),
            "spit": self.spit.to_dict(),
            "run": self.run.to_dict(),
        }


class SimhRunner:
    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        simh_root: Path | None = None,
        simh_binary: Path | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.simh_root = simh_root.resolve() if simh_root is not None else self.repo_root / "third_party/simh"
        self.simh_binary = simh_binary.resolve() if simh_binary is not None else self.simh_root / "BIN/i650"
        self.sw_root = self.simh_root / "I650/sw"
        self.it_root = self.sw_root / "it"
        self.template_root = self.repo_root / "ibm650_it/simh/ini_templates"

    def _check_prereqs(self) -> None:
        if not self.simh_binary.exists():
            raise FileNotFoundError(f"SIMH binary not found: {self.simh_binary}")
        if not self.it_root.exists():
            raise FileNotFoundError(f"SIMH IT assets not found: {self.it_root}")

    def _stage_it_assets(self, workdir: JobWorkdir) -> None:
        workdir.path("it").mkdir(parents=True, exist_ok=True)
        for name in [
            "it_compiler.dck",
            "soapII.dck",
            "soapII_patch.dck",
            "it_reservation_p1.dck",
            "it_package_p1.dck",
        ]:
            stage_file(self.it_root / name, workdir.path("it", name))

    def _render_template(self, template_name: str, **context: object) -> str:
        text = (self.template_root / template_name).read_text(encoding="utf-8")

        def render_if(match: re.Match[str]) -> str:
            key = match.group(1)
            body = match.group(2)
            return body if context.get(key) else ""

        text = re.sub(r"{%\s*if\s+(\w+)\s*%}(.*?){%\s*endif\s*%}", render_if, text, flags=re.DOTALL)
        text = re.sub(r"{{\s*(\w+)\s*}}", lambda match: str(context[match.group(1)]), text)
        return text

    def _run_batch(
        self,
        workdir: JobWorkdir,
        script_name: str,
        script_text: str,
        *,
        timeout_seconds: int = 30,
    ) -> subprocess.CompletedProcess[str]:
        self._check_prereqs()
        stage_text(script_text, workdir.path(script_name))
        proc = subprocess.run(
            [str(self.simh_binary), script_name],
            cwd=workdir.root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        return proc

    @staticmethod
    def _write_stdout(workdir: JobWorkdir, name: str, proc: subprocess.CompletedProcess[str]) -> Path:
        stdout_path = workdir.path(name)
        stdout_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        return stdout_path

    @staticmethod
    def _parse_accup(stdout_text: str) -> int:
        match = re.search(r"ACCUP:\s*([0-9+\-]+)", stdout_text)
        if not match:
            raise ValueError("failed to parse ACCUP from simulator output")
        value = match.group(1).strip()
        if value.endswith("-"):
            return -int(value[:-1])
        if value.endswith("+"):
            return int(value[:-1])
        return int(value)

    def translate_only(
        self,
        source_deck: Path,
        output_dir: Path,
        *,
        timeout_seconds: int = 30,
    ) -> TranslationOnlyResult:
        workdir = create_job_workdir(output_dir)
        self._stage_it_assets(workdir)
        stage_file(source_deck, workdir.path("source.dck"))
        script = self._render_template(
            "translate_only.ini.j2",
            source_deck="source.dck",
            console_log="translate_console.log",
            print_log="translate_print.txt",
            pit_raw="pit_raw.dck",
        )
        proc = self._run_batch(workdir, "translate_only.ini", script, timeout_seconds=timeout_seconds)
        stdout_log = self._write_stdout(workdir, "translate_stdout.log", proc)
        upper_acc = self._parse_accup(proc.stdout)
        pit_raw = workdir.path("pit_raw.dck")
        canonical_path = workdir.path("pit_raw_canonical.dck") if pit_raw.exists() else None
        if pit_raw.exists():
            canonicalize_pit_file(pit_raw, canonical_path)
        return TranslationOnlyResult(
            status="ok" if upper_acc == 0 and pit_raw.exists() else "compile_error",
            upper_acc=upper_acc,
            workdir=workdir.root,
            pit_raw=pit_raw if pit_raw.exists() else None,
            pit_raw_canonical=canonical_path if canonical_path and canonical_path.exists() else None,
            console_log=workdir.path("translate_console.log"),
            stdout_log=stdout_log,
            print_log=workdir.path("translate_print.txt"),
        )

    def split_reservation_cards(self, pit_raw: Path, output_dir: Path) -> ReservationSplitResult:
        reservation_cards = output_dir / "reservation_cards.dck"
        translation_body = output_dir / "translation_body.dck"
        split_tail_cards(pit_raw, 10, translation_body, reservation_cards)
        return ReservationSplitResult(
            reservation_cards=reservation_cards,
            translation_body=translation_body,
        )

    def build_pit_phase2_input_p1(
        self,
        reservation_cards: Path,
        translation_body: Path,
        output_path: Path,
    ) -> Path:
        package_reservation = translation_body.parent / "it" / "it_reservation_p1.dck"
        if not package_reservation.exists():
            package_reservation = self.it_root / "it_reservation_p1.dck"
        return join_decks([reservation_cards, package_reservation, translation_body], output_path)

    def assemble_pit(
        self,
        pit_phase2_input_p1: Path,
        output_dir: Path,
        *,
        timeout_seconds: int = 30,
    ) -> AssemblyResult:
        workdir = create_job_workdir(output_dir)
        self._stage_it_assets(workdir)
        stage_file(pit_phase2_input_p1, workdir.path("pit_phase2_input_p1.dck"))
        script = self._render_template(
            "assemble_pit.ini.j2",
            pit_input="pit_phase2_input_p1.dck",
            console_log="assemble_console.log",
            print_log="assemble_print.txt",
            soap_output="soap_output.dck",
        )
        proc = self._run_batch(workdir, "assemble_pit.ini", script, timeout_seconds=timeout_seconds)
        stdout_log = self._write_stdout(workdir, "assemble_stdout.log", proc)
        soap_output = workdir.path("soap_output.dck")
        return AssemblyResult(
            status="ok" if soap_output.exists() else "assemble_error",
            workdir=workdir.root,
            pit_phase2_input_p1=workdir.path("pit_phase2_input_p1.dck"),
            soap_output=soap_output if soap_output.exists() else None,
            console_log=workdir.path("assemble_console.log"),
            stdout_log=stdout_log,
            print_log=workdir.path("assemble_print.txt"),
        )

    def build_spit_p1(self, soap_output: Path, output_path: Path) -> SpitBuildResult:
        package_deck = self.it_root / "it_package_p1.dck"
        join_decks([package_deck, soap_output], output_path)
        return SpitBuildResult(spit_p1=output_path)

    def run_spit(
        self,
        spit_p1: Path,
        output_dir: Path,
        *,
        input_deck: Path | None = None,
        step_budget: str = "50M",
        timeout_seconds: int = 30,
    ) -> RunResult:
        workdir = create_job_workdir(output_dir)
        stage_file(spit_p1, workdir.path("spit_p1.dck"))
        if input_deck is not None:
            stage_file(input_deck, workdir.path("input.dck"))
        script = self._render_template(
            "run_spit.ini.j2",
            spit_p1="spit_p1.dck",
            input_deck="input.dck" if input_deck is not None else "",
            console_log="run_console.log",
            print_log="run_print.txt",
            output_deck="run_output.dck",
            step_budget=step_budget,
        )
        proc = self._run_batch(workdir, "run_spit.ini", script, timeout_seconds=timeout_seconds)
        stdout_log = self._write_stdout(workdir, "run_stdout.log", proc)
        output_deck = workdir.path("run_output.dck")
        return RunResult(
            status="ok" if output_deck.exists() else "run_error",
            workdir=workdir.root,
            spit_p1=workdir.path("spit_p1.dck"),
            output_deck=output_deck if output_deck.exists() else None,
            console_log=workdir.path("run_console.log"),
            stdout_log=stdout_log,
            print_log=workdir.path("run_print.txt"),
        )

    def reference_pipeline(
        self,
        *,
        source_deck: Path,
        output_dir: Path,
        input_deck: Path | None = None,
        step_budget: str = "50M",
        timeout_seconds: int = 30,
    ) -> PipelineResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        translate = self.translate_only(source_deck, output_dir / "translate", timeout_seconds=timeout_seconds)
        if translate.status != "ok" or translate.pit_raw is None:
            raise RuntimeError(f"translation failed with status={translate.status}, upper_acc={translate.upper_acc}")
        split = self.split_reservation_cards(translate.pit_raw, output_dir / "translate")
        pit_phase2 = self.build_pit_phase2_input_p1(
            split.reservation_cards,
            split.translation_body,
            output_dir / "assemble" / "pit_phase2_input_p1.dck",
        )
        assemble = self.assemble_pit(pit_phase2, output_dir / "assemble", timeout_seconds=timeout_seconds)
        if assemble.status != "ok" or assemble.soap_output is None:
            raise RuntimeError(f"assembly failed with status={assemble.status}")
        spit = self.build_spit_p1(assemble.soap_output, output_dir / "run" / "spit_p1.dck")
        run = self.run_spit(
            spit.spit_p1,
            output_dir / "run",
            input_deck=input_deck,
            step_budget=step_budget,
            timeout_seconds=timeout_seconds,
        )
        return PipelineResult(
            translate=translate,
            split=split,
            assemble=assemble,
            spit=spit,
            run=run,
        )

    def run_shipped_run_it(
        self,
        *,
        source_deck: Path,
        output_dir: Path,
        input_deck: Path | None = None,
    ) -> dict[str, str]:
        workdir = create_job_workdir(output_dir)
        self._stage_it_assets(workdir)
        stage_file(self.sw_root / "run_it.ini", workdir.path("run_it.ini"))
        stage_file(source_deck, workdir.path("source.dck"))
        if input_deck is not None:
            stage_file(input_deck, workdir.path("input.dck"))
        input_arg = "input.dck" if input_deck is not None else '""'
        stage_text(
            "\n".join(
                [
                    f"do run_it.ini source.dck {input_arg} baseline_out.dck",
                    "exit",
                    "",
                ]
            ),
            workdir.path("run_baseline.ini"),
        )
        proc = subprocess.run(
            [str(self.simh_binary), "run_baseline.ini"],
            cwd=workdir.root,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        stdout_path = workdir.path("baseline_stdout.log")
        stdout_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        baseline_out = workdir.path("baseline_out.dck")
        return {
            "stdout_log": str(stdout_path),
            "output_deck": str(baseline_out),
        }
