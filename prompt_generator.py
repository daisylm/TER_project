from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CategorySpec:
    name: str
    description: str
    prompt_count: int
    target_words: int
    complexity: int
    reference_distance: int
    access_pattern: str


@dataclass(frozen=True)
class PromptRecord:
    id: str
    category: str
    title: str
    prompt: str
    context: str
    task: str
    metrics: dict[str, Any]


class PromptGenerator:
    """Generate reproducible prompts for controlled LLM stress tests."""

    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)
        self.seed = seed
        self.category_specs = {
            "baseline": CategorySpec(
                name="baseline",
                description="Medium prompt, sequential reading, simple summarization or explanation.",
                prompt_count=4,
                target_words=650,
                complexity=2,
                reference_distance=1,
                access_pattern="sequential_local",
            ),
            "ram_stress": CategorySpec(
                name="ram_stress",
                description="Large prompt, many sections, repeated long-range and non-local references.",
                prompt_count=4,
                target_words=4200,
                complexity=4,
                reference_distance=9,
                access_pattern="repeated_scattered_long_range",
            ),
            "cpu_stress": CategorySpec(
                name="cpu_stress",
                description="Smaller prompt, dense constraints, multi-phase scenario reasoning.",
                prompt_count=4,
                target_words=300,
                complexity=9,
                reference_distance=4,
                access_pattern="multi_phase_constraint_reasoning",
            ),
        }

    def generate_dataset(self) -> list[PromptRecord]:
        """Return exactly 12 prompts: 4 baseline, 4 RAM stress, 4 CPU stress."""
        records: list[PromptRecord] = []
        records.extend(self._generate_baseline_prompts())
        records.extend(self._generate_ram_stress_prompts())
        records.extend(self._generate_cpu_stress_prompts())
        return records

    def _generate_baseline_prompts(self) -> list[PromptRecord]:
        spec = self.category_specs["baseline"]
        topics = [
            ("edge caching", "Explain how local caches reduce repeated network work."),
            ("batch scheduling", "Summarize how predictable batches improve throughput."),
            ("sensor calibration", "Explain why sequential calibration improves reliability."),
            ("log compaction", "Summarize how compacted logs help later reads."),
        ]

        prompts = []
        for index, (topic, task_hint) in enumerate(topics, start=1):
            sections = [
                self._make_expository_section(topic, f"Part {section}", 155)
                for section in range(1, 5)
            ]
            context = "\n\n".join(sections)
            task = (
                f"{task_hint} Read the sections in order and write a concise "
                "2-3 sentence answer using only the provided context."
            )
            prompts.append(
                self._build_record(
                    prompt_id=f"baseline_{index:02d}",
                    category=spec.name,
                    title=f"Baseline {index}: {topic.title()}",
                    context=context,
                    task=task,
                    spec=spec,
                    extra_metrics={
                        "num_sections": len(sections),
                        "task_type": "summarization_explanation",
                    },
                )
            )
        return prompts

    def _generate_ram_stress_prompts(self) -> list[PromptRecord]:
        spec = self.category_specs["ram_stress"]
        prompts = []

        for index in range(1, spec.prompt_count + 1):
            num_sections = 18 + index * 2
            topic = [
                "distributed inventory audit",
                "regional energy telemetry",
                "multi-site incident report",
                "archival research index",
            ][index - 1]
            anchor_ids = [2, num_sections // 2, num_sections - 2, num_sections]

            sections = []
            for section_id in range(1, num_sections + 1):
                related = self.random.sample(anchor_ids, k=min(2, len(anchor_ids)))
                sections.append(
                    self._make_reference_section(
                        topic=topic,
                        section_id=section_id,
                        target_words=185,
                        related_sections=related,
                    )
                )

            context = "\n\n".join(sections)
            task = (
                f"Use Section {anchor_ids[0]}, Section {anchor_ids[1]}, "
                f"Section {anchor_ids[2]}, and Section {anchor_ids[3]} to answer. "
                "First trace every repeated pointer to those sections. Then compare "
                "the earliest and latest evidence, identify any mismatch between the "
                "middle sections and the final section, and list five specific "
                "cross-section references that support your conclusion. Re-check the "
                "same referenced sections before giving the final answer."
            )
            prompts.append(
                self._build_record(
                    prompt_id=f"ram_stress_{index:02d}",
                    category=spec.name,
                    title=f"RAM Stress {index}: {topic.title()}",
                    context=context,
                    task=task,
                    spec=spec,
                    extra_metrics={
                        "num_sections": num_sections,
                        "referenced_sections": anchor_ids,
                        "max_reference_span": max(anchor_ids) - min(anchor_ids),
                        "repeated_cross_references": True,
                        "cross_reference_passes_required": 2,
                        "task_type": "long_range_comparison",
                    },
                )
            )
        return prompts

    def _generate_cpu_stress_prompts(self) -> list[PromptRecord]:
        spec = self.category_specs["cpu_stress"]
        problem_builders = [
            self._logic_grid_problem,
            self._resource_allocation_problem,
            self._routing_problem,
            self._dependency_schedule_problem,
        ]

        prompts = []
        for index, builder in enumerate(problem_builders, start=1):
            context = self._make_multiphase_context(builder(index), index)
            task = (
                "Phase 1: solve Scenario A, Scenario B, and Scenario C separately. "
                "Phase 2: compare the three results and explain which constraints "
                "changed the outcome. Phase 3: detect contradictions between the "
                "scenario results and the shared rules. Phase 4: validate the final "
                "answer against every constraint and reject any invalid candidate."
            )
            prompts.append(
                self._build_record(
                    prompt_id=f"cpu_stress_{index:02d}",
                    category=spec.name,
                    title=f"CPU Stress {index}: Multi-Step Reasoning",
                    context=context,
                    task=task,
                    spec=spec,
                    extra_metrics={
                        "num_sections": 4,
                        "scenarios": 3,
                        "reasoning_phases": 4,
                        "reasoning_steps_expected": 12 + index,
                        "requires_comparison": True,
                        "requires_contradiction_detection": True,
                        "task_type": "multi_phase_constraint_solving",
                    },
                )
            )
        return prompts

    def _build_record(
        self,
        prompt_id: str,
        category: str,
        title: str,
        context: str,
        task: str,
        spec: CategorySpec,
        extra_metrics: dict[str, Any],
    ) -> PromptRecord:
        prompt = f"{context}\n\n## Task\n\n{task}"
        metrics = {
            "estimated_words": len(prompt.split()),
            "estimated_tokens": int(len(prompt.split()) * 1.3),
            "target_words": spec.target_words,
            "complexity": spec.complexity,
            "reference_distance": spec.reference_distance,
            "access_pattern": spec.access_pattern,
        }
        metrics.update(extra_metrics)
        return PromptRecord(
            id=prompt_id,
            category=category,
            title=title,
            prompt=prompt,
            context=context,
            task=task,
            metrics=metrics,
        )

    def _make_expository_section(self, topic: str, label: str, target_words: int) -> str:
        sentences = [
            f"The {topic} workflow begins with a clear input queue and a stable ordering rule.",
            "Each item is read once, transformed once, and then passed to the next stage.",
            "This simple structure keeps related information close together in the prompt.",
            "The important facts appear near the paragraph that explains their meaning.",
            "A reader can follow the text sequentially without jumping between distant sections.",
            "The result is a medium-size context with predictable access patterns.",
            "The task asks for a summary instead of a search across scattered evidence.",
            "Most of the useful information is repeated in adjacent sentences for clarity.",
        ]
        return self._section_from_sentences(f"## {label}: {topic.title()}", sentences, target_words)

    def _make_reference_section(
        self,
        topic: str,
        section_id: int,
        target_words: int,
        related_sections: list[int],
    ) -> str:
        signal = self.random.choice(["throughput", "latency", "capacity", "error rate", "retention"])
        direction = self.random.choice(["increased", "decreased", "remained stable", "shifted sharply"])
        sentences = [
            f"Section {section_id} records the {topic} measurement for segment {section_id:02d}.",
            f"The primary signal is {signal}, and the observed trend {direction} after the checkpoint.",
            f"The local note references Section {related_sections[0]} for an earlier baseline.",
            f"It also references Section {related_sections[-1]} for a later validation point.",
            f"Repeated pointer A: compare this section with Section {related_sections[0]} before deciding.",
            f"Repeated pointer B: after reading the local value, return to Section {related_sections[-1]}.",
            f"Repeated pointer C: if Section {section_id} conflicts with the task, inspect Section {related_sections[0]} again.",
            "Several filler observations are intentionally included to enlarge the context.",
            "The benchmark should force the model to preserve identifiers while scanning many sections.",
            "Important values are not always adjacent to the task, so lookup distance matters.",
            "The section contains both local observations and remote pointers to other evidence.",
            f"Audit code S{section_id:02d}-{signal.replace(' ', '_')} marks this section as relevant.",
        ]
        return self._section_from_sentences(
            f"## Section {section_id}: {topic.title()}",
            sentences,
            target_words,
        )

    def _make_multiphase_context(self, base_problem: str, variant: int) -> str:
        scenario_a = f"""## Scenario A
Use the shared rules exactly as written. Find the valid result for variant {variant} without adding or removing any constraint."""
        scenario_b = """## Scenario B
Use the shared rules, but add this temporary condition: the first valid candidate in alphabetical order must be rejected. Recompute the result after applying that extra condition."""
        scenario_c = """## Scenario C
Use the shared rules, but add this temporary condition: the result must preserve the earliest possible placement or lowest possible value for the first named item. Recompute and note any conflict with Scenario B."""
        return "\n\n".join([base_problem, scenario_a, scenario_b, scenario_c])

    def _section_from_sentences(self, heading: str, sentences: list[str], target_words: int) -> str:
        words = 0
        body = []
        while words < target_words:
            sentence = self.random.choice(sentences)
            body.append(sentence)
            words += len(sentence.split())
        return f"{heading}\n" + " ".join(body)

    def _logic_grid_problem(self, variant: int) -> str:
        return f"""## Logic Grid Variant {variant}

Four services, Atlas, Beacon, Cipher, and Delta, were deployed on Monday through Thursday.

Rules:
1. Atlas was deployed before Cipher.
2. Beacon was not deployed on Monday or Thursday.
3. Delta was deployed exactly one day after Beacon.
4. Cipher was not deployed immediately after Atlas.
5. The service deployed on Wednesday depends on the service deployed on Monday.
6. No two services share a deployment day.

Question: Determine the deployment day for each service and explain why no other schedule works."""

    def _resource_allocation_problem(self, variant: int) -> str:
        return f"""## Resource Allocation Variant {variant}

A test runner must allocate exactly 120 units across CPU, memory, cache, and I/O experiments.

Rules:
1. CPU receives at least 25 units.
2. Memory receives more units than cache.
3. Cache receives an even number of units.
4. I/O receives exactly 15 fewer units than CPU.
5. If memory receives more than 40 units, cache must receive at least 24 units.
6. CPU and cache together must not exceed 65 units.
7. All allocations are positive integers divisible by 5.

Question: Find every valid allocation and identify the allocation that maximizes memory."""

    def _routing_problem(self, variant: int) -> str:
        return f"""## Routing Variant {variant}

A request must travel from Node A to Node H.

Edges:
A -> B, A -> C
B -> D, B -> E
C -> E, C -> F
D -> G
E -> G, E -> H
F -> C, F -> H
G -> H

Rules:
1. A node may be visited at most once.
2. The route must include exactly one of D or F.
3. If the route uses E, it must not go directly from E to H.
4. The final route must have the fewest valid hops.

Question: List all valid routes, then choose the shortest route and justify the choice."""

    def _dependency_schedule_problem(self, variant: int) -> str:
        return f"""## Dependency Schedule Variant {variant}

Six jobs, A, B, C, D, E, and F, must run in a single sequence.

Rules:
1. A must run before D.
2. B must run before E.
3. C must run after A but before F.
4. D cannot run immediately after A.
5. E must run in position 4 or 5.
6. F must be the final job unless D is final.
7. Exactly one job can run between B and E.

Question: Produce a valid sequence, explain each placement, and show whether the sequence is unique."""

    def save(self, records: list[PromptRecord], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        records_as_dicts = [asdict(record) for record in records]
        grouped = {
            category: [record for record in records_as_dicts if record["category"] == category]
            for category in self.category_specs
        }
        payload = {
            "metadata": {
                "seed": self.seed,
                "total_prompts": len(records),
                "category_order": list(self.category_specs),
                "category_specs": {
                    name: asdict(spec) for name, spec in self.category_specs.items()
                },
            },
            "prompts": records_as_dicts,
            "prompts_by_category": grouped,
        }

        json_path = output_dir / "controlled_prompts.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        prompt_dir = output_dir / "prompt_texts"
        prompt_dir.mkdir(exist_ok=True)
        for record in records:
            (prompt_dir / f"{record.id}.txt").write_text(record.prompt, encoding="utf-8")

        manifest_path = output_dir / "prompt_manifest.json"
        manifest = [
            {
                "id": record.id,
                "category": record.category,
                "title": record.title,
                "file": str(Path("prompt_texts") / f"{record.id}.txt"),
                "metrics": record.metrics,
            }
            for record in records
        ]
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def print_summary(self, records: list[PromptRecord]) -> None:
        print(f"Generated {len(records)} prompts with seed {self.seed}.")
        for category in self.category_specs:
            category_records = [record for record in records if record.category == category]
            total_words = sum(record.metrics["estimated_words"] for record in category_records)
            avg_words = round(total_words / len(category_records))
            print(f"- {category}: {len(category_records)} prompts, avg {avg_words} words")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a controlled LLM prompt dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_prompts"),
        help="Directory for JSON dataset and individual prompt text files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = PromptGenerator(seed=args.seed)
    records = generator.generate_dataset()
    generator.save(records, args.output_dir)
    generator.print_summary(records)
    print(f"Saved dataset to {args.output_dir / 'controlled_prompts.json'}")
    print(f"Saved individual prompts to {args.output_dir / 'prompt_texts'}")


if __name__ == "__main__":
    main()
