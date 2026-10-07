#!/usr/bin/env python3
"""Collect paired outputs on Colab and complete REPORT.md's qualitative section.

Run from the repo after copying this script and the completed report to Colab:
    python scripts/complete_qualitative.py

Keeps the frozen baselines, verdict, adapters and original qualitative.json intact.
Only updates the marked report section if rerun scores match the saved evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- qualitative:start -->", "<!-- qualitative:end -->"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def cell(value):
    return html.escape(str(value)).replace("|", "\\|").replace("\n", "<br>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-update-report", action="store_true",
                        help="Save paired JSON only; keep REPORT.md unchanged.")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from labkit import evaluate as ev, generate
    from labkit.config import get_tier

    frozen = load_json(ROOT / "results/baselines_frozen.json")
    verdict = load_json(ROOT / "results/verdict.json")
    target = load_jsonl(ROOT / "data/eval_target.jsonl")
    regression = load_jsonl(ROOT / "data/eval_regression.jsonl")
    tier = get_tier(frozen["tier"])
    if frozen.get("smoke_mode") or frozen["n_target"] != len(target) or (
            frozen["n_regression"] != len(regression)):
        raise SystemExit("Expected the completed full evaluation, not smoke mode.")
    if tier.model_id != frozen["model"]:
        raise SystemExit("BASE_MODEL differs from the frozen model; restore it first.")
    sha = hashlib.sha256(generate.OPTIMIZED_PROMPT.encode()).hexdigest()[:16]
    if sha != frozen["optimized_prompt_sha"]:
        raise SystemExit("Optimized prompt differs from the frozen prompt.")
    checksums = load_json(ROOT / "data/checksums.json")
    for name in ("eval_target.jsonl", "eval_regression.jsonl"):
        raw = (ROOT / "data" / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest()[:16] != checksums[name]:
            raise SystemExit(f"{name} differs from the reference corpus.")
    report_path = ROOT / "submission/REPORT.md"
    report_text = report_path.read_text(encoding="utf-8")
    if not args.no_update_report and (
            report_text.count(START) != 1 or report_text.count(END) != 1
            or report_text.index(START) >= report_text.index(END)):
        raise SystemExit("Copy the completed REPORT.md with its markers to Colab first.")

    import torch
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise SystemExit("Run this script on the Colab GPU holding the four adapters.")

    predictions = {}
    for key, system in (("baseline_b", generate.OPTIMIZED_PROMPT),
                        ("fine_tune", generate.NAIVE_PROMPT)):
        model, tok = generate.load_base(tier)
        if key == "fine_tune":
            model = PeftModel.from_pretrained(model, str(ROOT / "adapters/correct"))
        model.eval()
        tp, _ = generate.generate_batch(model, tok, [r["input"] for r in target],
                                        system=system, label=f"paired/{key}/target")
        rp, _ = generate.generate_batch(
            model, tok, [r["instruction"] for r in regression], system=None,
            max_new_tokens=96, label=f"paired/{key}/regression")
        predictions[key] = {"target": tp, "regression": rp}
        del model, tok
        generate.free_memory()

    pairs, scores = [], {}
    for group, source in (("target", target), ("regression", regression)):
        totals = {"baseline_b": 0.0, "fine_tune": 0.0}
        for i, row in enumerate(source):
            item = {"group": group, "i": i,
                    "input": row["input"] if group == "target" else row["instruction"],
                    "reference": row["label"] if group == "target" else row["keywords"]}
            for key in totals:
                pred = predictions[key][group][i]
                score = (ev.triage_field_accuracy(pred, row["label"]) if group == "target"
                         else ev.keyword_recall(pred, row["keywords"]))
                item[key] = pred
                item[key + "_score"] = score
                totals[key] += score
            delta = item["fine_tune_score"] - item["baseline_b_score"]
            item["outcome"] = "FT thua" if delta < 0 else "FT thắng" if delta > 0 else "Hòa"
            pairs.append(item)
        for key, total in totals.items():
            scores[f"{key}/{group}"] = total / len(source)

    ft_saved = next(row for row in verdict["comparison"] if row["run"].startswith("(c)"))
    expected = {f"{key}/{group}": saved[group]
                for key, saved in (("baseline_b", frozen["baseline_b"]),
                                   ("fine_tune", ft_saved))
                for group in ("target", "regression")}
    consistent = all(math.isclose(scores[k], v, abs_tol=0.000051, rel_tol=0)
                     for k, v in expected.items())
    output = ROOT / "results/qualitative_pairs.json"
    output.write_text(json.dumps({"model": tier.model_id, "optimized_prompt_sha": sha,
                                  "scores": scores, "expected_scores": expected,
                                  "matches_saved_scores": consistent, "pairs": pairs},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved complete outputs: {output}", flush=True)
    if not consistent:
        raise SystemExit("Rerun scores differ from saved evaluation; REPORT.md unchanged. "
                         "Inspect qualitative_pairs.json before using these examples.")
    if args.no_update_report:
        return 0

    losses = [p for p in pairs if p["outcome"] == "FT thua"]
    wins = [p for p in pairs if p["outcome"] == "FT thắng"]
    selected = losses[:2] + wins[:2]
    for item in pairs:
        if len(selected) >= 5:
            break
        if item not in selected:
            selected.append(item)
    lines = ["Nguồn bổ sung: `results/qualitative_pairs.json`. Output được sinh lại bằng "
             "cùng model, prompt, greedy decode, batch size và giới hạn token của NB2/NB5. "
             "Điểm target/regression của cả hai model khớp artifact đã lưu trong sai số "
             "làm tròn. Đây là bằng chứng bổ sung sau thí nghiệm; không thay baseline, "
             "verdict hoặc adapter.", "",
             "Chỉ số i đếm từ 0 trong từng tập. Với target, tham chiếu là JSON nhãn; "
             "với regression, tham chiếu là danh sách từ khóa của scorer, không phải "
             "câu trả lời mẫu. Thắng/thua được xác định bằng điểm scorer trên cùng mẫu.", "",
             "| Nhóm / i | Đầu vào | Tham chiếu | Baseline (b) | Fine-tune (c) | Điểm b / c | Kết quả |",
             "|---|---|---|---|---|---|---|"]
    for p in selected:
        values = [f"{p['group']} / {p['i']}", p["input"],
                  json.dumps(p["reference"], ensure_ascii=False), p["baseline_b"], p["fine_tune"],
                  f"{p['baseline_b_score']:.4f} / {p['fine_tune_score']:.4f}", p["outcome"]]
        lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    lines += ["", f"Trên 65 mẫu của hai nhóm có {len(wins)} ca FT thắng, {len(losses)} ca "
              f"FT thua và {len(pairs)-len(wins)-len(losses)} ca hòa theo scorer. "
              "Các ca target và regression được phân biệt rõ, không cộng điểm hai nhóm "
              "thành một thang chung."]
    if len(losses) < 2:
        lines += ["", "Không tìm đủ hai ca FT thua trong dữ liệu đối chiếu; "
                  "không dựng thêm ví dụ để đạt yêu cầu rubric."]
    lines += ["", "Ở artifact target ban đầu, sáu lỗi i=3,5,12,39,41,46 đều có nhãn "
              "urgency=thap nhưng preview dự đoán trung_binh với cụm 'Khi nào tiện'. "
              "Có 44/50 ticket đúng hoàn toàn và 194/200 trường đúng. Phân tích lỗi "
              "này khác với việc xác định thắng/thua baseline ở bảng đối chiếu trên."]
    before, rest = report_text.split(START, 1)
    _, after = rest.split(END, 1)
    report_path.write_text(before + START + "\n\n" + "\n".join(lines) + "\n\n" + END + after,
                           encoding="utf-8")
    print(f"Updated {report_path}; download it and qualitative_pairs.json from Colab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
