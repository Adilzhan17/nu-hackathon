"""Human review of train events. Source annotations are strictly read-only.

The append-only journal is authoritative; current.json is a disposable snapshot.
Confirmations describe visible actions, never automatically become violation labels.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid


ACTIONS = [
    ("unclear_action", "Действие не удалось различить"),
    ("person_enters", "Человек входит"), ("person_leaves", "Человек выходит"),
    ("leave_workstation", "Сотрудник уходит от рабочего места"),
    ("return_to_workstation", "Сотрудник возвращается"),
    ("cash_received", "Получает деньги"), ("cash_transferred", "Передаёт / возвращает деньги"),
    ("cash_placed", "Кладёт деньги на поверхность"), ("cash_removed", "Убирает деньги с поверхности"),
    ("cash_visible", "Деньги остаются видны"), ("cash_unpacking", "Распаковывает деньги"),
    ("cash_bundling", "Собирает / разделяет пачки денег"),
    ("cash_counter_loading", "Загружает деньги в счётную машинку"),
    ("cash_counting", "Пересчитывает деньги"),
    ("cash_detector_check", "Проверяет деньги детектором"),
    ("cash_stored", "Убирает деньги в сейф / хранилище"),
    ("document_transfer", "Передаёт / возвращает документ"),
    ("document_handling", "Держит / перекладывает документ"),
    ("document_checked", "Рассматривает документ"),
    ("document_stored", "Убирает документ"),
    ("key_handling", "Держит / использует ключ"), ("key_left", "Оставляет ключ"),
    ("stamp_handling", "Держит / использует печать"), ("stamp_stored", "Убирает печать"),
    ("safe_opening", "Открывает сейф"), ("safe_closing", "Закрывает дверь сейфа"),
    ("safe_locking", "Запирает сейф ключом"), ("safe_open_visible", "Дверь сейфа видимо открыта"),
    ("computer_work", "Работает за компьютером"), ("phone_used", "Использует телефон"),
    ("phone_put_away", "Убирает телефон"), ("drink_at_workstation", "Пьёт / ставит напиток"),
    ("object_handling", "Манипулирует неразличимым предметом"),
    ("other_observed_action", "Другое видимое действие — опишу"),
]
OBJECTS = [
    ("cash", "Деньги"), ("document", "Документ / удостоверение"), ("key", "Ключ"),
    ("stamp", "Печать"), ("safe", "Сейф"), ("cash_counter", "Счётная машинка"),
    ("banknote_detector", "Детектор банкнот"), ("computer", "Компьютер / ноутбук"),
    ("phone", "Телефон"), ("drink", "Напиток / стакан"), ("desk", "Стол"),
    ("door", "Дверь"), ("person", "Человек"), ("unidentified_item", "Неразличимый предмет"),
]
ACTION_IDS = {x[0] for x in ACTIONS}
OBJECT_IDS = {x[0] for x in OBJECTS}
VIOLATION_TYPES = [
    ("previous_cash_left", "На столе остались деньги предыдущей операции"),
    ("leave_during_service", "Сотрудник ушёл во время обслуживания"),
    ("cash_verification_missing", "Возможно, пропущена проверка банкнот"),
    ("identity_check_missing", "Возможно, пропущена проверка удостоверения"),
    ("personal_item_present", "Личный предмет на рабочем месте"),
    ("key_custody_risk", "Возможное нарушение хранения ключей"),
    ("other_visible", "Другое видимое событие — опишу"),
]
ACTION_OBJECTS = {
    **{k: {"cash"} for k in ACTION_IDS if k.startswith("cash_")},
    **{k: {"document"} for k in ACTION_IDS if k.startswith("document_")},
    **{k: {"safe"} for k in ACTION_IDS if k.startswith("safe_")},
    **{k: {"key"} for k in ACTION_IDS if k.startswith("key_")},
    **{k: {"stamp"} for k in ACTION_IDS if k.startswith("stamp_")},
    **{k: {"phone"} for k in ACTION_IDS if k.startswith("phone_")},
    "computer_work": {"computer"}, "drink_at_workstation": {"drink"},
    "object_handling": {"unidentified_item"},
}
ALIASES = {
    "visitor_enters": "person_enters", "second_visitor_enters": "person_enters",
    "visitor_leaves": "person_leaves", "room_exit": "person_leaves",
    "person_leaves_view": "person_leaves", "document_presented_and_inspected": "document_checked",
    "document_presented": "document_transfer", "document_inspection": "document_checked",
    "document_returned": "document_transfer", "documents_stored": "document_stored",
    "cash_counter_operation": "cash_counting", "counted_cash_handling": "cash_bundling",
    "cash_bundle_split": "cash_bundling", "partial_cash_return": "cash_transferred",
    "cash_presented": "cash_transferred", "cash_received_and_unpacked": "cash_received",
    "cash_placed_on_desk": "cash_placed", "counted_cash_bundling": "cash_bundling",
    "cash_preparation_and_loading": "cash_counter_loading", "safe_door_closing": "safe_closing",
    "safe_closing_and_locking": "safe_locking", "safe_locking_keys_taken": "safe_locking",
    "safe_key_manipulation": "key_handling", "stamps_stored": "stamp_stored",
    "unclear_handheld_item": "object_handling",
}
REASON_LABELS = {
    "user_reference_not_recovered_by_model": "Модель не восстановила пользовательское описание",
    "possible_object_disagreement": "Расхождение в определении предмета",
    "possible_temporal_disagreement": "Расхождение временных границ",
    "model_visibility_or_timing_uncertainty": "Неясный предмет или время действия",
    "unresolved_visual_interval": "Неоднозначный фрагмент",
    "organizer_incident_type_unresolved": "Тип события не подтверждён",
    "low_model_self_rating": "Модель не уверена в описании",
}


class ReviewError(ValueError):
    def __init__(self, message, code="invalid_review", status_code=400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ReviewError(f"{name}: требуется конечное число")
    return float(value)


class HumanReview:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.report = self.root / "data/autolabel/detailed_train_report"
        self.store = self.root / "data/human_review"
        self.store.mkdir(parents=True, exist_ok=True)
        self.journal = self.store / "decisions.jsonl"
        self.snapshot = self.store / "current.json"

    @contextlib.contextmanager
    def _locked(self):
        with (self.store / ".lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _sources(self):
        paths = [self.root / "data/training/manifest.csv"] + [self.report / name for name in (
            "video_timelines.jsonl", "user_reference_alignment.jsonl", "moments.jsonl", "review_queue.jsonl")]
        contents = {p: p.read_bytes() if p.exists() else b"" for p in paths}
        fingerprint = hashlib.sha256(_json({str(p.relative_to(self.root)): hashlib.sha256(raw).hexdigest()
                                           for p, raw in contents.items()}).encode()).hexdigest()
        return contents, fingerprint

    def _tasks(self):
        import csv
        import io
        contents, fingerprint = self._sources()
        def rows(name):
            return [json.loads(s) for s in contents[self.report / name].decode("utf-8").splitlines() if s.strip()]
        groups = {}
        for row in csv.DictReader(io.StringIO(contents[self.root / "data/training/manifest.csv"].decode("utf-8"))):
            key = row.get("group_id")
            entry = groups.setdefault(key, {"splits": set(), "videos": set(), "hashes": set()})
            entry["splits"].add(row.get("split"))
            entry["videos"].add(row.get("video"))
            entry["hashes"].add(row.get("sha256", ""))
        timelines = {r["group_id"]: r for r in rows("video_timelines.jsonl")}
        tasks = []

        def add(row, kind, source_id, priority, description, action, objects, reason):
            group = row.get("group_id")
            allowed = groups.get(group)
            if not allowed or allowed["splits"] != {"train"} or len(allowed["hashes"]) != 1 or row.get("video") not in allowed["videos"]:
                return
            if row.get("split", "train") != "train":
                return
            if "quarantin" in str(row.get("source_status", "")).lower() or "quarantin" in str(row.get("status", "")).lower():
                return
            timeline = timelines.get(group)
            if not timeline:
                return
            duration = float(timeline["duration_sec"])
            start, end = float(row["start_sec"]), float(row["end_sec"])
            if not all(math.isfinite(v) for v in (duration, start, end)) or duration <= 0 or start < 0 or end > duration + .05 or end <= start:
                return
            end = min(duration, end)
            action = ALIASES.get(action, action)
            if action not in ACTION_IDS:
                action = "other_observed_action"
            objects = sorted(set(objects) & OBJECT_IDS)
            if not objects:
                objects = sorted(ACTION_OBJECTS.get(action, set()))
            preview = timeline.get("preview", "")
            expected_preview = f"previews/{group}.mp4"
            if preview != expected_preview:
                return
            evidence = [float(t) for t in row.get("evidence_timestamps", [])
                        if isinstance(t, (int, float)) and not isinstance(t, bool)
                        and math.isfinite(t) and start <= t <= end]
            identifier = "hr_" + hashlib.sha256(f"{kind}:{group}:{source_id}".encode()).hexdigest()[:20]
            tasks.append({
                "id": identifier, "video": row["video"], "group_id": group, "split": "train",
                "video_sha256": next(iter(allowed["hashes"])),
                "duration_sec": duration, "preview_url": f"/autolabel/detailed_train_report/{preview}",
                "preview_offset_sec": float(timeline.get("preview_metadata", {}).get("clip_first_source_sec", 0)),
                "full_preview_url": f"/autolabel/detailed_train_report/{preview}",
                "full_preview_offset_sec": float(timeline.get("preview_metadata", {}).get("clip_first_source_sec", 0)),
                "start_sec": start, "end_sec": end,
                "context_start_sec": max(0, min(start, float(row.get("context_start_sec", start - 5)))),
                "context_end_sec": min(duration, max(end, float(row.get("context_end_sec", end + 5)))),
                "suggested_action": action, "objects": objects, "description": str(description or ""),
                "reason": reason, "source_kind": kind, "source_id": str(source_id),
                "evidence_timestamps": evidence, "priority": priority, "source_fingerprint": fingerprint,
            })

        for row in rows("user_reference_alignment.jsonl"):
            missing = row.get("model_support_status") == "not_recovered_by_model"
            add(row, "user_reference", row["reference_id"], 100 if missing else 70,
                row.get("source_observation"), row.get("source_action", ""), [],
                "Описание пользователя не найдено моделью; проверьте по видео" if missing else
                "Проверьте пользовательское описание и точные границы")
        for row in rows("moments.jsonl"):
            conflicts = bool(row.get("conflict_ids"))
            add(row, "model_moment", row["moment_id"], 110 if conflicts else 20,
                "Предположение модели: " + dict(ACTIONS).get(row.get("observed_action"), "видимое действие"),
                row.get("observed_action", ""), row.get("objects", []),
                "Конфликт разметки: проверьте действие и предмет" if conflicts else
                "Неподтверждённое предложение модели")
        for row in rows("review_queue.jsonl"):
            # A quarantined source must not leak through a merged review task.
            observations = row.get("observations", [])
            if "quarantined_source" in row.get("reasons", []) or any(
                "quarantin" in str(o.get("status", "")).lower() for o in observations):
                continue
            conflict = bool(row.get("conflict_ids"))
            reasons = "; ".join(REASON_LABELS.get(r, r) for r in row.get("reasons", []))
            description = " / ".join(str(o.get("observation", "")) for o in observations)
            action = observations[0].get("action", "") if observations else ""
            add(row, "compact_review", row["review_id"], 120 if conflict else 50,
                description, action, [], reasons or "Посмотрите эпизод в контексте")
        known_ids = {t["id"] for t in tasks}
        for task in _read_rows(self.store / "manual_tasks.jsonl"):
            if task.get("source_fingerprint") == fingerprint and task.get("base_task_id") in known_ids:
                tasks.append(task)
                known_ids.add(task["id"])
        tasks.sort(key=lambda r: (-r["priority"], r["group_id"], r["start_sec"], r["id"]))
        return tasks, fingerprint

    def _state(self):
        decisions, revision = {}, 0
        # Replay avoids losing a successfully fsynced decision after a crash before snapshot replace.
        for row in _read_rows(self.journal):
            if row.get("revision") != revision + 1:
                raise ReviewError("Журнал разметки повреждён: неверная ревизия", "journal_corrupt", 500)
            old = decisions.get(row["task_id"])
            if row.get("version") != (old["version"] if old else 0) + 1:
                raise ReviewError("Журнал разметки повреждён: неверная версия", "journal_corrupt", 500)
            revision = row["revision"]
            decisions[row["task_id"]] = row
        return decisions, revision

    @staticmethod
    def _recommended(tasks, decisions):
        pending = [t for t in tasks if t["id"] not in decisions or
                   decisions[t["id"]].get("source_fingerprint") != t["source_fingerprint"]]
        current = {t["id"]: t for t in tasks}
        reviewed = [d for task_id, d in decisions.items() if task_id in current
                    and d.get("source_fingerprint") == current[task_id]["source_fingerprint"]
                    and d.get("decision") in {"accept", "reject"}]
        selected = []
        for task in pending:
            overlaps = False
            for previous in selected + reviewed:
                if previous["group_id"] != task["group_id"]:
                    continue
                intersection = min(previous["end_sec"], task["end_sec"]) - max(previous["start_sec"], task["start_sec"])
                if intersection > .5 * min(previous["end_sec"] - previous["start_sec"], task["end_sec"] - task["start_sec"]):
                    overlaps = True
                    break
            if not overlaps:
                selected.append(task)
            if len(selected) == 20:
                break
        return [t["id"] for t in selected]

    @staticmethod
    def _counts(tasks, decisions):
        valid = {t["id"]: decisions[t["id"]] for t in tasks if t["id"] in decisions
                 and decisions[t["id"]].get("source_fingerprint") == t["source_fingerprint"]}
        counts = {"total": len(tasks), "reviewed": len(valid), "pending": len(tasks) - len(valid),
                  "accept": 0, "reject": 0, "needs_context": 0, "stale": 0,
                  "confirmed_actions": 0, "confirmed_violations": 0}
        counts["stale"] = sum(t["id"] in decisions and t["id"] not in valid for t in tasks)
        for d in valid.values():
            counts[d["decision"]] += 1
        counts["confirmed_actions"] = counts["accept"]
        counts["priority_pending"] = len(HumanReview._recommended(tasks, decisions))
        return counts

    def catalog(self):
        with self._locked():
            tasks, fingerprint = self._tasks()
            decisions, revision = self._state()
            return {"schema_version": 1, "tasks": tasks, "decisions": decisions, "revision": revision,
                    "source_fingerprint": fingerprint, "actions": [{"id": k, "label": v,
                        "required_objects": sorted(ACTION_OBJECTS.get(k, set()))} for k, v in ACTIONS],
                    "objects": [{"id": k, "label": v} for k, v in OBJECTS],
                    "violation_types": [{"id": k, "label": v} for k, v in VIOLATION_TYPES],
                    "counts": self._counts(tasks, decisions), "recommended_task_ids": self._recommended(tasks, decisions),
                    "label_policy": "Подтверждается видимое действие. Отказ и сомнение не являются отрицательной разметкой нарушения."}

    def save(self, payload):
        if not isinstance(payload, dict):
            raise ReviewError("Ожидается JSON-объект")
        with self._locked():
            tasks, fingerprint = self._tasks()
            task = next((t for t in tasks if t["id"] == payload.get("task_id")), None)
            if not task:
                raise ReviewError("Эпизод не найден в текущей train-очереди", "task_not_found", 404)
            if payload.get("source_fingerprint") != fingerprint:
                raise ReviewError("Исходный отчёт обновился. Перезагрузите эпизод", "source_changed", 409)
            decisions, revision = self._state()
            previous = decisions.get(task["id"])
            expected = payload.get("expected_version")
            if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
                raise ReviewError("expected_version: требуется целое число >= 0")
            if expected != (previous["version"] if previous else 0):
                raise ReviewError("Этот эпизод уже изменён другим участником. Перезагрузите его", "stale_version", 409)
            decision = payload.get("decision")
            if decision not in {"accept", "reject", "needs_context"}:
                raise ReviewError("Неизвестное решение")
            start = _number(payload.get("start_sec", task["start_sec"]), "start_sec")
            end = _number(payload.get("end_sec", task["end_sec"]), "end_sec")
            if not 0 <= start < end <= task["duration_sec"]:
                raise ReviewError("Границы должны находиться внутри видео, начало раньше конца")
            visibility = payload.get("visibility", "unclear")
            if visibility not in {"clear", "partial", "unclear"}:
                raise ReviewError("Неизвестная видимость")
            if visibility == "unclear" and decision != "needs_context":
                raise ReviewError("Если действие не видно, сохраните «Неясно / нужен контекст»")
            assessment = payload.get("event_assessment", "not_assessed")
            if assessment not in {"not_assessed", "normal_action", "possible_violation"}:
                raise ReviewError("Неизвестная оценка события")
            action = payload.get("observed_action") or None
            if action is not None and action not in ACTION_IDS:
                raise ReviewError("Выберите действие из списка")
            objects = payload.get("objects", [])
            if not isinstance(objects, list) or any(not isinstance(o, str) or o not in OBJECT_IDS for o in objects):
                raise ReviewError("Выберите предметы из списка")
            objects = sorted(set(objects))
            reviewer = payload.get("reviewer", "")
            note = payload.get("note", "")
            violation_type = payload.get("violation_type") or None
            for name, value, maximum in [("reviewer", reviewer, 100), ("note", note, 4000), ("violation_type", violation_type or "", 160)]:
                if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
                    raise ReviewError(f"{name}: недопустимый текст или превышена длина {maximum}")
            if not reviewer.strip():
                raise ReviewError("Укажите имя или инициалы участника")
            evidence = payload.get("evidence_time_sec")
            if evidence is not None:
                evidence = _number(evidence, "evidence_time_sec")
                if not start <= evidence <= end:
                    raise ReviewError("Кадр-доказательство должен быть внутри отмеченного действия")
            if decision == "accept":
                if action in {None, "unclear_action"} or evidence is None:
                    raise ReviewError("Подтверждение требует действия, видимого доказательства и его таймкода")
                required = ACTION_OBJECTS.get(action, set())
                if required and not required.issubset(objects):
                    raise ReviewError("Предмет не соответствует действию: добавьте " + ", ".join(dict(OBJECTS)[k] for k in sorted(required)))
                if action == "other_observed_action" and not note.strip():
                    raise ReviewError("Опишите другое видимое действие в комментарии")
                if assessment == "possible_violation" and not violation_type:
                    raise ReviewError("Для возможного нарушения укажите предполагаемый тип")
                if assessment == "possible_violation" and violation_type not in {k for k, _ in VIOLATION_TYPES}:
                    raise ReviewError("Выберите предполагаемый тип нарушения из списка")
                if assessment == "possible_violation" and violation_type == "other_visible" and not note.strip():
                    raise ReviewError("Опишите другое возможное нарушение в комментарии")
                if assessment != "possible_violation" and violation_type:
                    raise ReviewError("Тип нарушения допустим только для оценки possible_violation")
            else:
                # Neither rejecting a proposal nor uncertainty creates a negative event/violation label.
                assessment, violation_type = "not_assessed", None
            now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
            record = {"schema_version": 1, "task_id": task["id"], "version": expected + 1,
                      "revision": revision + 1, "created_at": previous["created_at"] if previous else now,
                      "updated_at": now, "reviewer": reviewer.strip(), "decision": decision,
                      "video": task["video"], "group_id": task["group_id"], "split": "train",
                      "video_sha256": task["video_sha256"],
                      "source_kind": task["source_kind"], "source_id": task["source_id"],
                      "source_fingerprint": fingerprint, "start_sec": start, "end_sec": end,
                      "observed_action": action, "objects": objects, "note": note.strip(),
                      "evidence_time_sec": evidence, "visibility": visibility,
                      "event_assessment": assessment, "violation_type": violation_type,
                      "action_confirmed": decision == "accept", "violation_confirmed": False,
                      "assessment_authority": "human_review", "normal_action_is_whole_video_label": False,
                      "eligible_for_action_training": decision == "accept",
                      "eligible_for_violation_training": False,
                      "rejected_proposal_is_negative_label": False}
            with self.journal.open("a", encoding="utf-8") as handle:
                handle.write(_json(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            decisions[task["id"]] = record
            self._write_snapshot({"schema_version": 1, "revision": revision + 1, "decisions": decisions})
            return {"decision": record, "revision": revision + 1, "counts": self._counts(tasks, decisions)}

    def add_task(self, payload):
        """Add a missed action, restricted to a video already present in the train queue."""
        if not isinstance(payload, dict):
            raise ReviewError("Ожидается JSON-объект")
        with self._locked():
            tasks, fingerprint = self._tasks()
            base = next((t for t in tasks if t["id"] == payload.get("base_task_id")), None)
            if base is None:
                raise ReviewError("Исходный train-эпизод не найден", "task_not_found", 404)
            if payload.get("source_fingerprint") != fingerprint:
                raise ReviewError("Исходный отчёт обновился. Перезагрузите страницу", "source_changed", 409)
            reviewer, description = payload.get("reviewer", ""), payload.get("description", "")
            if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 100 or "\x00" in reviewer:
                raise ReviewError("Укажите имя или инициалы участника (до 100 символов)")
            if not isinstance(description, str) or len(description) > 4000 or "\x00" in description:
                raise ReviewError("Описание: максимум 4000 символов")
            start = _number(payload.get("start_sec"), "start_sec")
            end = _number(payload.get("end_sec"), "end_sec")
            if not 0 <= start < end <= base["duration_sec"]:
                raise ReviewError("Границы нового действия должны находиться внутри видео")
            identifier = "hr_manual_" + uuid.uuid4().hex
            task = dict(base, id=identifier, source_kind="human_added", source_id=identifier,
                        base_task_id=base["id"], start_sec=start, end_sec=end,
                        context_start_sec=max(0, start - 5), context_end_sec=min(base["duration_sec"], end + 5),
                        suggested_action="unclear_action", objects=[], evidence_timestamps=[],
                        description=description.strip(), reason="Пропущенное действие добавлено участником; ещё не подтверждено",
                        priority=130, added_by=reviewer.strip(),
                        added_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"))
            with (self.store / "manual_tasks.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(_json(task) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            tasks.append(task)
            decisions, revision = self._state()
            return {"task": task, "revision": revision, "counts": self._counts(tasks, decisions)}

    def _write_snapshot(self, value):
        fd, filename = tempfile.mkstemp(prefix=".current-", suffix=".json", dir=self.store)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(_json(value) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(filename, self.snapshot)
            directory = os.open(self.store, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def export_jsonl(self):
        with self._locked():
            tasks, fingerprint = self._tasks()
            current_ids = {t["id"] for t in tasks}
            decisions, _ = self._state()
            output = []
            for task_id, decision in sorted(decisions.items()):
                stale = task_id not in current_ids or decision.get("source_fingerprint") != fingerprint
                item = dict(decision, source_stale=stale)
                if stale:
                    item["eligible_for_action_training"] = False
                output.append(_json(item))
            return "".join(line + "\n" for line in output)
