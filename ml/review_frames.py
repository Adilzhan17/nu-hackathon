"""Extract selected local frames for manual annotation. No model or network calls."""
import hashlib
import io
import math
import os
from pathlib import Path
import threading
import time
import uuid
from PIL import Image


class AssistError(ValueError):
    def __init__(self, message, code="invalid_request", status_code=400):
        super().__init__(message)
        self.code, self.status_code = code, status_code


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def _number(value):
    if type(value) not in {int, float} or not math.isfinite(value):
        raise AssistError("Требуется конечное число.")
    return float(value)


def extract_frame(path, timestamp):
    """Return nearest native frame, with timestamps relative to the first decoded PTS."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "SLICE"
        stream.thread_count = 1
        if stream.width * stream.height > 36_000_000:
            raise AssistError("Слишком большое разрешение исходного видео.")
        first = next(container.decode(stream), None)
        if first is None or first.pts is None:
            raise AssistError("Не удалось прочитать время исходного кадра.")
        origin = float(first.pts * first.time_base)
        best, best_time = first, 0.0
        if timestamp > 0:
            container.seek(int((origin + timestamp) / stream.time_base), stream=stream, backward=True)
            best = None
            deadline = time.monotonic() + 20
            for frame in container.decode(stream):
                if time.monotonic() > deadline:
                    raise AssistError("Не удалось быстро найти кадр. Выберите ближайший момент.")
                if frame.pts is None:
                    continue
                relative = float(frame.pts * frame.time_base) - origin
                if relative < -1e-6:
                    continue
                if best is None or abs(relative - timestamp) < abs(best_time - timestamp):
                    best, best_time = frame, relative
                if relative >= timestamp:
                    break
        if best is None or abs(best_time - timestamp) > 1.0:
            raise AssistError("Запрошенный момент не найден в исходном видео.")
        return best.to_image().convert("RGB"), max(0.0, best_time)


class ReviewFrames:
    def __init__(self, root, human, objects):
        self.root = Path(root).resolve()
        self.human, self.objects = human, objects
        self.directory = self.root / "data/human_review/selected_frames"
        (self.directory / "frames").mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.hash_cache = {}

    def _verify_hash(self, path, expected):
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with self.lock:
            actual = self.hash_cache.get(identity)
        if actual is None:
            actual = _digest(path)
            with self.lock:
                self.hash_cache[identity] = actual
        if actual != expected:
            raise AssistError("Исходный файл изменился. Обновите очередь.", "stale_source", 409)

    def prepare_frame(self, payload):
        if not isinstance(payload, dict) or bool(payload.get("task_id")) == bool(payload.get("image_id")):
            raise AssistError("Выберите один эпизод или кадр из очереди.")
        if any(k in payload for k in ("path", "url", "video", "image", "base64", "image_url")):
            raise AssistError("Кадр можно выбрать только из серверной очереди.")
        original_bytes = None
        if payload.get("task_id"):
            catalog = self.human.catalog()
            task = next((r for r in catalog["tasks"] if r["id"] == payload["task_id"]), None)
            if not task:
                raise AssistError("Эпизод не найден.", "unknown_task", 404)
            if payload.get("source_fingerprint") != task["source_fingerprint"]:
                raise AssistError("Очередь изменилась. Обновите страницу.", "stale_source", 409)
            timestamp = _number(payload.get("timestamp_sec"))
            if not 0 <= timestamp <= task["duration_sec"]:
                raise AssistError("Момент выходит за длительность видео.")
            base = (self.root / "data/raw/open").resolve()
            path = (base / task["video"]).resolve()
            if not path.is_relative_to(base) or not path.is_file():
                raise AssistError("Исходное видео недоступно.", "missing_source", 404)
            self._verify_hash(path, task["video_sha256"])
            image, actual = extract_frame(path, timestamp)
            metadata = dict(video=task["video"], group_id=task["group_id"], source_sha256=task["video_sha256"],
                            split="train", timestamp_sec=actual, requested_timestamp_sec=timestamp,
                            source={"task_id": task["id"], "source_fingerprint": task["source_fingerprint"]})
        else:
            self.objects.check_sources()
            row = self.objects.rows.get(payload["image_id"])
            if not row:
                raise AssistError("Кадр не найден.", "unknown_frame", 404)
            if payload.get("image_sha256") != row["image_sha256"]:
                raise AssistError("Кадр изменился. Обновите страницу.", "stale_source", 409)
            path = self.objects.image_path(row) if hasattr(self.objects, "image_path") else self.objects.source / row["image"]
            self._verify_hash(path, row["image_sha256"])
            with Image.open(path) as source:
                if source.width * source.height > 36_000_000:
                    raise AssistError("Слишком большое разрешение кадра.")
                image = source.convert("RGB")
            original_bytes = path.read_bytes()
            metadata = dict(video=row["video"], group_id=row["group_id"], source_sha256=row["source_sha256"], split="train",
                timestamp_sec=row["timestamp"], requested_timestamp_sec=row["timestamp"],
                source={"image_id": row["image_id"], "image_sha256": row["image_sha256"]})
        # Existing reviewed JPEGs are copied byte-for-byte to preserve their lineage.
        if original_bytes is None:
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=96, subsampling=0)
            data = out.getvalue()
        else:
            data = original_bytes
        sha = hashlib.sha256(data).hexdigest()
        path = self.directory / "frames" / (sha + ".jpg")
        if not path.exists():
            temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(data)
            os.chmod(temporary, 0o600)
            temporary.replace(path)
        return dict(metadata, path=path, width=image.width, height=image.height, image_sha256=sha)
