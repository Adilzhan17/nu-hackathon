"""Local review UI. Run with .venv/bin/python ml/annotator.py.

Open http://127.0.0.1:8765 through the IDE's port forwarding when working remotely.
Uses local videos only. Requires PyAV to remux H.264 MOV into browser-friendly MP4.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse


def preview_id(row):
    return "video_" + hashlib.sha256(row["video"].encode()).hexdigest()[:16]


def suggested_window(row):
    duration = float(row["duration_sec"])
    anchor = float(row.get("anchor_sec") or duration / 2)
    if not (math.isfinite(duration) and duration > 0 and math.isfinite(anchor) and 0 <= anchor <= duration):
        raise ValueError("Invalid duration or event timestamp")
    return max(0, anchor - 4), min(duration, anchor + 4)


class ReviewStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def read(self):
        raw = self.path.read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        return list(reader), reader.fieldnames, hashlib.sha256(raw).hexdigest()

    def update(self, payload):
        with self.lock:
            rows, fields, revision = self.read()
            if payload.get("revision") != revision:
                raise FileExistsError("Таблица изменилась. Обновите страницу перед сохранением.")
            for field in ("annotation_mode", "label_scope"):
                if field not in fields:
                    fields.append(field)
            index = payload.get("index")
            if type(index) is not int or not 0 <= index < len(rows):
                raise ValueError("Неизвестная строка")
            row = rows[index]
            if payload.get("action") == "accept_source_normals":
                for candidate in rows:
                    if candidate.get("source_verdict") != "без нарушения" or candidate.get("review_status") != "pending":
                        continue
                    duration = float(candidate["duration_sec"])
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError("Invalid video duration")
                    candidate.update(start_sec="0", end_sec=str(duration), label="normal",
                                     review_status="source_labeled", annotation_mode="source_video_label",
                                     label_scope="binary", observation="")
            elif payload.get("action") == "duplicate":
                row = dict(row)
                row.update(start_sec="", end_sec="", label="", observation="", rule_id="", review_status="pending",
                           annotation_mode="", label_scope="")
                rows.append(row)
                index = len(rows) - 1
            else:
                status = payload.get("review_status")
                if status not in {"reviewed", "uncertain", "pending"}:
                    raise ValueError("Неизвестный статус")
                label = str(payload.get("label", "")).strip()
                observation = str(payload.get("observation", "")).strip()
                if len(label) > 120 or len(observation) > 4000:
                    raise ValueError("Слишком длинный текст")
                start, end = payload.get("start_sec", ""), payload.get("end_sec", "")
                quick = payload.get("mode") == "quick"
                if status == "reviewed":
                    if quick:
                        start, end = suggested_window(row)
                        label = label or ("normal" if row.get("source_verdict") == "без нарушения" else "violation_unspecified")
                    try:
                        start, end = float(start), float(end)
                        duration = float(row["duration_sec"])
                    except (TypeError, ValueError):
                        raise ValueError("Укажите начало и конец в секундах") from None
                    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
                        raise ValueError("Нужно: 0 ≤ начало < конец ≤ длительность видео")
                    if not label or (not quick and not observation):
                        raise ValueError("Укажите тип события и наблюдаемое действие")
                    start, end = str(round(start, 3)), str(round(end, 3))
                else:
                    start, end = "", ""
                row.update(start_sec=start, end_sec=end, label=label, observation=observation, review_status=status)
                row["annotation_mode"] = ("point_window" if quick else "exact") if status == "reviewed" else ""
                row["label_scope"] = ("binary" if label in {"normal", "violation_unspecified"} else "event_type") if status == "reviewed" else ""
                if payload.get("action") == "save_and_duplicate":
                    extra = dict(row)
                    extra.update(start_sec="", end_sec="", label="", observation="", rule_id="", review_status="pending",
                                 annotation_mode="", label_scope="")
                    rows.append(extra)
                    index = len(rows) - 1
            backup = self.path.parent / "backups"
            backup.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            shutil.copy2(self.path, backup / f"event_review-{stamp}.csv")
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=self.path.parent, delete=False) as out:
                    temp_path = Path(out.name)
                    writer = csv.DictWriter(out, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temp_path, self.path)
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink()
            return index


def prepare_previews(rows, source, destination):
    import av

    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    for row in rows:
        group = preview_id(row)
        if group in seen:
            continue
        seen.add(group)
        video = (source / row["video"]).resolve()
        if not video.is_relative_to(source) or not video.is_file():
            raise ValueError("Invalid video path")
        target = destination / f"{group}.mp4"
        if target.exists():
            continue
        partial = target.with_suffix(".part.mp4")
        try:
            with av.open(str(video)) as incoming, av.open(str(partial), "w", options={"movflags": "+faststart"}) as outgoing:
                track = incoming.streams.video[0]
                output_track = outgoing.add_stream_from_template(track)
                for packet in incoming.demux(track):
                    if packet.dts is None:
                        continue
                    packet.stream = output_track
                    outgoing.mux(packet)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
    print(f"Prepared {len(seen)} local MP4 previews (remux, no AI, no audio)", flush=True)


def make_handler(store, previews, source=None, training=None, video_jobs=None, jury=None, human=None, objects=None, review_frames=None):
    source = Path(source).resolve() if source is not None else None

    class Handler(BaseHTTPRequestHandler):
        def local_training_control(self):
            # The public page is a monitor. Remote training commands are reserved
            # for the loopback interface; cloudflared preserves the public Host.
            return self.client_address[0] == '127.0.0.1' and self.headers.get('Host') in {
                '127.0.0.1:' + str(self.server.server_port), 'localhost:' + str(self.server.server_port)}

        def json_response(self, value, status=200):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = urlparse(self.path).path
            selected_image=re.fullmatch(r'/api/human-objects/images/(human_[a-f0-9]{24})\.jpg',path)
            if selected_image and objects is not None:
                row=objects.rows.get(selected_image[1])
                if row is None:return self.send_error(404)
                return self.send_file(objects.image_path(row),'image/jpeg')
            if path=='/api/training/split' and human is not None:
                from suite_dashboard import read_json
                return self.json_response(read_json(human.root/'data/training/split95_v1/report.json',{'status':'not_prepared'}))
            if path in {'/label','/label/'} and human is not None:
                return self.send_file(Path(__file__).with_name('human_review.html'), 'text/html; charset=utf-8')
            if path in {'/label/objects','/label/objects/'} and objects is not None:
                return self.send_file(Path(__file__).with_name('human_objects.html'), 'text/html; charset=utf-8')
            if path in {'/api/human-review','/api/human-objects','/api/human-review/export','/api/human-objects/export'}:
                target=objects if path.startswith('/api/human-objects') else human
                if target is None:return self.send_error(404)
                try:
                    if path.endswith('/export'):
                        raw=target.export_jsonl().encode('utf-8')
                        self.send_response(200)
                        self.send_header('Content-Type','application/x-ndjson; charset=utf-8')
                        self.send_header('Content-Length',str(len(raw)))
                        self.send_header('Content-Disposition','attachment; filename="human-decisions.jsonl"')
                        self.send_header('Cache-Control','no-store')
                        self.end_headers();self.wfile.write(raw);return
                    return self.json_response(target.catalog())
                except (ValueError,OSError) as exc:
                    return self.json_response({'error':str(exc)},getattr(exc,'status_code',422))
            download=re.fullmatch(r'/api/human-datasets/([a-zA-Z0-9_-]{8,100})\.zip',path)
            if download and human is not None:
                exports=human.root/'data/human_review/exports'
                target=exports/download[1]/'dataset.zip'
                if target.is_file() and not target.is_symlink() and not target.parent.is_symlink() and target.resolve().is_relative_to(exports.resolve()):
                    return self.send_file(target,'application/zip',attachment=True,download_name=download[1]+'.zip')
                return self.send_error(404)
            if jury is not None and jury.handle_get(self):
                return
            from frontend_proxy import forward
            if forward(self, 'GET'):
                return
            if video_jobs is not None:
                if path == '/readiness':
                    return self.send_file(Path(__file__).with_name('readiness.html'), 'text/html; charset=utf-8')
                if path == '/api/readiness':
                    from suite_dashboard import read_json
                    report=read_json(video_jobs.root/'runs/readiness/report.json', {})
                    try:
                        report['live_human_review']={
                            'actions':human.catalog()['counts'] if human is not None else None,
                            'objects':objects.catalog()['counts'] if objects is not None else None,
                            'note':'Current human decisions; model metrics above describe earlier checkpoints.'}
                    except (ValueError,OSError) as exc:
                        report['live_human_review']={'error':str(exc)}
                    return self.json_response(report)
                if path == '/api/regulatory-sources':
                    from suite_dashboard import read_json
                    return self.json_response(read_json(Path(__file__).with_name('regulatory_sources.json'), {}))
                if path in {'/analyze','/analyze/'}:
                    return self.send_file(Path(__file__).with_name('analyze.html'),'text/html; charset=utf-8')
                if path == '/review_timeline.js':
                    return self.send_file(Path(__file__).with_name('review_timeline.js'),'text/javascript; charset=utf-8')
                if path == '/api/models':
                    return self.json_response({'models':video_jobs.models()})
                if path == '/api/training/suite':
                    from suite_dashboard import snapshot
                    return self.json_response(snapshot(video_jobs))
                if path in ('/scenarios','/scenarios/'):
                    return self.send_file(Path(__file__).with_name('scenario.html'),'text/html; charset=utf-8')
                if path == '/api/scenario':
                    from suite_dashboard import read_json
                    root=video_jobs.root/'runs/scenario_v2';report=read_json(root/'report.json',{})
                    scenario_sources={v['video']:v['sha256'] for v in read_json(root/'plan.json',{}).get('videos',[])}
                    for v in report.get('videos',[]):
                        v['preview']='/autolabel/detailed_train_report/previews/video_'+scenario_sources[v['video']][:16]+'.mp4'
                    return self.json_response(dict(status=read_json(root/'status.json',{}),report=report))
                if path == '/api/scenario/report':
                    target=video_jobs.root/'runs/scenario_v2/report.json'
                    if target.is_file():return self.send_file(target,'application/json',attachment=True,download_name='scenario-pilot.json')
                    return self.send_error(404)
                scenario_asset=re.fullmatch(r'/scenario-evidence/(video_[a-f0-9]{16}/(?:survey|window_\d{3}|recheck_\d{2})/f\d{3}\.jpg)',path)
                if scenario_asset:
                    base=(video_jobs.root/'runs/scenario_v2').resolve();target=(base/scenario_asset[1]).resolve()
                    if target.is_relative_to(base) and target.is_file():return self.send_file(target,'image/jpeg')
                    return self.send_error(404)
                if path == '/api/training/quality-report':
                    target=video_jobs.root/'runs/quality_v2/audit.json'
                    if target.is_file():return self.send_file(target,'application/json',attachment=True,download_name='quality-report.json')
                    return self.send_error(404)
                if path == '/training/archive':
                    return self.send_file(Path(__file__).with_name('training_archive.html'), 'text/html; charset=utf-8')
                auto_asset=re.fullmatch(r'/autolabel/(review\.html|proposals\.jsonl|conflicts\.jsonl|review_queue\.jsonl|context/review_\d{4}\.mp4)',path)
                if auto_asset:
                    target=video_jobs.root/'data/autolabel/pilot_final'/auto_asset[1]
                    if target.is_file():
                        mime='video/mp4' if target.suffix=='.mp4' else 'text/html; charset=utf-8' if target.suffix=='.html' else 'application/x-ndjson'
                        return self.send_file(target,mime,attachment=target.suffix=='.jsonl',download_name=target.name)
                    return self.send_error(404)
                detailed_asset=re.fullmatch(r'/autolabel/(detailed_train_report/(?:index\.html|previews/video_[a-f0-9]{16}\.mp4)|objects_v1/(?:index\.html|images/frame_\d{4}\.jpg)|detailed_train/frames/candidate_\d{5}_\d{3}\.jpg)',path)
                if detailed_asset:
                    base=(video_jobs.root/'data/autolabel').resolve();target=(base/detailed_asset[1]).resolve()
                    if target.is_relative_to(base) and target.is_file():
                        mime={'.html':'text/html; charset=utf-8','.jpg':'image/jpeg','.mp4':'video/mp4'}[target.suffix]
                        return self.send_file(target,mime)
                    return self.send_error(404)
                model_asset=re.fullmatch(r'/api/models/(x3d|slowfast|videomaev2|videomaev2_large|videomaev2_giant|yolov8|yolo11)/(model\.pt|validation\.json|settings\.json|cpu_validation\.json)',path)
                if model_asset:
                    target=video_jobs.root/'artifacts/models'/model_asset[1]/model_asset[2]
                    if target.is_file():return self.send_file(target,'application/octet-stream',attachment=True,download_name=model_asset[1]+'-'+model_asset[2])
                match = re.fullmatch(r'/api/uploads/([a-f0-9]{32})(?:/(media|report|(?:yolov8|yolo11)-objects\.json|(?:x3d|slowfast|videomaev2|videomaev2_large|videomaev2_giant|yolov8|yolo11)-\d+\.jpg))?',path)
                if match:
                    try:
                        folder=video_jobs.folder(match[1])
                        asset=match[2]
                        if not asset:return self.json_response(video_jobs.get(match[1]))
                        if asset=='report':return self.send_file(folder/'job.json','application/json',attachment=True,download_name='analysis.json')
                        if asset=='media':
                            target=folder/'preview.mp4' if (folder/'preview.mp4').exists() else folder/'source'
                            if target.is_file():return self.send_file(target,'video/mp4')
                        elif (folder/asset).is_file():return self.send_file(folder/asset,'application/json' if asset.endswith('.json') else 'image/jpeg')
                    except FileNotFoundError:pass
                    return self.send_error(404)
            if training is not None:
                if path in {'/training', '/training/'}:
                    return self.send_file(Path(__file__).with_name('training.html'), 'text/html; charset=utf-8')
                if path == '/api/training/status':
                    return self.json_response(dict(training.snapshot(), can_control=self.local_training_control()))
                match = re.fullmatch(r'/api/training/artifact/(\d{8}-\d{6}-[a-f0-9]{6})/(best\.pt|validation\.json|environment\.txt)', path)
                if match:
                    target = training.directory / match[1] / match[2]
                    if target.is_file():
                        return self.send_file(target, 'application/octet-stream', attachment=True, download_name=target.name)
                if path == '/api/training/observations':
                    target = training.root / 'data/transcripts/observations.jsonl'
                    if target.is_file():
                        return self.send_file(target, 'application/x-ndjson', attachment=True, download_name=target.name)
            if path == "/api/items":
                with store.lock:
                    rows, _, revision = store.read()
                return self.json_response({"items": [dict(row, preview_id=preview_id(row)) for row in rows], "revision": revision})
            if path in {'/', '/review', '/review/'}:
                return self.send_file(Path(__file__).with_name("annotator.html"), "text/html; charset=utf-8")
            if path == "/api/export":
                with store.lock:
                    return self.send_file(store.path, "text/csv; charset=utf-8", attachment=True)
            if re.fullmatch(r"/download/video_[0-9a-f]{16}\.mp4", path):
                key = path.rsplit("/", 1)[1][:-4]
                with store.lock:
                    rows, _, _ = store.read()
                row = next((row for row in rows if preview_id(row) == key), None)
                target = previews / f"{key}.mp4"
                if row is not None and target.is_file():
                    return self.send_file(target, "video/mp4", attachment=True,
                                          download_name=Path(row["video"]).with_suffix(".mp4").name)
            if source is not None and re.fullmatch(r"/download/video_[0-9a-f]{16}\.mov", path):
                key = path.rsplit("/", 1)[1][:-4]
                with store.lock:
                    rows, _, _ = store.read()
                row = next((row for row in rows if preview_id(row) == key), None)
                if row is not None:
                    target = (source / row["video"]).resolve()
                    if target.is_relative_to(source) and target.is_file():
                        return self.send_file(target, "video/quicktime", attachment=True)
            if re.fullmatch(r"/video/video_[0-9a-f]{16}\.mp4", path):
                target = previews / path.rsplit("/", 1)[1]
                if target.is_file():
                    return self.send_file(target, "video/mp4")
            self.send_error(404)

        def send_file(self, path, content_type, attachment=False, download_name=None):
            with path.open("rb") as stream:
                size = os.fstat(stream.fileno()).st_size
                start, end = 0, size - 1
                requested = self.headers.get("Range")
                if requested:
                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                    if not match or not any(match.groups()):
                        return self.send_error(416)
                    if match[1]:
                        start = int(match[1])
                        end = min(int(match[2]), end) if match[2] else end
                    else:
                        start = max(0, size - int(match[2]))
                    if start > end or start >= size:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                self.send_response(206 if requested else 200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if attachment:
                    fallback = "event_review.csv" if content_type.startswith("text/csv") else "video.mp4" if content_type == "video/mp4" else "video.mov"
                    self.send_header("Content-Disposition", f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(download_name or path.name, safe='')}")
                if requested:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                stream.seek(start)
                remaining = end - start + 1
                try:
                    while remaining:
                        chunk = stream.read(min(256 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_POST(self):
            if self.path == '/api/human-objects/frames':
                return self.selected_frame_request()
            if self.path in {'/api/human-review','/api/human-review/tasks','/api/human-review/dataset','/api/human-objects','/api/human-objects/dataset'}:
                return self.human_request()
            if jury is not None and jury.handle_post(self):
                return
            from frontend_proxy import forward
            if forward(self, 'POST'):
                return
            if video_jobs is not None and self.path.startswith('/api/uploads'):
                return self.upload_request('POST')
            if self.path != "/api/save" and not (training is not None and self.path == '/api/training/action'):
                return self.send_error(404)
            origin = urlparse(self.headers.get("Origin", ""))
            if origin.scheme not in {"http", "https"} or origin.netloc != self.headers.get("Host"):
                return self.json_response({"error": "Origin rejected"}, 403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected object")
                if self.path == '/api/training/action':
                    if not self.local_training_control():
                        return self.json_response({'error': 'Публичная панель доступна для просмотра; запуск и остановка выполняются с сервера.'}, 403)
                    if payload.get('action') == 'check' and 'user' in payload:
                        training.set_user(payload['user'])
                    return self.json_response(training.action(payload.get('action')))
                index = store.update(payload)
                self.json_response({"saved": True, "index": index})
            except FileExistsError as exc:
                self.json_response({"error": str(exc)}, 409)
            except (ValueError, TypeError) as exc:
                self.json_response({"error": str(exc)}, 422)

        def do_PUT(self):
            if video_jobs is not None:return self.upload_request('PUT')
            self.send_error(404)

        def do_HEAD(self):
            from frontend_proxy import forward
            if not forward(self, 'HEAD'):
                self.send_error(405)

        def selected_frame_request(self):
            if review_frames is None or objects is None:return self.send_error(404)
            try:origin=urlparse(self.headers.get('Origin',''))
            except ValueError:return self.json_response({'error':'Origin rejected'},403)
            if origin.scheme not in {'http','https'} or origin.netloc!=self.headers.get('Host'):
                return self.json_response({'error':'Origin rejected'},403)
            try:
                self.connection.settimeout(30)
                if self.headers.get('Transfer-Encoding'):raise ValueError('Content-Length required')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16384:raise ValueError('Некорректный размер запроса')
                raw=self.rfile.read(size)
                if len(raw)!=size:raise ValueError('Запрос получен не полностью')
                payload=json.loads(raw)
                if not isinstance(payload,dict):raise ValueError('Expected object')
                metadata=review_frames.prepare_frame(payload)
                return self.json_response(objects.register_frame(metadata),201)
            except (KeyError,FileNotFoundError):return self.json_response({'error':'Кадр или запрос не найден'},404)
            except (ValueError,TypeError,OSError) as exc:
                return self.json_response({'error':str(exc)},getattr(exc,'status_code',422))

        def human_request(self):
            target=objects if self.path.startswith('/api/human-objects') else human
            if target is None:return self.send_error(404)
            try:origin=urlparse(self.headers.get('Origin',''))
            except ValueError:return self.json_response({'error':'Origin rejected'},403)
            if origin.scheme not in {'http','https'} or origin.netloc!=self.headers.get('Host'):
                return self.json_response({'error':'Origin rejected'},403)
            try:
                self.connection.settimeout(30)
                if self.headers.get('Transfer-Encoding'):raise ValueError('Content-Length required')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=128*1024:raise ValueError('Некорректный размер решения')
                raw=self.rfile.read(size)
                if len(raw)!=size:raise ValueError('Запрос получен не полностью; решение не сохранено')
                payload=json.loads(raw)
                if not isinstance(payload,dict):raise ValueError('Expected object')
                if self.path.endswith('/tasks'):
                    return self.json_response(human.add_task(payload),201)
                if self.path.endswith('/dataset'):
                    if self.path.startswith('/api/human-objects'):
                        result=objects.export_dataset()
                    else:
                        from export_human_dataset import export_snapshot
                        result=export_snapshot(human.root,store=human)
                    result['download_url']='/api/human-datasets/'+result['export_id']+'.zip'
                    return self.json_response(result,201)
                return self.json_response(target.save(payload))
            except (ValueError,TypeError,OSError,KeyError) as exc:
                return self.json_response({'error':str(exc)},getattr(exc,'status_code',422))

        def do_DELETE(self):
            if video_jobs is not None:return self.upload_request('DELETE')
            self.send_error(404)

        def upload_request(self,method):
            origin=urlparse(self.headers.get('Origin',''))
            if origin.scheme not in {'http','https'} or origin.netloc != self.headers.get('Host'):
                return self.json_response({'error':'Origin rejected'},403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                match=re.fullmatch(r'/api/uploads/([a-f0-9]{32})(/complete)?',self.path)
                if method=='DELETE' and match and not match[2]:
                    return self.json_response(video_jobs.delete(match[1]))
                if method=='PUT' and match and not match[2]:
                    self.connection.settimeout(60)
                    return self.json_response(video_jobs.chunk(match[1],int(self.headers.get('X-Upload-Offset','-1')),size,self.rfile))
                if method=='POST' and (self.path=='/api/uploads' or match and match[2]):
                    if not 0<size<=65536:raise ValueError('Invalid request size')
                    payload=json.loads(self.rfile.read(size))
                    if not isinstance(payload,dict):raise ValueError('Expected object')
                    result=video_jobs.create(payload) if self.path=='/api/uploads' else video_jobs.submit(match[1])
                    return self.json_response(result,201 if self.path=='/api/uploads' else 202)
                return self.send_error(404)
            except FileNotFoundError as exc:return self.json_response({'error':str(exc)},404)
            except FileExistsError as exc:return self.json_response({'error':str(exc)},409)
            except (ValueError,TypeError,TimeoutError) as exc:return self.json_response({'error':str(exc)},422)

    return Handler


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--csv", type=Path, default=root / "data/review/event_review.csv")
    parser.add_argument("--source", type=Path, default=root / "data/raw/open")
    args = parser.parse_args()
    store = ReviewStore(args.csv)
    previews = args.csv.resolve().parent / "previews"
    prepare_previews(store.read()[0], args.source.resolve(), previews)
    from training_control import TrainingControl
    from video_jobs import VideoJobs
    training = TrainingControl(root)
    video_jobs = VideoJobs(root)
    from jury_api import JuryAPI
    jury = JuryAPI(root)
    from human_review import HumanReview
    from object_review import ObjectReview
    human,objects=HumanReview(root),ObjectReview(root)
    from review_frames import ReviewFrames
    frames=ReviewFrames(root,human,objects)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, previews, args.source, training, video_jobs, jury, human, objects, frames))
    print(f"Annotation tool: http://127.0.0.1:{args.port} (use IDE port forwarding)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
