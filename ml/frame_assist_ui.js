/* Human-triggered assistance. Model output is never a saved annotation. */
(() => {
  'use strict';
  const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
  const style = el('style');
  style.textContent = `.fa-panel{margin:14px 0;padding:16px;border:1px solid #ded7e1;border-radius:12px;background:#f7f4f8;color:#302936}.fa-panel h2{font-size:18px;margin:0 0 5px}.fa-panel p{margin:7px 0}.fa-panel small{display:block;color:#65596e;font-size:12px;line-height:1.5}.fa-toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:10px}.fa-toolbar button{min-height:40px;padding:9px 13px;cursor:pointer;background:#f7f4f8;border:1px solid #ded7e1;border-radius:8px;color:#302936;font:600 14px system-ui}.fa-toolbar button:disabled{opacity:.5;cursor:wait}.fa-status{font-size:13px;color:#302936;overflow-wrap:anywhere}.fa-status[data-error=true]{color:#aa2143}.fa-result{margin-top:14px;overflow-wrap:anywhere}.fa-result h3{font:600 14px system-ui;margin:14px 0 6px;color:#216345}.fa-result ul{margin:5px 0;padding-left:20px}.fa-result li{margin:5px 0;font-size:14px}.fa-result .fa-context h3{color:#805313}.fa-frame-label{font-size:13px;color:#805313}.fa-link{display:inline-block;color:#216345;margin-top:10px}.fa-disclaimer{border-left:3px solid #ded7e1;padding-left:9px}.fa-busy{width:10px;height:10px;border-radius:50%;background:#f8f3f6;display:inline-block;margin-right:7px}.fa-result:empty{display:none}`;
  document.head.append(style);
  async function json(url, options = {}) {
    const response = await fetch(url, {cache: 'no-store', ...options});
    let value; try { value = await response.json(); } catch (_) { throw Error('Сервер вернул неполный ответ. Повторите запрос.'); }
    if (!response.ok) { const err = Error(typeof value.error === 'string' ? value.error : value.message || value.error?.message || 'Ошибка ' + response.status); err.status = response.status; throw err; }
    return value;
  }
  let statusPromise;
  function modelStatus() { return statusPromise ||= json('/api/frame-assist/status'); }
  function lines(value) {
    if (value == null) return [];
    if (!Array.isArray(value)) value = [value];
    return value.map(item => typeof item === 'string' ? item : item?.description || item?.text || item?.claim || item?.fact || item?.question || item?.reason || '').filter(Boolean);
  }
  class Panel {
    constructor(host, {onResult, onStart, title = 'Помощник по кадру'} = {}) {
      this.host = host; this.onResult = onResult; this.onStart = onStart; this.serial = 0; this.job = null; this.result = null; this.source = null;
      host.classList.add('fa-panel'); host.append(el('h2', title));
      host.append(el('small', 'По нажатию выбранный кадр, его увеличенные участки и контекст Qwen отправляются в OpenAI.'));
      host.append(el('small', 'Ответ — подсказка: предметы и действия подтверждаете вы.', 'fa-disclaimer'));
      host.append(el('small', 'Здесь рамки предлагает VLM, это не YOLO-checkpoint. Классы подсказок: cash, document_id, key, stamp, other. В редакторе сохранённые рамки имеют источник «человек». Наш локальный дообученный детектор доступен в анализе видео: /analyze.', 'fa-disclaimer'));
      const modelLine = el('small', 'Проверяем доступность помощника…'); host.append(modelLine);
      modelStatus().then(value => { modelLine.textContent = (value.model ? 'Модель: ' + value.model + '. ' : '') + (value.configured ? 'Запуск по кнопке.' : 'Ключ на сервере пока не настроен.'); }).catch(() => { modelLine.textContent = 'Доступность помощника будет проверена при запуске.'; });
      this.toolbar = el('div', undefined, 'fa-toolbar'); host.append(this.toolbar);
      this.status = el('p', 'Модель запускается только по вашей кнопке.', 'fa-status'); this.status.setAttribute('role', 'status'); this.status.setAttribute('aria-live', 'polite'); host.append(this.status);
      this.content = el('div', undefined, 'fa-result'); host.append(this.content);
    }
    message(text, error = false) { this.status.textContent = text; this.status.dataset.error = String(error); }
    clear() { this.serial++; this.job = null; this.result = null; this.source = null; this.content.replaceChildren(); this.message('Модель запускается только по вашей кнопке.'); this.onStart?.(false); }
    render(job) {
      this.content.replaceChildren(); const result = job.result || job.output || {};
      this.job = job; this.result = result;
      if (this.source?.label) this.content.append(el('p', this.source.label, 'fa-frame-label'));
      if (result.description || result.summary) this.content.append(el('p', result.description || result.summary));
      const sections = [ ['Что видно на кадре', result.visible_facts || result.observations, ''], ['Контекст и предположения — требуют проверки', result.context_hypotheses || result.hypotheses, 'fa-context'], ['Расхождения с контекстом Qwen', result.context_disagreements, 'fa-context'], ['Что нельзя уверенно определить', result.uncertainties || result.unknowns, 'fa-context'], ['Что уточнить человеку', result.review_questions || result.questions, ''] ];
      for (const [title, values, cls] of sections) { const items = lines(values); if (!items.length) continue; const section = el('section', undefined, cls); section.append(el('h3', title)); const list = el('ul'); items.forEach(s => list.append(el('li', s))); section.append(list); this.content.append(section); }
      const candidates = result.candidates || result.objects || [];
      this.content.append(el('small', `Предложений рамок: ${candidates.length}. Они не добавлены в разметку и могут быть неточными.`));
      const model = job.model || result.model;
      this.message('Готово' + (model ? ' · ' + model : '') + '. Проверьте подсказки на изображении.');
      this.onResult?.(result, job, this.source);
    }
    async run(payload, source) {
      const serial = ++this.serial; this.source = source; this.result = null; this.job = null; this.content.replaceChildren(); this.onStart?.(true); this.message('Отправляем выбранный кадр…');
      try {
        const job = await json('/api/frame-assist', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
        if (serial !== this.serial) return null;
        return await this.poll(job, serial);
      } catch (error) { if (serial === this.serial) this.message(error.message, true); return null; }
      finally { if (serial === this.serial) this.onStart?.(false); }
    }
    async resume(id, source) {
      if (!/^[A-Za-z0-9_-]{8,100}$/.test(id)) { this.message('Некорректный номер подсказки.', true); return; }
      const serial = ++this.serial; this.source = source; this.onStart?.(true);
      try { const job = await json('/api/frame-assist/jobs/' + encodeURIComponent(id)); if (serial !== this.serial) return; await this.poll(job, serial); }
      catch (e) { if (serial === this.serial) this.message(e.message, true); }
      finally { if (serial === this.serial) this.onStart?.(false); }
    }
    async poll(initial, serial) {
      let job = initial; const deadline = Date.now() + 310000;
      while (serial === this.serial) {
        const status = job.status || job.state;
        if (['done', 'completed', 'complete', 'succeeded'].includes(status)) { if (!job.result && !job.output) { const id = job.id || job.job_id; if (!id || !/^[A-Za-z0-9_-]{8,100}$/.test(id)) throw Error('Не удалось получить результат подсказки.'); job = await json('/api/frame-assist/jobs/' + encodeURIComponent(id)); if (serial !== this.serial) return null; if (!job.result && !job.output) throw Error('Подсказка ещё не содержит результата. Обновите её позже.'); } this.render(job); return job; }
        if (['failed', 'error', 'cancelled'].includes(status)) throw Error(typeof job.error === 'string' ? job.error : job.error?.message || job.message || 'Не удалось получить подсказку. Разметка не изменилась.');
        const id = job.id || job.job_id;
        if (!id || !/^[A-Za-z0-9_-]{8,100}$/.test(id)) throw Error('Не удалось получить номер задания.');
        if (Date.now() > deadline) throw Error('Ожидание заняло больше пяти минут. Обновите страницу для проверки состояния.');
        this.job = job; this.message((status === 'queued' ? 'В очереди' : 'Модель рассматривает кадр и контекст') + '… Можно продолжить просмотр.');
        await new Promise(resolve => setTimeout(resolve, 1300)); if (serial !== this.serial) return null;
        job = await json('/api/frame-assist/jobs/' + encodeURIComponent(id));
      }
      return null;
    }
  }
  window.ForteFrameAssist = {Panel, json, el};
})();
