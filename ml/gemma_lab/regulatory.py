"""Verified public-rule metadata attached to Gemma scenario findings.

The mapping is deliberately conservative: a visual scenario is a reason for
human review, not a final legal conclusion.  The source snapshot and digest are
kept in ``ml/regulatory_sources.json``.
"""

SOURCE_TITLE = (
    "Правила осуществления кассовых операций и операций по инкассации банкнот, "
    "монет и ценностей; постановление Правления НБРК от 29.11.2019 № 231"
)
SOURCE_URL = "https://adilet.zan.kz/rus/docs/V1900019680"


RULE_SOURCES = {
    "identity_before_cash": {
        "title": SOURCE_TITLE,
        "section": "пункт 19, подпункт 1",
        "url": SOURCE_URL,
        "review_title": "Порядок идентификации клиента перед приёмом наличных",
        "legal_note": (
            "Видео может показать последовательность действий, но не подтверждает "
            "содержание документа или проверку по цифровому сервису."
        ),
    },
    "previous_cash_before_next_customer": {
        "title": SOURCE_TITLE,
        "section": "пункт 20",
        "url": SOURCE_URL,
        "review_title": "Хранение ранее принятых наличных",
        "legal_note": (
            "Нужно проследить те же деньги и отличить завершённую операцию от текущей; "
            "наличные на столе сами по себе не доказывают нарушение."
        ),
    },
    "phone_before_counting": {
        "title": SOURCE_TITLE,
        "section": "пункт 12",
        "url": SOURCE_URL,
        "review_title": "Личные средства связи в кассовом узле",
        "legal_note": (
            "Пункт содержит исключения. Видимый телефон требует проверки контекста, "
            "но не образует нарушение автоматически."
        ),
    },
    "safe_on_departure": {
        "title": SOURCE_TITLE,
        "section": "пункты 20 и 22",
        "url": SOURCE_URL,
        "review_title": "Сохранность ценностей и уход с рабочего места",
        "legal_note": (
            "Открытый сейф и уход должны быть непрерывно видны; окончательная "
            "квалификация зависит от этапа обслуживания и внутреннего режима доступа."
        ),
    },
    "valuables_on_departure": {
        "title": SOURCE_TITLE,
        "section": "пункт 80; для наличных также пункт 20",
        "url": SOURCE_URL,
        "review_title": "Хранение ключей, печатей и ценностей при уходе",
        "legal_note": (
            "Пункт 80 требует порядка, исключающего несанкционированное использование, "
            "но конкретный запрет оставлять ключи на сейфе определяется внутренними "
            "документами банка. Результат является признаком для проверки."
        ),
    },
    "authenticity_after_counting": {
        "title": SOURCE_TITLE,
        "section": "пункт 19, подпункт 2; пункты 57–58",
        "url": SOURCE_URL,
        "review_title": "Пересчёт и проверка подлинности наличных",
        "legal_note": (
            "По видео нельзя определить подлинность банкнот или функции оборудования; "
            "оценивается только наблюдаемая последовательность."
        ),
    },
    "cash_workflow": {
        "title": SOURCE_TITLE,
        "section": "пункт 19, подпункт 2; пункты 57–58",
        "url": SOURCE_URL,
        "review_title": "Последовательность пересчёта и проверки наличных",
        "legal_note": (
            "Системные проводки и результаты проверки банкнот не видны на видео и "
            "не выводятся моделью."
        ),
    },
    "items_before_work": {
        "title": SOURCE_TITLE,
        "section": "пункт 12",
        "url": SOURCE_URL,
        "review_title": "Личные вещи в кассовом узле",
        "legal_note": (
            "Нужно подтвердить принадлежность предмета, устройство помещения и "
            "применимость предусмотренных исключений."
        ),
    },
    "cash_before_drink": {
        "title": SOURCE_TITLE,
        "section": "пункт 20",
        "url": SOURCE_URL,
        "review_title": "Сохранность наличных во время перерыва",
        "legal_note": (
            "Нужна непрерывная связь с теми же деньгами; напиток в кадре сам по себе "
            "не является нарушением."
        ),
    },
}


def regulatory_source(rule_id):
    """Return a fresh serializable mapping or ``None`` for operational-only rules."""
    value = RULE_SOURCES.get(rule_id)
    return dict(value) if value else None


def attach_regulatory_sources(job):
    """Enrich historical responses without changing their saved evidence."""
    result = job.get("result") if isinstance(job, dict) else None
    if not isinstance(result, dict):
        return job
    for rule in result.get("rules", []):
        if isinstance(rule, dict) and "regulatory_source" not in rule:
            rule["regulatory_source"] = regulatory_source(rule.get("rule_id"))
    return job
