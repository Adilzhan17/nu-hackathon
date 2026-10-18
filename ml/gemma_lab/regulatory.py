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
    "door_after_entry": {
        "title": "Требования к организации охраны и устройству помещений банков; постановление Правления НБРК от 24.02.2020 № 14",
        "section": "пункт 16, подпункт 1",
        "url": "https://adilet.zan.kz/rus/docs/V2000020075",
        "review_title": "Устройство входной двери кассы",
        "qualification": "reference_only",
        "legal_note": (
            "Пункт требует отдельного входа с закрывающейся на замок дверью. "
            "Он сам по себе не устанавливает срок закрывания после прохода. "
            "Для квалификации оставленной открытой двери нужен внутренний режим доступа."
        ),
    },
    "equipment_on_departure": {
        "title": "Требования к обеспечению информационной безопасности банков; постановление Правления АРРФР от 03.04.2026 № 53",
        "section": "пункт 53; пункт 71, подпункт 9",
        "url": "https://adilet.zan.kz/rus/docs/V2600038336",
        "review_title": "Защита и автоматическая блокировка рабочего места",
        "qualification": "reference_only",
        "legal_note": (
            "Пункт 71(9) предусматривает блокировку или завершение сеанса после более "
            "30 минут неактивности. Короткое видео и светящийся экран этого не доказывают. "
            "Обязанность выключать оборудование при каждом уходе из этих пунктов не следует."
        ),
    },
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
        "section": "пункт 80",
        "url": SOURCE_URL + "#z226",
        "review_title": "Хранение ключей и печатей при уходе",
        "legal_note": (
            "Пункт 80 требует порядка, исключающего несанкционированное использование, "
            "но конкретный запрет оставлять ключи на сейфе определяется внутренними "
            "документами банка. Пункт 76 относится к кладовой и сейфовой комнате, "
            "а не автоматически к рабочему сейфу. Результат является признаком для проверки."
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
    return dict(value, verified_at="2026-10-18", jurisdiction="Республика Казахстан",
                qualification=value.get('qualification', 'requires_context_review')) if value else None


def source_for_assessment(rule, observations):
    """Select the law from the evidenced object, never from a binary label."""
    if rule.get('rule_id') != 'valuables_on_departure':
        return regulatory_source(rule.get('rule_id'))
    ids = {e.get('observation_id') for e in rule.get('evidence', [])}
    properties = {o.get('property') for o in observations
                  if o.get('observation_id') in ids and o.get('kind') == 'state'}
    if properties & {'key_location', 'stamp_location'}:
        return regulatory_source('valuables_on_departure')
    if 'cash_location' in properties:
        return regulatory_source('previous_cash_before_next_customer')
    # The cash/keys provisions do not establish a general paper-document rule.
    return None


def attach_regulatory_sources(job):
    """Enrich historical responses without changing their saved evidence."""
    result = job.get("result") if isinstance(job, dict) else None
    if not isinstance(result, dict):
        return job
    for rule in result.get("rules", []):
        if isinstance(rule, dict):
            rule["regulatory_source"] = source_for_assessment(rule, result.get('observations', []))
            rule["legal_status"] = 'context_review_required' if rule['regulatory_source'] else 'internal_rule_required'
    return job
