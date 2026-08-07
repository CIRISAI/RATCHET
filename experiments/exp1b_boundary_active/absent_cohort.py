"""Refuse a cohort that is ABSENT but reads as behaviour.

`load_chains_from_tee_dir` already refuses an EMPTY cohort, because a silent
empty return would compute every downstream statistic over nothing and report a
clean run. This is the case that guard does not catch: a cohort that is the
right SHAPE and the wrong CONTENT.

The EN battery run of 2026-08-05 (CIRISAgent run 31183628588) is the worked
example. Three individually-reasonable links composed into it:

1. the runtime starts without an LLM, silently — no API key means
   ``logger.info(...)`` and ``return``, not a warning and not a refusal, so the
   agent boots, accepts messages, and has nothing to think with;
2. every interact rides to a fixed 180 s ceiling, because a `setdefault` wins
   over the battery's own 1800 whenever it is set first;
3. the harness scores ``success=bool(response_text)``, and the timeout literal
   "Still processing. Check back later…" is non-empty text.

Result: nine green checkmarks, nine timeout strings recorded as the agent's
answers, and `task_id` never assigned because no task was ever created. Scored,
that judges the agent against a canned string — the same shape as the AM q06
failure, where a timeout string failed on script ratio for containing no
Ethiopic and was read as the agent writing in the wrong script.

For a pre-registered campaign this is worse than missing data. Missing data is
visible. This is absent data wearing the shape of behaviour, and it reads as a
result.

The three signatures, any one of which is disqualifying:

* **fixed duration** — nine questions at 180.3 s is a ceiling, not nine agents
  each deliberating for exactly three minutes;
* **no task id** — no task created means the failure is upstream of the LLM
  call entirely;
* **still-processing literal** — matched PER LOCALE. An English-only check
  reproduces the AM q06 defect on any non-English cell.

Wire into the loader before scoring; do not use as an after-the-fact review.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence

#: Localized still-processing strings, EXTRACTED from the agent's own bundles at
#: v2.9.11-stable — `agent.still_processing` and `agent.processor_paused` in
#: `ciris_engine/data/localized/<locale>.json`. All 29 carry both. Regenerate
#: when the agent version moves.
#:
#: English-only matching is the AM q06 defect: the literal is emitted in the
#: cell's own language, so a non-English cell sails past an English check and is
#: then judged for writing in the wrong script. An unlisted locale refuses as
#: UNKNOWN, never as clean.
STILL_PROCESSING: Dict[str, Sequence[str]] = {
    'am': (
        'አሁንም በሂደት ላይ ነው። እባክዎን ቆይተው ይመልከቱ። የወኪሉ ምላሽ ዋስትና የለውም።',
        'አቀናባሪው ቆሟል — ተግባሩ ወደ ወረፋ ተጨምሯል። ለመቀጠል ሂደቱን እንደገና ያስጀምሩ።',
    ),
    'ar': (
        'المعالجة ما زالت جارية. يرجى العودة للتحقق لاحقاً. استجابة الوكيل غير مضمونة.',
        'المعالج متوقف مؤقتاً — تمت إضافة المهمة إلى قائمة الانتظار. استأنف المعالجة للمتابعة.',
    ),
    'bn': (
        'এখনও প্রক্রিয়াকরণ চলছে। অনুগ্রহ করে পরে আবার দেখুন। এজেন্টের প্রতিক্রিয়ার নিশ্চয়তা নেই।',
        'প্রসেসর বিরতিতে আছে — কাজটি সারিতে যোগ করা হয়েছে। চালিয়ে যেতে প্রক্রিয়াকরণ পুনরায় শুরু করুন।',
    ),
    'de': (
        'Die Verarbeitung läuft noch. Bitte schauen Sie später erneut nach. Eine Antwort des Agenten ist nicht garantiert.',
        'Prozessor pausiert – die Aufgabe wurde der Warteschlange hinzugefügt. Nehmen Sie die Verarbeitung wieder auf, um fortzufahren.',
    ),
    'en': (
        'Still processing. Check back later. Agent response is not guaranteed.',
        'Processor paused - task added to queue. Resume processing to continue.',
    ),
    'es': (
        'Todavía en proceso. Vuelve a consultar más tarde. La respuesta del agente no está garantizada.',
        'Procesador en pausa — la tarea se añadió a la cola. Reanuda el procesamiento para continuar.',
    ),
    'fa': (
        'پردازش هنوز ادامه دارد. لطفاً بعداً دوباره سر بزنید. پاسخ عامل تضمین نمی\u200cشود.',
        'پردازنده متوقف شده است — وظیفه به صف اضافه شد. برای ادامه، پردازش را از سر بگیرید.',
    ),
    'fr': (
        "Traitement toujours en cours. Veuillez revenir plus tard. La réponse de l'agent n'est pas garantie.",
        'Processeur en pause — la tâche a été ajoutée à la file. Reprenez le traitement pour continuer.',
    ),
    'ha': (
        'Ana ci gaba da sarrafawa. Don Allah a sake dubawa daga baya. Ba a tabbatar da amsar wakili ba.',
        "An dakatar da na'urar sarrafawa — an ƙara aikin a layin jira. Don Allah a sake kunna sarrafawa don a ci gaba.",
    ),
    'hi': (
        'अभी भी प्रोसेस हो रहा है। कृपया बाद में फिर देखें। एजेंट की प्रतिक्रिया की गारंटी नहीं है।',
        'प्रोसेसर रुका हुआ है — कार्य कतार में जोड़ दिया गया है। जारी रखने के लिए प्रोसेसिंग फिर से शुरू करें।',
    ),
    'id': (
        'Masih diproses. Silakan periksa kembali nanti. Respons agen tidak dijamin.',
        'Prosesor dijeda — tugas telah ditambahkan ke antrean. Lanjutkan pemrosesan untuk meneruskan.',
    ),
    'it': (
        "Elaborazione ancora in corso. Torna a controllare più tardi. La risposta dell'agente non è garantita.",
        "Processore in pausa — l'attività è stata aggiunta alla coda. Riprendi l'elaborazione per continuare.",
    ),
    'ja': (
        'まだ処理中です。しばらくしてからもう一度ご確認ください。エージェントの応答は保証されません。',
        'プロセッサーは一時停止中です。タスクはキューに追加されました。続行するには処理を再開してください。',
    ),
    'ko': (
        '아직 처리 중입니다. 잠시 후 다시 확인해 주세요. 에이전트의 응답은 보장되지 않습니다.',
        '프로세서가 일시정지되었습니다 — 작업이 큐에 추가되었습니다. 계속하려면 처리를 재개하세요.',
    ),
    'mr': (
        'अजूनही प्रक्रिया सुरू आहे. कृपया नंतर पुन्हा पाहा. एजंटच्या प्रतिसादाची हमी नाही.',
        'प्रोसेसर थांबवला आहे — कार्य रांगेत जोडले आहे. पुढे चालू ठेवण्यासाठी प्रक्रिया पुन्हा सुरू करा.',
    ),
    'my': (
        'လုပ်ဆောင်နေဆဲ ဖြစ်သည်။ နောက်မှ ပြန်စစ်ကြည့်ပါ။ အေးဂျင့်၏ တုံ့ပြန်မှုကို အာမမခံပါ။',
        'Processor ရပ်နားထားသည် — လုပ်ငန်းကို တန်းစီထဲ ထည့်ထားသည်။ ဆက်လက်ဆောင်ရွက်ရန် လုပ်ဆောင်မှုကို ပြန်စတင်ပါ။',
    ),
    'pa': (
        'ਅਜੇ ਵੀ ਪ੍ਰੋਸੈਸਿੰਗ ਜਾਰੀ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਬਾਅਦ ਵਿੱਚ ਦੁਬਾਰਾ ਦੇਖੋ। ਏਜੰਟ ਦੇ ਜਵਾਬ ਦੀ ਗਾਰੰਟੀ ਨਹੀਂ ਹੈ।',
        'ਪ੍ਰੋਸੈਸਰ ਰੁਕਿਆ ਹੋਇਆ ਹੈ — ਕੰਮ ਕਤਾਰ ਵਿੱਚ ਜੋੜ ਦਿੱਤਾ ਗਿਆ ਹੈ। ਜਾਰੀ ਰੱਖਣ ਲਈ ਪ੍ਰੋਸੈਸਿੰਗ ਮੁੜ ਸ਼ੁਰੂ ਕਰੋ।',
    ),
    'pt': (
        'Ainda em processamento. Volte a verificar mais tarde. A resposta do agente não é garantida.',
        'Processador pausado — a tarefa foi adicionada à fila. Retome o processamento para continuar.',
    ),
    'ru': (
        'Обработка ещё продолжается. Пожалуйста, зайдите позже. Ответ агента не гарантирован.',
        'Процессор приостановлен — задача добавлена в очередь. Возобновите обработку, чтобы продолжить.',
    ),
    'sw': (
        'Bado inashughulikiwa. Tafadhali angalia tena baadaye. Hakuna uhakika kwamba wakala atajibu.',
        'Kichakataji kimesitishwa — kazi imeongezwa kwenye foleni. Anza tena uchakataji ili kuendelea.',
    ),
    'ta': (
        'இன்னும் செயலாக்கம் நடந்து வருகிறது. பின்னர் மீண்டும் பார்க்கவும். முகவரின் பதிலுக்கு உத்தரவாதம் இல்லை.',
        'செயலி இடைநிறுத்தப்பட்டுள்ளது — பணி வரிசையில் சேர்க்கப்பட்டது. தொடர, செயலாக்கத்தை மீண்டும் தொடங்கவும்.',
    ),
    'te': (
        'ఇంకా ప్రాసెస్ చేస్తోంది. దయచేసి తర్వాత మళ్లీ చూడండి. ఏజెంట్ ప్రతిస్పందనకు హామీ లేదు.',
        'ప్రాసెసర్ పాజ్ చేయబడింది — పని క్యూలో చేర్చబడింది. కొనసాగించడానికి ప్రాసెసింగ్\u200cను పునఃప్రారంభించండి.',
    ),
    'th': (
        'ยังอยู่ระหว่างการประมวลผล กรุณากลับมาตรวจสอบอีกครั้งในภายหลัง ไม่มีการรับประกันว่าเอเจนต์จะตอบกลับ',
        'ตัวประมวลผลหยุดชั่วคราว — เพิ่มงานลงในคิวแล้ว ดำเนินการประมวลผลต่อเพื่อทำงานต่อไป',
    ),
    'tr': (
        'İşlem hâlâ sürüyor. Lütfen daha sonra tekrar kontrol edin. Ajanın yanıt vermesi garanti edilmez.',
        'İşlemci duraklatıldı — görev kuyruğa eklendi. Devam etmek için işlemeyi sürdürün.',
    ),
    'uk': (
        'Обробка ще триває. Будь ласка, зайдіть пізніше. Відповідь агента не гарантована.',
        'Процесор призупинено — завдання додано до черги. Відновіть обробку, щоб продовжити.',
    ),
    'ur': (
        'ابھی پروسیسنگ جاری ہے۔ براہ کرم کچھ دیر بعد دوبارہ دیکھیں۔ ایجنٹ کے جواب کی ضمانت نہیں ہے۔',
        'پروسیسر موقوف ہے — کام قطار میں شامل کر دیا گیا ہے۔ جاری رکھنے کے لیے پروسیسنگ دوبارہ شروع کریں۔',
    ),
    'vi': (
        'Vẫn đang xử lý. Vui lòng quay lại kiểm tra sau. Phản hồi của tác nhân không được đảm bảo.',
        'Bộ xử lý đã tạm dừng — nhiệm vụ đã được thêm vào hàng đợi. Hãy tiếp tục xử lý để tiến hành.',
    ),
    'yo': (
        'Ìṣiṣẹ́ ṣì ń lọ lọ́wọ́. Ṣàyẹ̀wò padà nígbà mìíràn. A kò ṣe ìdánilójú pé aṣojú náà yóò dáhùn.',
        'Olùṣàgbékalẹ̀ ti dáwọ́ dúró — a ti fi iṣẹ́ náà sínú ìlà ìdúró. Tún ìṣiṣẹ́ bẹ̀rẹ̀ láti tẹ̀síwájú.',
    ),
    'zh': (
        '仍在处理中，请稍后再来查看。不保证智能体一定会回复。',
        '处理器已暂停 — 任务已加入队列。请恢复处理以继续。',
    ),
}

#: Durations within this fraction of each other are a ceiling, not deliberation.
DURATION_TOLERANCE = 0.02

#: Absent task identifiers as the harness writes them.
ABSENT_TASK_IDS = {"", "-", "—", "None", "null"}


class AbsentCohort(RuntimeError):
    """The cohort is the right shape and the wrong content. Do not score it."""


def _fixed_duration(durations: Sequence[float]) -> Optional[str]:
    usable = [d for d in durations if d and d > 0]
    if len(usable) < 3:
        return None
    spread = (max(usable) - min(usable)) / max(usable)
    if spread <= DURATION_TOLERANCE:
        return (
            f"{len(usable)} responses within {spread:.1%} of {statistics.median(usable):.1f}s "
            f"— a fixed ceiling, not {len(usable)} independent deliberations"
        )
    return None


def _absent_tasks(task_ids: Sequence[Any]) -> Optional[str]:
    absent = [t for t in task_ids if str(t).strip() in ABSENT_TASK_IDS]
    if absent and len(absent) == len(task_ids):
        return f"all {len(task_ids)} rows carry no task id — no task was created, so this is upstream of the LLM call"
    if absent:
        return f"{len(absent)}/{len(task_ids)} rows carry no task id"
    return None


def _still_processing(responses: Sequence[str], locale: str) -> Optional[str]:
    needles = STILL_PROCESSING.get(locale)
    if needles is None:
        return (
            f"locale {locale!r} has no registered still-processing literal — cannot rule out "
            f"timeout text scored as an answer. Register it before scoring; an unchecked locale "
            f"is not a clean one"
        )
    hits = [r for r in responses if any(n in (r or "") for n in needles)]
    if hits:
        return f"{len(hits)}/{len(responses)} responses are the still-processing literal, recorded as the agent's answer"
    return None


def assert_cohort_present(
    rows: Sequence[Dict[str, Any]],
    *,
    locale: str,
    duration_key: str = "duration_s",
    task_key: str = "task_id",
    response_key: str = "agent_response",
) -> None:
    """Raise :class:`AbsentCohort` if the cohort is absent-but-shaped.

    Checks all three signatures and reports EVERY one that fires, so a caller
    fixing the run sees the whole picture rather than the first item.
    """
    if not rows:
        raise AbsentCohort("zero rows — refusing to score an empty cohort")

    problems = [
        p
        for p in (
            _fixed_duration([r.get(duration_key) for r in rows]),  # type: ignore[arg-type]
            _absent_tasks([r.get(task_key) for r in rows]),
            _still_processing([r.get(response_key, "") for r in rows], locale),
        )
        if p
    ]
    if problems:
        raise AbsentCohort(
            f"cohort ({locale}, n={len(rows)}) is absent, not weak — it has the shape of behaviour "
            f"and none of the content:\n  - " + "\n  - ".join(problems)
        )
