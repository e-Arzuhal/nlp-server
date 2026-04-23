"""
Keyword-based classifier evaluation with a curated Turkish legal text test set.
Metrics: precision, recall, F1 per class + macro/weighted averages.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

from app.services.contract_classifier import classify_contract

logger = logging.getLogger(__name__)

# Labeled test samples: (text_snippet, expected_type)
# Covers all 7 classes with 3-5 samples each (28 total).
_TEST_SAMPLES: list[tuple[str, str]] = [
    # is_sozlesmesi
    ("İşçi, işveren ile imzalanan bu iş akdi gereği brüt maaş alacaktır.", "is_sozlesmesi"),
    ("Deneme süresi 2 ay olup fazla mesai ücret tarifesi ayrıca düzenlenmiştir.", "is_sozlesmesi"),
    ("SGK primleri işveren tarafından ödenecek; yıllık izin süresi 14 gündür.", "is_sozlesmesi"),
    ("Kıdem tazminatı ve ihbar süresi Türk İş Kanunu hükümlerine tabidir.", "is_sozlesmesi"),
    ("İşçinin çalışma saatleri haftalık 45 saati geçemez; ücret bankaya yatırılır.", "is_sozlesmesi"),

    # kira_sozlesmesi
    ("Kiracı, kiraya veren ile belirlenen kira bedelini her ayın 5'inde ödeyecektir.", "kira_sozlesmesi"),
    ("Depozito olarak iki aylık kira bedeli alınmış; tahliye halinde iade edilecektir.", "kira_sozlesmesi"),
    ("Konut olarak kullanılacak olan işbu kira süresi 1 yıldır.", "kira_sozlesmesi"),
    ("Kira artışı her yıl TÜİK açıklanan TÜFE oranında uygulanacaktır.", "kira_sozlesmesi"),
    ("Aidat ve ortak giderler kiracıya aittir; işyeri kirası için KDV eklenir.", "kira_sozlesmesi"),

    # satis_sozlesmesi
    ("Satıcı mülkiyeti alıcıya tapu devri yoluyla devredecektir.", "satis_sozlesmesi"),
    ("Satış bedeli nakit olarak ödenecek; teslim tarihi sözleşme imzasından 30 gün sonradır.", "satis_sozlesmesi"),
    ("Alıcı, satın alma bedelini ödeme planına göre taksit taksit ödeyecektir.", "satis_sozlesmesi"),
    ("Mülkiyet devri tapuda gerçekleşecek olup satıcı tüm borçları temizleyecektir.", "satis_sozlesmesi"),

    # hizmet_sozlesmesi
    ("Hizmet sağlayıcı danışmanlık hizmeti verecek; fatura her ayın sonunda kesilecektir.", "hizmet_sozlesmesi"),
    ("Serbest meslek makbuzu ile proje bazlı freelance çalışma kararlaştırılmıştır.", "hizmet_sozlesmesi"),
    ("KDV dahil hizmet bedeli aylık 50.000 TL'dir.", "hizmet_sozlesmesi"),
    ("Proje tesliminde eksiklik olursa hizmet bedeli %10 kesilir.", "hizmet_sozlesmesi"),

    # vekaletname
    ("Müvekkil, vekiline noter huzurunda yetki vermiştir; temsil yetkisi kapsamlıdır.", "vekaletname"),
    ("Vekalet kapsamında tüm hukuki işlemler vekaleten yürütülecektir.", "vekaletname"),
    ("İşbu vekaletname, noterlikçe düzenlenmiş olup vekil her türlü işlemi temsilen yapabilir.", "vekaletname"),
    ("Müvekkil, vekile gayrimenkul satış için geniş yetki vermiştir.", "vekaletname"),

    # taahhutname
    ("Taahhüt eder ki söz konusu borcu 3 ay içinde ödeyecektir.", "taahhutname"),
    ("Beyan eder ve taahhüt ederim; belirlenen şartlara uymayı kabul ediyorum.", "taahhutname"),
    ("İşbu taahhütname ile yükümlülük altına girdiğimi kabul ve beyan ederim.", "taahhutname"),

    # kefalet_sozlesmesi
    ("Kefil olduğumu beyan ederim; müteselsil kefalet hükümleri geçerlidir.", "kefalet_sozlesmesi"),
    ("Borçlunun borcunu güvence altına almak için kefil sıfatıyla kefalet veriyorum.", "kefalet_sozlesmesi"),
    ("Müteselsil kefil olarak kefalet bedelini ödemekle yükümlüyüm.", "kefalet_sozlesmesi"),
]


def evaluate() -> Dict[str, Any]:
    """Run classifier on test set; return per-class + macro/weighted metrics."""
    labels = sorted({expected for _, expected in _TEST_SAMPLES})
    label_index = {l: i for i, l in enumerate(labels)}
    n = len(labels)

    tp = [0] * n
    fp = [0] * n
    fn = [0] * n

    predictions: list[tuple[str | None, str]] = []
    for text, expected in _TEST_SAMPLES:
        predicted, _ = classify_contract(text)
        predictions.append((predicted, expected))

        exp_i = label_index[expected]
        if predicted == expected:
            tp[exp_i] += 1
        else:
            fn[exp_i] += 1
            if predicted and predicted in label_index:
                fp[label_index[predicted]] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    support = [0] * n
    for _, expected in _TEST_SAMPLES:
        support[label_index[expected]] += 1

    for i, label in enumerate(labels):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        rec = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support[i],
        }

    total = len(_TEST_SAMPLES)
    correct = sum(tp)
    accuracy = round(correct / total, 4) if total > 0 else 0.0

    macro_prec = round(sum(v["precision"] for v in per_class.values()) / n, 4)
    macro_rec = round(sum(v["recall"] for v in per_class.values()) / n, 4)
    macro_f1 = round(sum(v["f1"] for v in per_class.values()) / n, 4)

    weighted_prec = round(sum(per_class[l]["precision"] * support[i] for i, l in enumerate(labels)) / total, 4)
    weighted_rec = round(sum(per_class[l]["recall"] * support[i] for i, l in enumerate(labels)) / total, 4)
    weighted_f1 = round(sum(per_class[l]["f1"] * support[i] for i, l in enumerate(labels)) / total, 4)

    return {
        "accuracy": accuracy,
        "total_samples": total,
        "correct": correct,
        "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
        "weighted": {"precision": weighted_prec, "recall": weighted_rec, "f1": weighted_f1},
        "per_class": per_class,
        "classifier": "keyword_overlap",
        "note": "Evaluated on 28-sample curated Turkish legal text test set (7 classes).",
    }


# Cache result at import time so /metrics is O(1)
_cached_metrics: Dict[str, Any] | None = None


def get_cached_metrics() -> Dict[str, Any]:
    global _cached_metrics
    if _cached_metrics is None:
        _cached_metrics = evaluate()
        logger.info("classifier_metrics_computed", extra={
            "accuracy": _cached_metrics["accuracy"],
            "macro_f1": _cached_metrics["macro"]["f1"],
        })
    return _cached_metrics
